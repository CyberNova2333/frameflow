#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Train GameSR-2x (GameSR-Std & GameSR-Lite) on clean open-licensed frames + game UI degradations."""
from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameflow.sr_config import get_sr_preset
from frameflow.sr_dataset import GameSRDataset, ValidationSRDataset
from frameflow.sr_export import export_sr_model
from frameflow.sr_losses import GameSRLoss
from frameflow.sr_model import GameSRNet


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - gt.float()) ** 2).item()
    return 99.0 if mse < 1e-10 else -10.0 * math.log10(mse)


@torch.no_grad()
def update_ema(ema: GameSRNet, model: GameSRNet, decay: float = 0.999) -> None:
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.mul_(decay).add_(mp, alpha=1.0 - decay)


@torch.no_grad()
def run_validation(
    model: GameSRNet,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    sr_psnrs: list[float] = []
    bil_psnrs: list[float] = []
    for batch in val_loader:
        lr = batch["LR"].to(device, non_blocking=True)
        hr = batch["HR"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp_dtype is not None and device.type == "cuda"),
        ):
            out = model(lr)
            pred = out["render"].float()
            base = out["base_hr"].float()
        for i in range(pred.shape[0]):
            sr_psnrs.append(psnr(pred[i : i + 1], hr[i : i + 1]))
            bil_psnrs.append(psnr(base[i : i + 1], hr[i : i + 1]))

    mean_sr = sum(sr_psnrs) / max(1, len(sr_psnrs))
    mean_bil = sum(bil_psnrs) / max(1, len(bil_psnrs))
    return {
        "val_psnr": round(mean_sr, 3),
        "val_bilinear_psnr": round(mean_bil, 3),
        "val_gain_db": round(mean_sr - mean_bil, 3),
        "val_n": len(sr_psnrs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="sr_std")
    ap.add_argument("--data-root", default="data/sr_data")
    ap.add_argument("--out", default="checkpoints/sr_std")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=4000)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    cfg = get_sr_preset(args.preset)
    model = GameSRNet(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location="cpu", weights_only=False)
        t_cfg = get_sr_preset(ck.get("preset", "sr_std"))
        teacher = GameSRNet(t_cfg).to(device).eval()
        teacher.load_state_dict(ck.get("ema", ck["model"]))
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[train-sr] Loaded frozen teacher {t_cfg.name} ({teacher.param_count():,} params) from {args.teacher}")

    data_root = Path(args.data_root)
    train_root = data_root / "train"
    val_root = data_root / "val"
    sources: list[tuple[Path, float]] = []
    for sub, w in [("open_media", 0.45), ("synla", 0.35), ("ui_synth", 0.20)]:
        p = train_root / sub
        if p.exists():
            sources.append((p, w))
    if not sources and train_root.exists():
        sources.append((train_root, 1.0))

    epoch_samples = max(args.batch * 256, 8192)
    ds = GameSRDataset(sources, hr_crop=args.crop, scale=cfg.scale, ui_prob=0.10 if any(p.name == "ui_synth" for p, _ in sources) else 0.25, samples_per_epoch=epoch_samples)
    print(
        f"[train-sr] Dataset buckets: "
        f"{[(b[0][0].parent.name if b[0] else 'empty', len(b[0]), round(w, 2)) for b, w in zip(ds.buckets, ds.norm_weights)]}"
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=(4 if args.workers > 0 else None),
    )

    val_dirs = [val_root / s for s in ("open_media", "synla", "ui_synth") if (val_root / s).exists()] or ([val_root] if val_root.exists() else [])
    vds = ValidationSRDataset(val_dirs, hr_crop=args.crop, scale=cfg.scale, max_samples=96)
    val_loader = DataLoader(vds, batch_size=min(16, args.batch), shuffle=False, num_workers=min(2, args.workers))
    print(f"[train-sr] Validation set: {len(vds)} pairs from {[d.name for d in val_dirs]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    warmup_steps = min(800, max(20, args.steps // 20))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        prog = (step_idx - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = GameSRLoss().to(device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    start_step = 0
    best_psnr = -1.0
    resume_path = out_dir / "latest.pt" if args.auto_resume and (out_dir / "latest.pt").exists() else None
    if resume_path is not None:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        start_step = int(ck.get("step", 0))
        best_psnr = float(ck.get("best_psnr", -1.0))
        print(f"[train-sr] Resumed from {resume_path} at step {start_step} (best_psnr={best_psnr:.2f} dB)")

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))

    stop_requested = False

    def handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[train-sr] Signal {signum} received; saving checkpoint after current step...", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def save_checkpoint(step_num: int, is_best: bool = False) -> None:
        payload = {
            "preset": args.preset,
            "step": step_num,
            "best_psnr": best_psnr,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
        }
        torch.save(payload, out_dir / "latest.pt")
        exp = export_sr_model(ema, out_dir / "export", train_step=step_num)
        if is_best:
            torch.save(payload, out_dir / "best.pt")
            export_sr_model(ema, out_dir / "export_best", train_step=step_num)
        print(
            f"[train-sr] Saved step={step_num} ({exp['n_params']:,} params, {exp['bin_bytes']/1024:.1f} KB)"
            + (" [NEW BEST]" if is_best else ""),
            flush=True,
        )

    def log_jsonl(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(
        f"[train-sr] {cfg.name} ({model.param_count():,} params) device={device} "
        f"amp={args.amp} batch={args.batch} crop={args.crop} steps={args.steps} out={out_dir}",
        flush=True,
    )

    step = start_step
    t0 = time.time()
    running: dict[str, float] = {}
    running_n = 0

    while step < args.steps and not stop_requested:
        for batch in dl:
            if step >= args.steps or stop_requested:
                break
            model.train()
            lr_img = batch["LR"].to(device, non_blocking=True)
            hr_img = batch["HR"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(amp_dtype is not None and device.type == "cuda"),
            ):
                t_out = None
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(lr_img)
                out = model(lr_img)
                loss, metrics = criterion(out, hr_img, teacher_out=t_out)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

            sched.step()
            update_ema(ema, model, decay=0.995 if step < 2000 else 0.999)
            step += 1

            metrics["psnr"] = psnr(out["render"].detach(), hr_img)
            for mk, mv in metrics.items():
                running[mk] = running.get(mk, 0.0) + mv
            running_n += 1

            if step % args.log_every == 0 or step == args.steps:
                dt = max(1e-3, time.time() - t0)
                avg = {k: round(v / running_n, 4) for k, v in running.items()}
                lr_now = opt.param_groups[0]["lr"]
                ips = round(running_n / dt, 2)
                print(
                    f"step {step:6d}/{args.steps} | lr={lr_now:.2e} | loss={avg['total']:.4f} "
                    f"l1={avg['l1']:.4f} cen={avg['census']:.4f} psnr={avg['psnr']:.2f}dB | {ips:.1f} it/s",
                    flush=True,
                )
                log_jsonl({"kind": "train", "step": step, "lr": lr_now, "it_s": ips, **avg})
                running.clear()
                running_n = 0
                t0 = time.time()

            if step % args.val_every == 0 or step == args.steps:
                vstat = run_validation(ema, val_loader, device, amp_dtype)
                is_best = vstat["val_psnr"] > best_psnr
                if is_best:
                    best_psnr = vstat["val_psnr"]
                print(
                    f"  [val] step={step} | PSNR={vstat['val_psnr']:.2f}dB "
                    f"(bilinear={vstat['val_bilinear_psnr']:.2f}dB, gain=+{vstat['val_gain_db']:.2f}dB)"
                    + (" *" if is_best else ""),
                    flush=True,
                )
                log_jsonl({"kind": "val", "step": step, **vstat})
                if is_best:
                    save_checkpoint(step, is_best=True)

            if step % args.save_every == 0 or step == args.steps or stop_requested:
                save_checkpoint(step, is_best=False)


if __name__ == "__main__":
    main()
