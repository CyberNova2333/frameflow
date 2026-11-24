#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Train GameFI-Std (775K) or distill GameFI-Lite (222K) on local GPU.

Production-ready features:
  - Mixed precision bf16 (ideal for RTX 50-series Blackwell)
  - Multi-source weighted dataset mixing (`--data dir1 dir2 --data-weights 0.75 0.25`)
  - Deterministic validation loop (`--val-data`) tracking best PSNR & saving `best.pt` + `export_best/`
  - Knowledge distillation (`--preset lite --teacher checkpoints/std/best.pt`)
  - Graceful SIGINT/SIGTERM checkpoint saving + `--auto-resume`
  - Structured JSONL metrics log (`<out>/metrics.jsonl`)
"""
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frameflow.config import get_preset
from frameflow.dataset import (
    SequenceFrameDataset,
    SyntheticSmokeDataset,
    ValidationTripletDataset,
    WeightedMixDataset,
)
from frameflow.export import export_model
from frameflow.losses import FrameFlowLoss
from frameflow.model import FrameFlowNet


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - gt.float()) ** 2).item()
    return 99.0 if mse < 1e-10 else -10.0 * math.log10(mse)


@torch.no_grad()
def update_ema(ema: FrameFlowNet, model: FrameFlowNet, decay: float = 0.999) -> None:
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.mul_(decay).add_(mp, alpha=1.0 - decay)


@torch.no_grad()
def run_validation(
    model: FrameFlowNet,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    psnr_all: list[float] = []
    psnr_half: list[float] = []
    psnr_arb: list[float] = []
    hud_l1: list[float] = []

    for batch in val_loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        gt = batch["GT"].to(device, non_blocking=True)
        t = batch["t"].to(device, non_blocking=True)
        hm = batch["hud_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp_dtype is not None and device.type == "cuda"),
        ):
            out = model(a, b, t)
            pred = out["render"].float()

        for i in range(pred.shape[0]):
            p = psnr(pred[i : i + 1], gt[i : i + 1])
            psnr_all.append(p)
            if abs(float(t[i]) - 0.5) < 1e-3:
                psnr_half.append(p)
            else:
                psnr_arb.append(p)
            if hm[i].sum() > 0:
                err = ((pred[i] - gt[i]).abs() * hm[i]).sum() / (hm[i].sum() * 3.0 + 1e-6)
                hud_l1.append(float(err.item()) * 255.0)

    def avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "val_psnr": avg(psnr_all),
        "val_psnr_t05": avg(psnr_half),
        "val_psnr_arb": avg(psnr_arb),
        "val_hud_mad255": avg(hud_l1),
        "val_n": len(psnr_all),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="std", choices=["std", "standard", "lite"])
    ap.add_argument("--out", default="checkpoints/std")
    ap.add_argument("--data", nargs="*", default=[], help="One or more clip directories")
    ap.add_argument("--data-weights", nargs="*", type=float, default=[], help="Optional sampling weights matching --data")
    ap.add_argument("--val-data", nargs="*", default=[], help="Validation clip directories")
    ap.add_argument("--synthetic", action="store_true", help="Use procedural smoke dataset")
    ap.add_argument("--teacher", default=None, help="Path to GameFI-Std checkpoint (.pt) for distillation")
    ap.add_argument("--resume", default=None, help="Explicit checkpoint path to resume from")
    ap.add_argument("--auto-resume", action="store_true", help="Automatically resume from <out>/latest.pt if present")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hud-prob", type=float, default=0.35)
    ap.add_argument("--k-train", type=int, default=None, help="Override K iterations during training")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--val-samples", type=int, default=96)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    cfg = get_preset(args.preset)
    model = FrameFlowNet(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location="cpu", weights_only=False)
        t_cfg = get_preset(ck.get("preset", "std"))
        teacher = FrameFlowNet(t_cfg).to(device).eval()
        teacher.load_state_dict(ck.get("ema", ck["model"]))
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[train] Loaded frozen teacher ({t_cfg.name}, {teacher.param_count():,} params) from {args.teacher}")

    epoch_samples = max(args.batch * 256, 4096)
    if args.synthetic or not args.data:
        ds = SyntheticSmokeDataset(length=epoch_samples, crop_size=args.crop, hud_prob=args.hud_prob)
        print(f"[train] Using SyntheticSmokeDataset (crop={args.crop}, hud_prob={args.hud_prob})")
    else:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        valid_pairs = []
        for idx_d, d in enumerate(args.data):
            p = Path(d)
            if not p.exists():
                continue
            has_imgs = any(f.is_file() and f.suffix.lower() in exts for f in p.rglob("*"))
            if has_imgs:
                w = args.data_weights[idx_d] if idx_d < len(args.data_weights) else 1.0
                valid_pairs.append((p, w))
        if not valid_pairs:
            raise FileNotFoundError(f"None of --data paths contain image frames: {args.data}")
        valid_roots = [p for p, _ in valid_pairs]
        valid_weights = [w for _, w in valid_pairs]
        if args.data_weights and len(valid_roots) > 1:
            sub_ds = [
                SequenceFrameDataset([r], crop_size=args.crop, hud_prob=args.hud_prob, samples_per_epoch=epoch_samples)
                for r in valid_roots
            ]
            ds = WeightedMixDataset(sub_ds, valid_weights, samples_per_epoch=epoch_samples)
            print(f"[train] WeightedMixDataset: {[(str(r), len(s.clips), w) for r, s, w in zip(valid_roots, sub_ds, valid_weights)]}")
        else:
            ds = SequenceFrameDataset(valid_roots, crop_size=args.crop, hud_prob=args.hud_prob, samples_per_epoch=epoch_samples)
            print(f"[train] Loaded {len(ds.clips)} clips ({len(ds.index)} base frames) from {[str(r) for r in valid_roots]}")

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

    val_loader = None
    val_roots = [Path(d) for d in args.val_data if Path(d).exists()]
    if val_roots:
        vds = ValidationTripletDataset(val_roots, crop_size=args.crop, max_samples=args.val_samples)
        val_loader = DataLoader(vds, batch_size=min(8, args.batch), shuffle=False, num_workers=min(2, args.workers))
        print(f"[train] Validation enabled: {len(vds)} deterministic triplets from {[str(r) for r in val_roots]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    warmup_steps = min(1000, max(20, args.steps // 20))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        prog = (step_idx - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = FrameFlowLoss().to(device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    start_step = 0
    best_psnr = -1.0
    resume_path = Path(args.resume) if args.resume else (out_dir / "latest.pt" if args.auto_resume and (out_dir / "latest.pt").exists() else None)
    if resume_path is not None and resume_path.exists():
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
        print(f"[train] Resumed from {resume_path} at step {start_step} (best_psnr={best_psnr:.2f} dB)")

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))

    stop_requested = False

    def handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[train] Signal {signum} received; saving clean checkpoint after current step...", flush=True)

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
        exp = export_model(ema, out_dir / "export", train_step=step_num)
        if is_best:
            torch.save(payload, out_dir / "best.pt")
            export_model(ema, out_dir / "export_best", train_step=step_num)
        print(
            f"[train] Saved step={step_num} ({exp['n_params']:,} params, {exp['bin_bytes']/1024:.1f} KB)"
            + (" [NEW BEST]" if is_best else ""),
            flush=True,
        )

    def log_jsonl(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(
        f"[train] {cfg.name} ({model.param_count():,} params) device={device} "
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
            a = batch["A"].to(device, non_blocking=True)
            b = batch["B"].to(device, non_blocking=True)
            gt = batch["GT"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True)
            hud_mask = batch["hud_mask"].to(device, non_blocking=True)

            k_cur = args.k_train if args.k_train is not None else (
                1 if cfg.k_default == 1 else (1 if (step % 3 == 0) else cfg.k_default)
            )

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(amp_dtype is not None and device.type == "cuda"),
            ):
                t_out = None
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(a, b, t)
                out = model(a, b, t, k=k_cur)
                loss, metrics = criterion(out, a, b, gt, t, hud_mask=hud_mask, teacher_out=t_out)

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

            metrics["psnr"] = psnr(out["render"].detach(), gt)
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
                    f"l1={avg['l1']:.4f} warp={avg['warp']:.4f} psnr={avg['psnr']:.2f}dB | {ips:.1f} it/s",
                    flush=True,
                )
                log_jsonl({"kind": "train", "step": step, "lr": lr_now, "it_s": ips, **avg})
                running.clear()
                running_n = 0
                t0 = time.time()

            if val_loader is not None and (step % args.val_every == 0 or step == args.steps):
                vstat = run_validation(ema, val_loader, device, amp_dtype)
                is_best = vstat["val_psnr"] > best_psnr
                if is_best:
                    best_psnr = vstat["val_psnr"]
                print(
                    f"  [val] step={step} | PSNR={vstat['val_psnr']:.2f}dB "
                    f"(t=0.5: {vstat['val_psnr_t05']:.2f}dB, arb-t: {vstat['val_psnr_arb']:.2f}dB) "
                    f"| HUD MAD={vstat['val_hud_mad255']:.2f}/255"
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
