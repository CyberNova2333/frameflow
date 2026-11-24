#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Verify GameSR-2x WebGPU AND WebGL2 browser runtimes against PyTorch + benchmark Intel Xe-LPG latency."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameflow.sr_export import export_sr_model
from frameflow.sr_model import GameSRNet


def load_bgr_tensor(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.Resampling.BILINEAR)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    bgr = rgb[:, :, ::-1].copy()
    return torch.from_numpy(bgr.transpose(2, 0, 1)).unsqueeze(0)


def tensor_to_u8_rgb(x: torch.Tensor) -> np.ndarray:
    bgr = (x[0].detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return bgr[:, :, ::-1].copy()


def psnr_u8(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    mad = float(np.mean(np.abs(diff)))
    max_e = int(np.max(np.abs(diff)))
    val = 99.0 if mse < 1e-10 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return val, mad, max_e


def init_nontrivial(model: GameSRNet, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "head.weight":
                torch.nn.init.normal_(p, mean=0.0, std=0.05)
            elif name == "head.bias":
                torch.nn.init.normal_(p, mean=0.0, std=0.01)


def main() -> None:
    pv_root = ROOT.parent / "pvsspw"
    src_960 = pv_root / "artifacts/flyhigh-proto/intel-v2/run_pair9_A.png"
    out_dir = ROOT / "exports/sr-runtime-verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_960.exists():
        yy, xx = np.mgrid[0:544, 0:960]
        arr = np.stack([(xx * 3) % 255, (yy * 5) % 255, ((xx + yy) * 2) % 255], axis=-1).astype(np.uint8)
        src_960 = out_dir / "synth_960.png"
        Image.fromarray(arr).save(src_960)

    src_480 = out_dir / "frame_480x272.png"
    Image.open(src_960).convert("RGB").resize((480, 272), Image.Resampling.BILINEAR).save(src_480)
    lr_480 = load_bgr_tensor(src_480)

    jobs = []
    for idx, preset in enumerate(["sr_lite", "sr_std"]):
        model = GameSRNet(preset).eval()
        init_nontrivial(model, seed=300 + idx)
        exp_dir = out_dir / preset
        info = export_sr_model(model, exp_dir, train_step=1, stem="model")
        print(f"[verify-sr] Exported {preset}: {info['n_params']:,} params ({info['bin_bytes']/1024:.1f} KB)")

        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.float16).to(torch.float32))
            out_480 = model(lr_480)["render"]
            gold_path = exp_dir / "pytorch_960x544.png"
            Image.fromarray(tensor_to_u8_rgb(out_480)).save(gold_path)

        for backend in ("webgpu", "webgl2"):
            jobs.append({
                "preset": preset,
                "backend": backend,
                "gold": str(gold_path),
                "out_png": str(exp_dir / f"{backend}_960x544.png"),
            })

    jobs_path = out_dir / "jobs.json"
    jobs_path.write_text(json.dumps({"frame480": str(src_480), "frame960": str(src_960), "jobs": jobs}, indent=2) + "\n")

    print("[verify-sr] Running headless Edge 157 dual-backend verification (WebGPU + WebGL2) on Intel Xe-LPG...")
    env = os.environ.copy()
    env.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/intel_icd.json")
    subprocess.run(["node", str(ROOT / "scripts/verify_sr_backends.mjs"), str(jobs_path)], check=True, env=env)

    rep = json.loads((out_dir / "report.json").read_text())
    print("\n=== PyTorch vs GameSR-2x Dual-Backend Parity & Latency (Intel Xe-LPG) ===")
    all_ok = True
    for job, res in zip(jobs, rep["results"]):
        gold = np.asarray(Image.open(job["gold"]).convert("RGB"))
        got = np.asarray(Image.open(job["out_png"]).convert("RGB"))
        p_val, mad, max_e = psnr_u8(gold, got)
        ok = p_val >= 50.0
        all_ok = all_ok and ok
        b480 = res.get("ms480")
        b960 = res.get("ms960")
        print(
            f"  {job['backend']:6s} | {job['preset']:7s} ({res['engineInfo'].get('precision','f16')}) "
            f"| PSNR={p_val:5.2f} dB | MAD={mad:.3f} | MaxErr={max_e:3d} "
            f"| 480->960={b480}ms | 960->1920={b960}ms | {'PASS' if ok else 'FAIL'}"
        )

    if not all_ok:
        sys.exit(1)
    print("\nALL GAMESR-2X WEBGPU + WEBGL2 PARITY CHECKS PASSED (PSNR >= 50 dB)!")


if __name__ == "__main__":
    main()
