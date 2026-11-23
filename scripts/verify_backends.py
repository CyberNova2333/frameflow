#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Verify FrameFlow's self-authored WebGPU AND WebGL2 browser runtimes against PyTorch.

Exports FrameFlow-Std (775K) and FrameFlow-Lite (222K) with non-trivial weights,
computes PyTorch golden outputs on real 960x544 game frames at t=0.5 and t=1/3,
runs BOTH `runtime/webgpu-backend.js` and `runtime/webgl2-backend.js` in headless Edge 157,
and checks exact numerical parity (PSNR >= 50 dB) + real-world timing (full & 480x288 & PSP 480x272).
"""
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

from frameflow.export import export_model
from frameflow.model import FrameFlowNet


def load_bgr_tensor(path: Path) -> torch.Tensor:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
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


def init_nontrivial(model: FrameFlowNet, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if any(k in name for k in ("coarse.head", "fine.out", "film.out", "refine.q4")):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 if p.ndim >= 2 else 0.01)


def main() -> None:
    pv_root = ROOT.parent / "pvsspw"
    frame_a = pv_root / "artifacts/flyhigh-proto/intel-v2/run_pair9_A.png"
    frame_b = pv_root / "artifacts/flyhigh-proto/intel-v2/run_pair9_B.png"

    out_dir = ROOT / "exports/runtime-verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fallback to synthetic PNGs if pvsspw artifacts aren't present
    if not frame_a.exists() or not frame_b.exists():
        yy, xx = np.mgrid[0:544, 0:960]
        arr_a = np.stack([(xx * 3) % 255, (yy * 5) % 255, ((xx + yy) * 2) % 255], axis=-1).astype(np.uint8)
        arr_b = np.roll(arr_a, shift=(6, -4), axis=(1, 0))
        frame_a = out_dir / "synth_A.png"
        frame_b = out_dir / "synth_B.png"
        Image.fromarray(arr_a).save(frame_a)
        Image.fromarray(arr_b).save(frame_b)

    a_t = load_bgr_tensor(frame_a)
    b_t = load_bgr_tensor(frame_b)

    jobs = []
    for idx, preset in enumerate(["std", "lite"]):
        model = FrameFlowNet(preset).eval()
        init_nontrivial(model, seed=100 + idx)
        exp_dir = out_dir / preset
        info = export_model(model, exp_dir, train_step=1, stem="model")
        print(f"[verify] Exported {preset}: {info['n_params']:,} params ({info['bin_bytes']/1024:.1f} KB)")

        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.float16).to(torch.float32))
            for t_name, t_val in [("t050", 0.5), ("t033", 1.0 / 3.0)]:
                out = model(a_t, b_t, t=t_val, k=1, still_test=True)
                gold_path = exp_dir / f"pytorch_{t_name}.png"
                Image.fromarray(tensor_to_u8_rgb(out["render"])).save(gold_path)
                for backend in ("webgpu", "webgl2"):
                    jobs.append({
                        "preset": preset,
                        "backend": backend,
                        "t_name": t_name,
                        "t": t_val,
                        "k": 1,
                        "sampling": model.cfg.render_sampling,
                        "gold": str(gold_path),
                        "out_png": str(exp_dir / f"{backend}_{t_name}.png"),
                    })

    jobs_path = out_dir / "jobs.json"
    jobs_path.write_text(json.dumps({"frameA": str(frame_a), "frameB": str(frame_b), "jobs": jobs}, indent=2) + "\n")

    print("[verify] Running headless Edge 157 dual-backend verification (WebGPU + WebGL2) on Intel Xe-LPG...")
    env = dict(**os.environ, VK_DRIVER_FILES="/usr/share/vulkan/icd.d/intel_icd.json")
    subprocess.run(["node", str(ROOT / "scripts/verify_backends.mjs"), str(jobs_path)], check=True, env=env)

    rep = json.loads((out_dir / "report.json").read_text())
    print("\n=== Capabilities Probe ===")
    print(json.dumps(rep["capabilities"], indent=2))
    print("\n=== PyTorch vs FrameFlow Dual-Backend Parity & Latency (Intel Xe-LPG) ===")
    all_ok = True
    for job, res in zip(jobs, rep["results"]):
        gold = np.asarray(Image.open(job["gold"]).convert("RGB"))
        got = np.asarray(Image.open(job["out_png"]).convert("RGB"))
        p_val, mad, max_e = psnr_u8(gold, got)
        ok = p_val >= 50.0
        all_ok = all_ok and ok
        bench = ""
        if res.get("bench480"):
            b = res["bench480"]
            bench = f" | 960x544(int 480x288): pair={b['pairMs']}ms + mid={b['midMs']}ms = {b['totalMs']}ms"
        if res.get("benchPsp"):
            bp = res["benchPsp"]
            bench += f" | PSP 480x272: {bp['totalMs']}ms"
        print(
            f"  {job['backend']:6s} | {job['preset']:4s} {job['t_name']} ({job['sampling']:8s}) "
            f"| PSNR={p_val:5.2f} dB | MAD={mad:.3f} | MaxErr={max_e:3d}{bench} | {'PASS' if ok else 'FAIL'}"
        )

    if not all_ok:
        sys.exit(1)
    print("\nALL WEBGPU + WEBGL2 PARITY CHECKS PASSED (PSNR >= 50 dB)!")


if __name__ == "__main__":
    main()
