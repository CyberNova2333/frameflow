#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""End-to-end parity verification: PyTorch GameFINet (FrameFlowNet) vs WebGPU runtime.

1. Instantiates both `std` (GameFI-Std, 775K) and `lite` (GameFI-Lite, 222K).
2. Initializes residual heads with small deterministic non-zero weights so every op path is exercised.
3. Exports each preset to `(graph.json, model.json, model.bin)`.
4. Runs PyTorch forward on a real 960x544 game frame pair at both t=0.5 and t=1/3.
5. Invokes headless Edge 157 WebGPU (`verify_parity.mjs`) and checks pixel PSNR (>= 50 dB).
"""
from __future__ import annotations

import json
import math
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
    # BGR [1, 3, H, W] -> RGB [H, W, 3] uint8
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
        # Give all heads small deterministic non-zero weights so every single op path is exercised!
        for name, p in model.named_parameters():
            if any(k in name for k in ("coarse.head", "fine.out", "film.out", "refine.q4")):
                if p.ndim >= 2:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    torch.nn.init.normal_(p, mean=0.0, std=0.01)


def main() -> None:
    pv_root = ROOT.parent / "pvsspw"
    frame_a = pv_root / "artifacts/flyhigh-proto/intel-v2/run_pair9_A.png"
    frame_b = pv_root / "artifacts/flyhigh-proto/intel-v2/run_pair9_B.png"
    if not frame_a.exists() or not frame_b.exists():
        raise FileNotFoundError(f"Missing real game test frames: {frame_a}")

    a_t = load_bgr_tensor(frame_a)
    b_t = load_bgr_tensor(frame_b)

    parity_dir = ROOT / "exports/parity-check"
    parity_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for idx, preset in enumerate(["std", "lite"]):
        model = FrameFlowNet(preset).eval()
        init_nontrivial(model, seed=42 + idx)
        exp_dir = parity_dir / preset
        info = export_model(model, exp_dir, train_step=1, stem="model")
        print(f"[parity] Exported {preset}: {info['n_tensors']} tensors, {info['n_params']:,} params, {info['bin_bytes']/1024:.1f} KB")

        # Run PyTorch golden outputs at t=0.5 and t=1/3 (matching f16 weights as exported!)
        # Reload exact f16-quantized weights back into PyTorch to compare cleanly against WebGPU
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.float16).to(torch.float32))
            for t_name, t_val in [("t050", 0.5), ("t033", 1.0 / 3.0)]:
                # Browser FhSession enables still_test=True (decisive bit 1) by default
                out = model(a_t, b_t, t=t_val, k=1, still_test=True)
                rgb = tensor_to_u8_rgb(out["render"])
                gold_path = exp_dir / f"pytorch_{t_name}.png"
                Image.fromarray(rgb).save(gold_path)
                jobs.append({
                    "preset": preset,
                    "t_name": t_name,
                    "t": t_val,
                    "k": 1,
                    "dir": str(exp_dir),
                    "sampling": model.cfg.render_sampling,
                    "gold": str(gold_path),
                    "gpu_out": str(exp_dir / f"webgpu_{t_name}.png"),
                })

    jobs_path = parity_dir / "jobs.json"
    jobs_path.write_text(json.dumps({"frameA": str(frame_a), "frameB": str(frame_b), "jobs": jobs}, indent=2) + "\n")

    print("[parity] Running headless Edge 157 WebGPU parity check on Intel Xe-LPG...")
    cmd = ["node", str(ROOT / "scripts/verify_parity.mjs"), str(jobs_path)]
    env = __import__("os").environ.copy()
    env.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/intel_icd.json")
    subprocess.run(cmd, check=True, env=env)

    print("\n=== PyTorch vs WebGPU Numerical Parity Results ===")
    all_ok = True
    gpu_report = json.loads((parity_dir / "webgpu_report.json").read_text())
    for job, gres in zip(jobs, gpu_report["results"]):
        gold = np.asarray(Image.open(job["gold"]).convert("RGB"))
        gpu = np.asarray(Image.open(job["gpu_out"]).convert("RGB"))
        p_val, mad, max_e = psnr_u8(gold, gpu)
        ok = p_val >= 50.0
        all_ok = all_ok and ok
        print(
            f"  {job['preset']:8s} {job['t_name']} (k={job['k']}, {job['sampling']:8s}) "
            f"| PSNR={p_val:5.2f} dB | MAD={mad:.3f} | MaxErr={max_e:3d} "
            f"| Intel GPU(full 960x544) pair={gres['pairFullMs']}ms mid={gres['midFullMs']}ms "
            f"| Intel GPU(int 480x288) pair={gres['pair480Ms']}ms mid={gres['mid480Ms']}ms "
            f"| {'PASS' if ok else 'FAIL'}"
        )

    if not all_ok:
        sys.exit(1)
    print("\nALL PARITY CHECKS PASSED (PSNR >= 50 dB on Standard & Lite, t=0.5 & t=1/3)!")


if __name__ == "__main__":
    main()
