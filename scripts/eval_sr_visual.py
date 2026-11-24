#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Evaluate trained GameSR-2x models visually & quantitatively vs Nearest/Bicubic/Sharpen baselines."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameflow.sr_config import get_sr_preset
from frameflow.sr_dataset import ValidationSRDataset
from frameflow.sr_model import GameSRNet


def load_sr_ckpt(ckpt_path: Path, preset: str) -> GameSRNet:
    model = GameSRNet(get_sr_preset(preset)).eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("ema", ck["model"]))
    return model


def run_sr_rgb(model: GameSRNet, rgb_u8: np.ndarray) -> np.ndarray:
    bgr = rgb_u8[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
    t = torch.from_numpy(np.transpose(bgr, (2, 0, 1)).copy()).unsqueeze(0)
    with torch.no_grad():
        out = model(t)["render"][0].clamp(0.0, 1.0).numpy()
    rgb = (np.transpose(out, (1, 2, 0))[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8)
    return rgb


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-10 else 10.0 * math.log10((255.0 * 255.0) / mse)


def make_strip(tiles: list[tuple[str, np.ndarray]]) -> Image.Image:
    h, w, _ = tiles[0][1].shape
    pad = 8
    header = 26
    canvas = Image.new("RGB", (len(tiles) * w + (len(tiles) + 1) * pad, h + header + pad), (20, 22, 28))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(tiles):
        x0 = pad + idx * (w + pad)
        canvas.paste(Image.fromarray(arr, mode="RGB"), (x0, header))
        draw.text((x0 + 4, 6), label, fill=(240, 244, 250))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite-ckpt", default="checkpoints/sr_lite/best.pt")
    ap.add_argument("--std-ckpt", default="checkpoints/sr_std/best.pt")
    ap.add_argument("--val-root", default="data/sr_data/val")
    ap.add_argument("--captures", nargs="*", default=[])
    ap.add_argument("--out", default="artifacts/sr_eval")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lite = load_sr_ckpt(Path(args.lite_ckpt), "sr_lite")
    std = load_sr_ckpt(Path(args.std_ckpt), "sr_std")

    val_root = Path(args.val_root)
    val_dirs = [val_root / s for s in ("open_media", "synla", "ui_synth") if (val_root / s).exists()] or [val_root]
    vds = ValidationSRDataset(val_dirs, hr_crop=256, scale=2, max_samples=96)

    scores: dict[str, list[float]] = {"nearest": [], "bilinear": [], "bicubic": [], "sharpen": [], "sr_lite": [], "sr_std": []}
    sample_strips = []
    for idx in range(len(vds)):
        lr_u8, hr_u8 = vds.items[idx]
        pil_lr = Image.fromarray(lr_u8, mode="RGB")
        h, w = hr_u8.shape[:2]
        nn = np.asarray(pil_lr.resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8)
        bil = np.asarray(pil_lr.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
        bic = np.asarray(pil_lr.resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)
        sharp = np.asarray(Image.fromarray(bic).filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2)), dtype=np.uint8)
        out_l = run_sr_rgb(lite, lr_u8)
        out_s = run_sr_rgb(std, lr_u8)

        for key, pred in [("nearest", nn), ("bilinear", bil), ("bicubic", bic), ("sharpen", sharp), ("sr_lite", out_l), ("sr_std", out_s)]:
            scores[key].append(psnr_u8(pred, hr_u8))

        if idx in (0, 25, len(vds) - 1):
            sample_strips.append((idx, make_strip([
                ("Nearest 2x", nn),
                ("Bicubic 2x", bic),
                ("Unsharp 2x", sharp),
                ("GameSR-Lite", out_l),
                ("GameSR-Std", out_s),
                ("GT HR", hr_u8),
            ])))

    for idx, strip in sample_strips:
        strip.save(out_dir / f"val_compare_{idx:02d}.png")

    summary = {k: round(sum(v) / max(1, len(v)), 3) for k, v in scores.items()}
    print("=== Quantitative Validation 2x Super-Resolution (PSNR dB, 96 samples) ===")
    print(json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for cidx, cap in enumerate(args.captures):
        cp = Path(cap)
        if not cp.exists():
            continue
        with Image.open(cp) as im:
            rgb = im.convert("RGB")
        # Extract crisp native crop (up to 480x270 -> upscales to 960x540, then zoom region for clear inspection)
        cw, ch = rgb.size
        crop_w, crop_h = min(240, cw // 2), min(160, ch // 2)
        x0, y0 = (cw - crop_w) // 2, (ch - crop_h) // 2
        patch = np.asarray(rgb.crop((x0, y0, x0 + crop_w, y0 + crop_h)), dtype=np.uint8)
        pil_p = Image.fromarray(patch)
        nn = np.asarray(pil_p.resize((crop_w * 2, crop_h * 2), Image.Resampling.NEAREST), dtype=np.uint8)
        bic = np.asarray(pil_p.resize((crop_w * 2, crop_h * 2), Image.Resampling.BICUBIC), dtype=np.uint8)
        out_l = run_sr_rgb(lite, patch)
        out_s = run_sr_rgb(std, patch)
        make_strip([
            ("Native 2x Nearest", nn),
            ("Bicubic 2x", bic),
            ("GameSR-Lite 2x", out_l),
            ("GameSR-Std 2x", out_s),
        ]).save(out_dir / f"capture_{cidx}_{cp.stem}.png")
    print(f"[eval] Saved comparison sheets -> {out_dir}")


if __name__ == "__main__":
    main()
