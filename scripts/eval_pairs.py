#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Evaluate trained FrameFlow checkpoints on real consecutive game frames and generate zoom montages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frameflow.config import get_preset
from frameflow.model import FrameFlowNet


def load_model(ckpt_path: Path, device: torch.device) -> tuple[FrameFlowNet, str]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    preset = ck.get("preset", "std")
    model = FrameFlowNet(get_preset(preset)).to(device).eval()
    model.load_state_dict(ck.get("ema", ck["model"]))
    label = f"{model.cfg.name}@s{ck.get('step', '?')}"
    return model, label


def load_bgr(path: Path) -> torch.Tensor:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    bgr = rgb[:, :, ::-1].copy()
    return torch.from_numpy(bgr.transpose(2, 0, 1)).unsqueeze(0)


def to_pil(x: torch.Tensor) -> Image.Image:
    bgr = (x[0].detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(bgr[:, :, ::-1].copy())


def find_motion_crop(img_a: Image.Image, img_b: Image.Image, cw: int = 220, ch: int = 180) -> tuple[int, int, int, int]:
    w, h = img_a.size
    a = np.asarray(img_a, dtype=np.float32)
    b = np.asarray(img_b, dtype=np.float32)
    diff = np.mean(np.abs(a - b), axis=2)
    best_s, best_xy = -1.0, ((w - cw) // 2, (h - ch) // 2)
    for y in range(max(0, int(h * 0.18)), max(1, int(h * 0.80) - ch), 16):
        for x in range(max(0, int(w * 0.15)), max(1, int(w * 0.85) - cw), 16):
            s = float(np.mean(diff[y : y + ch, x : x + cw]))
            if s > best_s:
                best_s = s
                best_xy = (x, y)
    x0, y0 = best_xy
    return (x0, y0, x0 + cw, y0 + ch)


def make_montage(
    title: str,
    img_a: Image.Image,
    img_b: Image.Image,
    preds: list[tuple[str, Image.Image]],
    out_path: Path,
) -> None:
    arr_a = np.asarray(img_a, dtype=np.float32)
    arr_b = np.asarray(img_b, dtype=np.float32)
    img_bl = Image.fromarray(((arr_a + arr_b) * 0.5 + 0.5).astype(np.uint8))

    cols = [("Frame N (A)", img_a), ("50/50 Blend", img_bl), *preds, ("Frame N+1 (B)", img_b)]
    w, h = img_a.size
    mbox = find_motion_crop(img_a, img_b)
    hud_box = (int(w * 0.63), int(h * 0.84), int(w * 0.96), h - 4)

    tw, th = 320, int(round(320 * h / w))
    mw, mh = 320, int(round(320 * (mbox[3] - mbox[1]) / max(1, mbox[2] - mbox[0])))
    hw, hh = 320, max(64, int(round(320 * (hud_box[3] - hud_box[1]) / max(1, hud_box[2] - hud_box[0]))))
    pad = 8
    y0, y1, y2 = 40, 40 + th + 26, 40 + th + 26 + mh + 26

    canvas = Image.new("RGB", (pad + len(cols) * (tw + pad), y2 + hh + pad + 8), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(235, 240, 250))

    for i, (label, im) in enumerate(cols):
        x = pad + i * (tw + pad)
        draw.text((x + 4, y0 - 18), label, fill=(220, 230, 245))
        canvas.paste(im.resize((tw, th), Image.Resampling.LANCZOS), (x, y0))
        if i in (0, len(cols) - 1):
            sx, sy = tw / w, th / h
            draw.rectangle([x + mbox[0] * sx, y0 + mbox[1] * sy, x + mbox[2] * sx, y0 + mbox[3] * sy], outline=(255, 220, 40), width=2)
        draw.text((x + 4, y1 - 18), "Motion Crop", fill=(255, 220, 100))
        canvas.paste(im.crop(mbox).resize((mw, mh), Image.Resampling.NEAREST), (x, y1))
        draw.text((x + 4, y2 - 18), "Static HUD Region", fill=(140, 235, 190))
        canvas.paste(im.crop(hud_box).resize((hw, hh), Image.Resampling.NEAREST), (x, y2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[eval] Saved montage -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained FrameFlow checkpoints on real game pairs")
    ap.add_argument("--ckpts", nargs="+", required=True, help="One or more checkpoint (.pt) paths")
    ap.add_argument("--pairs-dir", default="../pvsspw/artifacts/flyhigh-proto/intel-v2", help="Directory containing *_A.png / *_B.png pairs")
    ap.add_argument("--out", default="eval_outputs", help="Directory to write comparison montages")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--still-test", action="store_true", help="Enable browser still-HUD guard during eval")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(Path(p), device) for p in args.ckpts]
    pairs_dir = Path(args.pairs_dir)
    a_files = sorted(pairs_dir.glob("*_A.png"))
    if not a_files:
        raise FileNotFoundError(f"No *_A.png pairs found in {pairs_dir}")

    out_dir = Path(args.out)
    for fa in a_files:
        stem = fa.name[:-6]
        fb = pairs_dir / f"{stem}_B.png"
        if not fb.exists():
            continue
        ta, tb = load_bgr(fa).to(device), load_bgr(fb).to(device)
        img_a, img_b = Image.open(fa).convert("RGB"), Image.open(fb).convert("RGB")
        preds = []
        with torch.no_grad():
            for model, label in models:
                out = model(ta, tb, t=args.t, still_test=args.still_test)
                preds.append((label, to_pil(out["render"])))
        make_montage(f"{stem} | t={args.t:.2f} | still_test={args.still_test}", img_a, img_b, preds, out_dir / f"eval_{stem}.png")


if __name__ == "__main__":
    main()
