# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Training datasets & Game-Specific Synthetic Static HUD Overlay Augmentation.

All tensors returned in BGR [3, H, W] float32 in [0, 1] matching the FrameFlow model spec.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


def overlay_synthetic_hud(
    a: np.ndarray,
    gt: np.ndarray,
    b: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Overlay crisp 100%-static game-like HUD elements onto (A, GT, B) identically.

    Args:
        a, gt, b: uint8 HWC BGR arrays of identical shape [H, W, 3]
    Returns:
        (a_out, gt_out, b_out, hud_mask [H, W] float32 in {0, 1})
    """
    h, w, _ = a.shape
    hud_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hud_rgba)

    n_widgets = rng.randint(1, 3)
    for _ in range(n_widgets):
        kind = rng.choice(["bar", "box_text", "minimap", "badge"])
        if kind == "bar":
            bw = rng.randint(max(16, w // 4), max(20, int(w * 0.55)))
            bh = rng.randint(max(6, min(10, h // 8)), max(8, min(22, h // 4)))
            x0 = rng.randint(2, max(2, w - bw - 2))
            top_hi = max(2, min(h - bh - 2, h // 5))
            bot_lo = max(2, min(h - bh - 2, h - bh - h // 5))
            y0 = rng.choice([rng.randint(2, top_hi), rng.randint(bot_lo, max(bot_lo, h - bh - 2))])
            fill_pct = rng.uniform(0.3, 1.0)
            border_c = (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255), 255)
            bg_c = (20, 20, 28, rng.randint(180, 240))
            bar_c = (rng.randint(40, 255), rng.randint(60, 255), rng.randint(40, 255), 255)
            draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=bg_c, outline=border_c, width=2)
            fw = max(2, int((bw - 4) * fill_pct))
            draw.rectangle([x0 + 2, y0 + 2, x0 + 2 + fw, y0 + bh - 2], fill=bar_c)
        elif kind == "box_text":
            bw = rng.randint(min(32, w - 8), max(min(32, w - 8), min(180, w - 8)))
            bh = rng.randint(min(14, h - 8), max(min(14, h - 8), min(36, h - 8)))
            x0 = rng.randint(2, max(2, w - bw - 2))
            y0 = rng.randint(2, max(2, h - bh - 2))
            alpha = rng.randint(170, 255)
            bg_c = (rng.randint(15, 60), rng.randint(25, 110), rng.randint(30, 140), alpha)
            border_c = (230, 235, 245, 255)
            draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=bg_c, outline=border_c, width=2)
            label = rng.choice(["HP 999", "LV.10", "00:42", "MP 120", "SPECIAL", "STAGE 1", "SCORE 8400"])
            draw.text((x0 + 6, y0 + max(2, (bh - 10) // 2)), label, fill=(255, 255, 255, 255))
        elif kind == "minimap":
            rmax = max(10, min(48, min(h, w) // 4))
            rmin = min(22, rmax)
            rad = rng.randint(rmin, rmax)
            cx = rng.choice([rad + 4, max(rad + 4, w - rad - 4)])
            cy = rng.choice([rad + 4, max(rad + 4, h - rad - 4)])
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(18, 22, 26, 200), outline=(210, 215, 225, 255), width=2)
            draw.line([cx - rad + 4, cy, cx + rad - 4, cy], fill=(80, 90, 100, 200), width=1)
            draw.line([cx, cy - rad + 4, cx, cy + rad - 4], fill=(80, 90, 100, 200), width=1)
            draw.polygon([(cx, cy - 6), (cx - 5, cy + 5), (cx + 5, cy + 5)], fill=(255, 70, 200, 255))
        else:
            sz_max = max(14, min(40, min(h, w) // 3))
            sz = rng.randint(min(18, sz_max), sz_max)
            x0 = rng.randint(2, max(2, w - sz - 2))
            y0 = rng.randint(2, max(2, h - sz - 2))
            c = (rng.randint(80, 255), rng.randint(80, 255), rng.randint(80, 255), 255)
            draw.ellipse([x0, y0, x0 + sz, y0 + sz], fill=c, outline=(255, 255, 255, 255), width=2)
            draw.text((x0 + sz // 4, y0 + sz // 3), str(rng.randint(1, 99)), fill=(25, 25, 30, 255))

    hud_np = np.asarray(hud_rgba, dtype=np.float32)  # RGB + A
    alpha = hud_np[:, :, 3:4] / 255.0
    # Convert overlay RGB -> BGR
    hud_bgr = hud_np[:, :, [2, 1, 0]]
    mask = (alpha[:, :, 0] > 0.05).astype(np.float32)

    def blend_one(img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32) * (1.0 - alpha) + hud_bgr * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    return blend_one(a), blend_one(gt), blend_one(b), mask


class SequenceFrameDataset(Dataset):
    """Loads clips stored as directories of sequentially numbered images (.png / .jpg).

    Supports arbitrary-t sampling (`t in {0.25, 1/3, 0.5, 2/3, 0.75}`) so a single
    trained model handles both 2x (30->60) and 3x (20->60) interpolation cleanly.
    """

    def __init__(
        self,
        roots: list[str | Path],
        crop_size: int = 256,
        hud_prob: float = 0.35,
        max_span: int = 4,
        augment: bool = True,
        samples_per_epoch: int | None = None,
    ) -> None:
        self.crop_size = crop_size
        self.hud_prob = hud_prob
        self.max_span = max_span
        self.augment = augment
        self.clips: list[list[Path]] = []

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for root in roots:
            root_p = Path(root)
            if not root_p.exists():
                continue
            # Check direct image folder or nested clip subdirectories
            for d in [root_p, *sorted(p for p in root_p.rglob("*") if p.is_dir())]:
                imgs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)
                if len(imgs) >= 3:
                    self.clips.append(imgs)

        if not self.clips:
            raise RuntimeError(f"No valid frame sequence directories (>=3 images) found under {roots}")

        self.index: list[tuple[int, int]] = []
        for ci, frames in enumerate(self.clips):
            for fi in range(0, len(frames) - 2):
                self.index.append((ci, fi))
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.index)

    def _load_bgr(self, path: Path) -> np.ndarray:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = random.Random()
        if self.samples_per_epoch is not None:
            ci, fi = self.index[rng.randrange(len(self.index))]
        else:
            ci, fi = self.index[idx]

        frames = self.clips[ci]
        max_s = min(self.max_span, len(frames) - 1 - fi)
        span = rng.choice([s for s in (2, 3, 4) if s <= max_s]) if self.augment and max_s >= 2 else min(2, max_s)
        mid_off = rng.randint(1, span - 1) if self.augment else span // 2
        t_val = float(mid_off) / float(span)

        a = self._load_bgr(frames[fi])
        gt = self._load_bgr(frames[fi + mid_off])
        b = self._load_bgr(frames[fi + span])

        # Random crop (resize up first if source smaller than crop_size)
        h, w, _ = a.shape
        cs = self.crop_size
        if h < cs or w < cs:
            scale = max(cs / h, cs / w)
            nh, nw = int(math.ceil(h * scale)), int(math.ceil(w * scale))
            a = np.asarray(Image.fromarray(a).resize((nw, nh), Image.Resampling.BILINEAR))
            gt = np.asarray(Image.fromarray(gt).resize((nw, nh), Image.Resampling.BILINEAR))
            b = np.asarray(Image.fromarray(b).resize((nw, nh), Image.Resampling.BILINEAR))
            h, w, _ = a.shape

        if self.augment:
            y0 = rng.randint(0, h - cs)
            x0 = rng.randint(0, w - cs)
        else:
            y0 = (h - cs) // 2
            x0 = (w - cs) // 2

        a = a[y0 : y0 + cs, x0 : x0 + cs]
        gt = gt[y0 : y0 + cs, x0 : x0 + cs]
        b = b[y0 : y0 + cs, x0 : x0 + cs]

        if self.augment:
            if rng.random() < 0.5:
                a, gt, b = a[:, ::-1].copy(), gt[:, ::-1].copy(), b[:, ::-1].copy()
            if rng.random() < 0.25:
                a, gt, b = a[::-1, :].copy(), gt[::-1, :].copy(), b[::-1, :].copy()
            if rng.random() < 0.5:
                a, b = b, a
                t_val = 1.0 - t_val
            # Subtle channel permutation augmentation for color diversity
            if rng.random() < 0.2:
                perm = [0, 1, 2]
                rng.shuffle(perm)
                a, gt, b = a[:, :, perm].copy(), gt[:, :, perm].copy(), b[:, :, perm].copy()

        hud_mask = np.zeros((cs, cs), dtype=np.float32)
        if self.augment and self.hud_prob > 0 and rng.random() < self.hud_prob:
            a, gt, b, hud_mask = overlay_synthetic_hud(a, gt, b, rng)

        def to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float() / 255.0

        return {
            "A": to_t(a),
            "GT": to_t(gt),
            "B": to_t(b),
            "t": torch.tensor(t_val, dtype=torch.float32),
            "hud_mask": torch.from_numpy(hud_mask).unsqueeze(0),
        }


class ValidationTripletDataset(Dataset):
    """Deterministic validation dataset evaluating fixed (A, GT, B, t) triplets cleanly."""

    def __init__(
        self,
        roots: list[str | Path],
        crop_size: int = 256,
        max_samples: int = 128,
        hud_prob: float = 0.25,
        seed: int = 2026,
    ) -> None:
        self.crop_size = crop_size
        self.hud_prob = hud_prob
        self.seed = seed
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        triplets: list[tuple[Path, Path, Path, float]] = []

        for root in roots:
            root_p = Path(root)
            if not root_p.exists():
                continue
            for d in [root_p, *sorted(p for p in root_p.rglob("*") if p.is_dir())]:
                imgs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)
                if len(imgs) >= 4:
                    mid = len(imgs) // 2
                    # Include both t=0.5 (span 2) and t=1/3 (span 3) from each val clip
                    if mid >= 1 and mid + 1 < len(imgs):
                        triplets.append((imgs[mid - 1], imgs[mid], imgs[mid + 1], 0.5))
                    if len(imgs) >= 4:
                        triplets.append((imgs[0], imgs[1], imgs[3], 1.0 / 3.0))

        if not triplets:
            raise RuntimeError(f"No validation clips (>=4 frames) found under {roots}")

        if len(triplets) > max_samples:
            step = len(triplets) / max_samples
            triplets = [triplets[int(i * step)] for i in range(max_samples)]
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def _load_bgr(self, path: Path) -> np.ndarray:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pa, pg, pb, t_val = self.triplets[idx]
        a, gt, b = self._load_bgr(pa), self._load_bgr(pg), self._load_bgr(pb)
        h, w, _ = a.shape
        cs = self.crop_size
        if h < cs or w < cs:
            scale = max(cs / h, cs / w)
            nh, nw = int(math.ceil(h * scale)), int(math.ceil(w * scale))
            a = np.asarray(Image.fromarray(a).resize((nw, nh), Image.Resampling.BILINEAR))
            gt = np.asarray(Image.fromarray(gt).resize((nw, nh), Image.Resampling.BILINEAR))
            b = np.asarray(Image.fromarray(b).resize((nw, nh), Image.Resampling.BILINEAR))
            h, w, _ = a.shape
        y0, x0 = (h - cs) // 2, (w - cs) // 2
        a = a[y0 : y0 + cs, x0 : x0 + cs]
        gt = gt[y0 : y0 + cs, x0 : x0 + cs]
        b = b[y0 : y0 + cs, x0 : x0 + cs]

        rng = random.Random(self.seed + idx)
        hud_mask = np.zeros((cs, cs), dtype=np.float32)
        if self.hud_prob > 0 and rng.random() < self.hud_prob:
            a, gt, b, hud_mask = overlay_synthetic_hud(a, gt, b, rng)

        def to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float() / 255.0

        return {
            "A": to_t(a),
            "GT": to_t(gt),
            "B": to_t(b),
            "t": torch.tensor(t_val, dtype=torch.float32),
            "hud_mask": torch.from_numpy(hud_mask).unsqueeze(0),
        }


class WeightedMixDataset(Dataset):
    """Mixes multiple datasets according to explicit sampling probabilities (e.g. 75% open + 25% local game)."""

    def __init__(self, datasets: list[Dataset], weights: list[float], samples_per_epoch: int) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must be non-empty and of equal length")
        total = sum(weights)
        self.datasets = datasets
        self.cum_weights = []
        acc = 0.0
        for w in weights:
            acc += w / total
            self.cum_weights.append(acc)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = random.random()
        chosen = self.datasets[-1]
        for ds, cw in zip(self.datasets, self.cum_weights):
            if r <= cw:
                chosen = ds
                break
        return chosen[random.randrange(len(chosen))]


class SyntheticSmokeDataset(Dataset):
    """Zero-download procedural moving shapes + textured background + static HUD dataset.

    Used for fast smoke tests, parity verification, and unit testing of the training loop.
    """

    def __init__(self, length: int = 512, crop_size: int = 192, hud_prob: float = 0.5) -> None:
        self.length = length
        self.crop_size = crop_size
        self.hud_prob = hud_prob

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = random.Random(idx * 1337 + random.randint(0, 999999))
        cs = self.crop_size
        t_val = rng.choice([0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75])

        # Textured background with global camera pan
        yy, xx = np.mgrid[0:cs, 0:cs].astype(np.float32)
        bg_vx = rng.uniform(-14.0, 14.0)
        bg_vy = rng.uniform(-10.0, 10.0)
        freq1 = rng.uniform(0.04, 0.12)
        freq2 = rng.uniform(0.04, 0.12)
        c_base = np.array([rng.uniform(40, 160), rng.uniform(40, 160), rng.uniform(40, 160)], dtype=np.float32)

        # Moving foreground sprites (simulate cel-shaded character + limbs)
        sprites = []
        for _ in range(rng.randint(2, 5)):
            sprites.append({
                "x": rng.uniform(cs * 0.2, cs * 0.8),
                "y": rng.uniform(cs * 0.2, cs * 0.8),
                "vx": rng.uniform(-18.0, 18.0),
                "vy": rng.uniform(-18.0, 18.0),
                "rx": rng.uniform(10, 32),
                "ry": rng.uniform(10, 32),
                "col": (rng.randint(70, 255), rng.randint(70, 255), rng.randint(70, 255)),
                "rect": rng.random() < 0.5,
            })

        def render_at(tau: float) -> np.ndarray:
            sx = xx - bg_vx * tau
            sy = yy - bg_vy * tau
            pat = 0.5 + 0.35 * np.sin(sx * freq1) * np.cos(sy * freq2) + 0.15 * np.sin((sx + sy) * 0.08)
            img = np.clip(c_base[None, None, :] * pat[:, :, None], 0, 255).astype(np.uint8)
            pil = Image.fromarray(img)
            dr = ImageDraw.Draw(pil)
            for sp in sprites:
                cx = sp["x"] + sp["vx"] * tau
                cy = sp["y"] + sp["vy"] * tau
                box = [cx - sp["rx"], cy - sp["ry"], cx + sp["rx"], cy + sp["ry"]]
                if sp["rect"]:
                    dr.rectangle(box, fill=sp["col"], outline=(25, 20, 25), width=2)
                else:
                    dr.ellipse(box, fill=sp["col"], outline=(25, 20, 25), width=2)
            return np.asarray(pil, dtype=np.uint8)

        a = render_at(0.0)
        gt = render_at(t_val)
        b = render_at(1.0)
        hud_mask = np.zeros((cs, cs), dtype=np.float32)
        if rng.random() < self.hud_prob:
            a, gt, b, hud_mask = overlay_synthetic_hud(a, gt, b, rng)

        def to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float() / 255.0

        return {
            "A": to_t(a),
            "GT": to_t(gt),
            "B": to_t(b),
            "t": torch.tensor(t_val, dtype=torch.float32),
            "hud_mask": torch.from_numpy(hud_mask).unsqueeze(0),
        }
