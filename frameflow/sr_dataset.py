# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Training datasets & game-authentic 2x degradation pipeline for GameSRNet.

100% open-licensed training domain support:
  - Open movie / test sequence crops (CC-BY 3.0 Blender + Xiph/SVT)
  - SYNLA-Dataset v1 (MIT License, pure procedural synthetic line art & cel shapes)
  - Programmatic Game UI / HUD / CJK+ASCII Text generator (SIL OFL 1.1 Noto / Bitstream Vera/DejaVu)

All tensors are BGR [3, H, W] float32 in [0, 1] matching `GameSRNet`.
"""
from __future__ import annotations

import io
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch.utils.data import Dataset


_DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

_UI_STRINGS = [
    "HP 2480 / 3200", "MP 410", "LV.42", "EXP 94%", "STAGE 01 - 序章",
    "ワルプルギスの夜", "ソウルジェム", "魔法少女", "セーブ完了", "ロード中...",
    "オプション", "開始ゲーム", "继续游玩", "画质设定 2×", "ATTACK +35%",
    "CRITICAL!", "GUARD", "SCORE 184,900", "TIME 02:48.50", "CHAIN x16",
    "SKILL: マギカ・アロー", "ITEM x09", "NEXT: 1250", "READY?", "CLEAR!",
]


def _pick_font(size: int, font_paths: list[str] | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in (font_paths or _DEFAULT_FONT_CANDIDATES):
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_synthetic_game_ui(hr_size: int, rng: random.Random, font_paths: list[str] | None = None) -> np.ndarray:
    """Synthesize a crisp 2D RPG/action game UI & cel-shaded scene patch [hr_size, hr_size, 3] uint8 RGB."""
    # Rich cel-shaded background: angled gradients + crisp geometric cel polygons
    yy, xx = np.mgrid[0:hr_size, 0:hr_size].astype(np.float32) / float(hr_size)
    c0 = np.array([rng.randint(25, 220), rng.randint(25, 220), rng.randint(35, 235)], dtype=np.float32)
    c1 = np.array([rng.randint(20, 210), rng.randint(20, 210), rng.randint(30, 225)], dtype=np.float32)
    t = np.clip(rng.uniform(0.2, 0.9) * xx + rng.uniform(0.2, 0.9) * yy, 0.0, 1.0)[..., None]
    bg = np.clip(c0 * (1.0 - t) + c1 * t, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Crisp diagonal cel-shaded ribbons / triangles / character-like outline arcs
    for _ in range(rng.randint(3, 7)):
        fill_c = (rng.randint(30, 255), rng.randint(30, 255), rng.randint(30, 255), rng.randint(150, 255))
        edge_c = (rng.randint(10, 50), rng.randint(10, 50), rng.randint(15, 60), 255)
        kind = rng.choice(["poly", "ellipse", "line"])
        if kind == "poly":
            pts = [(rng.randint(-20, hr_size + 20), rng.randint(-20, hr_size + 20)) for _ in range(rng.randint(3, 5))]
            draw.polygon(pts, fill=fill_c, outline=edge_c, width=rng.randint(2, 4))
        elif kind == "ellipse":
            x0 = rng.randint(0, hr_size - 40)
            y0 = rng.randint(0, hr_size - 40)
            r = rng.randint(24, max(32, hr_size // 2))
            draw.ellipse([x0, y0, x0 + r, y0 + int(r * rng.uniform(0.7, 1.3))], fill=fill_c, outline=edge_c, width=rng.randint(2, 4))
        else:
            x0, y0 = rng.randint(0, hr_size), rng.randint(0, hr_size)
            x1, y1 = rng.randint(0, hr_size), rng.randint(0, hr_size)
            draw.line([x0, y0, x1, y1], fill=edge_c, width=rng.randint(2, 5))

    # Game UI panel + HP/MP bars + crisp CJK/Japanese/ASCII text
    n_ui = rng.randint(1, 3)
    for _ in range(n_ui):
        bx = rng.randint(6, max(8, hr_size // 3))
        by = rng.randint(6, max(8, hr_size - 72))
        bw = rng.randint(max(96, hr_size // 2), max(100, hr_size - bx - 6))
        bh = rng.randint(36, min(96, hr_size - by - 6))
        panel_bg = (rng.randint(10, 35), rng.randint(16, 48), rng.randint(28, 72), rng.randint(205, 245))
        border_c = (rng.randint(210, 255), rng.randint(215, 255), rng.randint(225, 255), 255)
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=rng.randint(4, 10), fill=panel_bg, outline=border_c, width=2)

        # Optional status bar inside panel
        if rng.random() < 0.65:
            bar_w = max(24, bw - 20)
            fill_w = max(8, int(bar_w * rng.uniform(0.25, 1.0)))
            bar_c = rng.choice([(60, 225, 110, 255), (70, 170, 255, 255), (245, 195, 55, 255), (240, 75, 120, 255)])
            draw.rectangle([bx + 10, by + bh - 14, bx + 10 + bar_w, by + bh - 6], fill=(20, 20, 28, 230), outline=(180, 190, 210, 255), width=1)
            draw.rectangle([bx + 11, by + bh - 13, bx + 10 + fill_w, by + bh - 7], fill=bar_c)

        txt = rng.choice(_UI_STRINGS)
        fsize = rng.randint(16, 28)
        font = _pick_font(fsize, font_paths)
        # Drop-shadow + crisp text (classic PSP/Vita dialogue style)
        draw.text((bx + 11, by + 7), txt, fill=(10, 12, 18, 230), font=font)
        draw.text((bx + 10, by + 6), txt, fill=(250, 252, 255, 255), font=font)

    return np.asarray(img, dtype=np.uint8)


def degrade_game_lr(hr_rgb: np.ndarray, scale: int, rng: random.Random) -> np.ndarray:
    """Create game-authentic LR input from clean HR RGB patch, teaching de-mosaic + de-blur."""
    h, w, _ = hr_rgb.shape
    lw, lh = w // scale, h // scale
    pil_hr = Image.fromarray(hr_rgb, mode="RGB")

    p = rng.random()
    if p < 0.35:
        # Profile A: Pixel / Nearest-neighbor & jagged 3D aliasing simulation
        if rng.random() < 0.35 and lw >= 64:
            # Slight sub-native step before nearest -> creates 2x/3x pixel staircase blocks
            mw = max(32, int(lw * rng.uniform(0.65, 0.90)))
            mh = max(32, int(lh * rng.uniform(0.65, 0.90)))
            lr_im = pil_hr.resize((mw, mh), Image.Resampling.NEAREST).resize((lw, lh), Image.Resampling.NEAREST)
        else:
            lr_im = pil_hr.resize((lw, lh), Image.Resampling.NEAREST)
    elif p < 0.60:
        # Profile B: Soft Bilinear + optional double-resample blur (fuzzy Vita/PSP 2D sprites & FMVs)
        if rng.random() < 0.50:
            blurred = pil_hr.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.35, 0.95)))
            lr_im = blurred.resize((lw, lh), Image.Resampling.BILINEAR)
        else:
            mw = max(32, int(lw * rng.uniform(0.62, 0.88)))
            mh = max(32, int(lh * rng.uniform(0.62, 0.88)))
            lr_im = pil_hr.resize((mw, mh), Image.Resampling.BILINEAR).resize((lw, lh), Image.Resampling.BILINEAR)
    elif p < 0.85:
        # Profile C: Downscale + RGB565 quantization / JPEG block compression
        resamp = rng.choice([Image.Resampling.BOX, Image.Resampling.BILINEAR, Image.Resampling.NEAREST])
        lr_im = pil_hr.resize((lw, lh), resamp)
        arr = np.asarray(lr_im, dtype=np.uint8).copy()
        if rng.random() < 0.55:
            # Simulate 16-bit RGB565 quantization (5-6-5 bits per channel)
            arr[:, :, 0] = (arr[:, :, 0] >> 3) << 3
            arr[:, :, 1] = (arr[:, :, 1] >> 2) << 2
            arr[:, :, 2] = (arr[:, :, 2] >> 3) << 3
            lr_im = Image.fromarray(arr, mode="RGB")
        if rng.random() < 0.65:
            buf = io.BytesIO()
            lr_im.save(buf, format="JPEG", quality=rng.randint(52, 88))
            buf.seek(0)
            lr_im = Image.open(buf).convert("RGB")
    else:
        # Profile D: Clean high-quality Lanczos / Bicubic downscale (preserves crisp fidelity)
        resamp = rng.choice([Image.Resampling.LANCZOS, Image.Resampling.BICUBIC, Image.Resampling.BOX])
        lr_im = pil_hr.resize((lw, lh), resamp)

    return np.asarray(lr_im, dtype=np.uint8)


def _rgb_u8_to_bgr_t(rgb: np.ndarray) -> torch.Tensor:
    bgr = rgb[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
    return torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1)))


def _rand_crop_or_resize(im: Image.Image, crop: int, rng: random.Random) -> np.ndarray:
    w, h = im.size
    if w >= crop * 2 and h >= crop * 2 and rng.random() < 0.45:
        # Random 2x super-sampled region downscaled via Lanczos -> extra clean crisp HR target!
        sc_crop = crop * 2
        x0 = rng.randint(0, w - sc_crop)
        y0 = rng.randint(0, h - sc_crop)
        patch = im.crop((x0, y0, x0 + sc_crop, y0 + sc_crop)).resize((crop, crop), Image.Resampling.LANCZOS)
    elif w >= crop and h >= crop:
        x0 = rng.randint(0, w - crop)
        y0 = rng.randint(0, h - crop)
        patch = im.crop((x0, y0, x0 + crop, y0 + crop))
    else:
        s = max(crop / max(w, 1), crop / max(h, 1))
        nw, nh = max(crop, int(round(w * s))), max(crop, int(round(h * s)))
        resized = im.resize((nw, nh), Image.Resampling.BICUBIC)
        x0 = rng.randint(0, nw - crop) if nw > crop else 0
        y0 = rng.randint(0, nh - crop) if nh > crop else 0
        patch = resized.crop((x0, y0, x0 + crop, y0 + crop))
    return np.array(patch, dtype=np.uint8, copy=True)


def _augment_hr(hr: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        hr = np.fliplr(hr)
    if rng.random() < 0.35:
        hr = np.flipud(hr)
    k = rng.randint(0, 3)
    if k:
        hr = np.rot90(hr, k)
    return np.array(hr, dtype=np.uint8, copy=True, order="C")


class GameSRDataset(Dataset):
    """Multi-source open-licensed dataset for 2x GameSRNet training."""

    def __init__(
        self,
        source_dirs: list[tuple[str | Path, float]],
        hr_crop: int = 256,
        scale: int = 2,
        ui_prob: float = 0.22,
        samples_per_epoch: int = 8192,
        augment: bool = True,
        font_paths: list[str] | None = None,
    ) -> None:
        self.hr_crop = hr_crop
        self.scale = scale
        self.ui_prob = ui_prob
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.font_paths = font_paths

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        self.buckets: list[tuple[list[Path], float]] = []
        for root, weight in source_dirs:
            rp = Path(root)
            if not rp.exists() or weight <= 0:
                continue
            files = sorted(f for f in rp.rglob("*") if f.is_file() and f.suffix.lower() in exts)
            if files:
                self.buckets.append((files, float(weight)))

        total_w = sum(w for _, w in self.buckets)
        self.norm_weights = [w / total_w for _, w in self.buckets] if total_w > 0 else []

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = random.Random(idx ^ random.getrandbits(32)) if self.augment else random.Random(idx + 1337)
        if not self.buckets or rng.random() < self.ui_prob:
            hr = generate_synthetic_game_ui(self.hr_crop, rng, self.font_paths)
        else:
            b_idx = rng.choices(range(len(self.buckets)), weights=self.norm_weights, k=1)[0]
            files = self.buckets[b_idx][0]
            path = files[rng.randrange(len(files))]
            with Image.open(path) as im:
                rgb_im = im.convert("RGB")
                hr = _rand_crop_or_resize(rgb_im, self.hr_crop, rng)
            if self.augment:
                hr = _augment_hr(hr, rng)
            if rng.random() < 0.20:
                # Composite crisp HUD elements on top of real scenes as well
                ui = generate_synthetic_game_ui(self.hr_crop, rng, self.font_paths)
                mask = (np.abs(ui.astype(np.int16) - ui[0, 0].astype(np.int16)).sum(axis=-1) > 40)
                if rng.random() < 0.5 and mask.any():
                    h0 = rng.randint(0, self.hr_crop // 2)
                    h1 = min(self.hr_crop, h0 + rng.randint(36, 96))
                    hr[h0:h1] = ui[h0:h1]

        lr = degrade_game_lr(hr, self.scale, rng)
        return {"LR": _rgb_u8_to_bgr_t(lr), "HR": _rgb_u8_to_bgr_t(hr)}


class ValidationSRDataset(Dataset):
    """Deterministic paired LR->HR validation benchmark across scenes, line art, and UI text."""

    def __init__(
        self,
        source_dirs: list[str | Path],
        hr_crop: int = 256,
        scale: int = 2,
        max_samples: int = 96,
        font_paths: list[str] | None = None,
    ) -> None:
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        files: list[Path] = []
        for root in source_dirs:
            rp = Path(root)
            if rp.exists():
                files.extend(sorted(f for f in rp.rglob("*") if f.is_file() and f.suffix.lower() in exts))

        rng = random.Random(20261124)
        n_ui = max(12, max_samples // 5)
        n_real = max_samples - n_ui

        if files:
            step = max(1, len(files) // max(1, n_real))
            picked = files[::step][:n_real]
            for f in picked:
                with Image.open(f) as im:
                    rgb = im.convert("RGB")
                    w, h = rgb.size
                    if w >= hr_crop and h >= hr_crop:
                        x0 = (w - hr_crop) // 2
                        y0 = (h - hr_crop) // 2
                        hr = np.asarray(rgb.crop((x0, y0, x0 + hr_crop, y0 + hr_crop)), dtype=np.uint8)
                    else:
                        hr = np.asarray(rgb.resize((hr_crop, hr_crop), Image.Resampling.BICUBIC), dtype=np.uint8)
                lr = degrade_game_lr(hr, scale, rng)
                self.items.append((lr, hr))

        while len(self.items) < max_samples:
            hr = generate_synthetic_game_ui(hr_crop, rng, font_paths)
            lr = degrade_game_lr(hr, scale, rng)
            self.items.append((lr, hr))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        lr, hr = self.items[idx]
        return {"LR": _rgb_u8_to_bgr_t(lr), "HR": _rgb_u8_to_bgr_t(hr)}
