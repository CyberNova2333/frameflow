#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Prepare 100%-open-licensed high-resolution frames & patches for GameSR-2x training.

Sources (zero DIV2K / zero NC / zero copyrighted anime scrapes):
  1. Blender Open Movies (`Big Buck Bunny` 1080p, `Tears of Steel` 720p) — CC-BY 3.0
  2. Xiph.org / SVT Derf Test Media (720p50 / 720p59.94) — Open Test Media
  3. SYNLA-Dataset v1 (`Dataset_Color`, 256x256 procedural shapes & lines) — MIT (bloc97)
  4. Synthetic Game UI & CJK/Japanese/Latin typography cards (`256x256`) — rendered via SIL OFL 1.1 / Bitstream Vera fonts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameflow.sr_dataset import generate_synthetic_game_ui
from prepare_data import resolve_ffmpeg


def is_val_key(key: str, val_ratio: float = 0.05) -> bool:
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < val_ratio


def extract_video_frames(
    video_path: Path,
    source_id: str,
    out_root: Path,
    every_n_frames: int,
    max_frames: int,
    min_luma: float = 12.0,
) -> dict:
    train_dir = out_root / "train" / "open_media"
    val_dir = out_root / "val" / "open_media"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    existing = list(train_dir.glob(f"{source_id}_*.jpg")) + list(val_dir.glob(f"{source_id}_*.jpg"))
    if existing:
        return {"source_id": source_id, "frames": len(existing), "skipped": True}

    ffmpeg_bin = resolve_ffmpeg()
    with tempfile.TemporaryDirectory(prefix=f"sr_{source_id}_") as tmp:
        tmp_dir = Path(tmp)
        vf = f"select='not(mod(n\\,{every_n_frames}))'"
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "warning",
            "-i", str(video_path),
            "-vf", vf,
            "-vsync", "0",
            "-q:v", "1",
            str(tmp_dir / "%05d.jpg"),
        ]
        print(f"[sr-data] Extracting full-res frames from {video_path.name} (every {every_n_frames} frames)...", flush=True)
        subprocess.run(cmd, check=True)
        frames = sorted(tmp_dir.glob("*.jpg"))

        kept: list[Path] = []
        for p in frames:
            with Image.open(p) as im:
                thumb = np.asarray(im.convert("RGB").resize((64, 36), Image.Resampling.BILINEAR), dtype=np.float32)
            if float(np.mean(thumb)) >= min_luma and float(np.std(thumb)) >= 10.0:
                kept.append(p)

        if len(kept) > max_frames:
            idxs = np.linspace(0, len(kept) - 1, max_frames, dtype=int)
            kept = [kept[i] for i in idxs]

        n_tr, n_va = 0, 0
        for idx, src in enumerate(kept):
            # Split by 16-frame block hash so adjacent shots don't leak between train and val
            block_key = f"{source_id}_b{idx // 16}"
            in_val = is_val_key(block_key, 0.06)
            dst = (val_dir if in_val else train_dir) / f"{source_id}_{idx:04d}.jpg"
            shutil.copy2(src, dst)
            if in_val:
                n_va += 1
            else:
                n_tr += 1

    print(f"[sr-data] {source_id}: kept {n_tr} train + {n_va} val full-res frames", flush=True)
    return {"source_id": source_id, "train": n_tr, "val": n_va, "frames": n_tr + n_va, "skipped": False}


def ingest_synla(synla_dir: Path, out_root: Path) -> dict:
    src_color = synla_dir / "Dataset_Color"
    if not src_color.exists():
        raise FileNotFoundError(f"SYNLA v1 Dataset_Color not found at {src_color}")
    train_dir = out_root / "train" / "synla"
    val_dir = out_root / "val" / "synla"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    existing = list(train_dir.glob("*.png")) + list(val_dir.glob("*.png"))
    if existing:
        return {"source_id": "synla_v1_color", "frames": len(existing), "skipped": True}

    imgs = sorted(src_color.glob("*.png"))
    n_tr, n_va = 0, 0
    for p in imgs:
        in_val = is_val_key(p.stem, 0.05)
        dst = (val_dir if in_val else train_dir) / p.name
        shutil.copy2(p, dst)
        if in_val:
            n_va += 1
        else:
            n_tr += 1
    print(f"[sr-data] synla_v1_color (MIT): copied {n_tr} train + {n_va} val PNGs", flush=True)
    return {"source_id": "synla_v1_color", "license": "MIT (bloc97/SYNLA-Dataset)", "train": n_tr, "val": n_va, "frames": n_tr + n_va}


def synthesize_ui_cards(out_root: Path, count: int = 1200, hr_size: int = 256) -> dict:
    train_dir = out_root / "train" / "ui_synth"
    val_dir = out_root / "val" / "ui_synth"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    existing = list(train_dir.glob("*.png")) + list(val_dir.glob("*.png"))
    if existing:
        return {"source_id": "synthetic_game_ui", "frames": len(existing), "skipped": True}

    rng = random.Random(20261124)
    n_tr, n_va = 0, 0
    for idx in range(count):
        arr = generate_synthetic_game_ui(hr_size, rng)
        in_val = idx >= int(count * 0.94)
        dst = (val_dir if in_val else train_dir) / f"ui_{idx:04d}.png"
        Image.fromarray(arr, mode="RGB").save(dst)
        if in_val:
            n_va += 1
        else:
            n_tr += 1
    print(f"[sr-data] synthetic_game_ui (SIL OFL 1.1 NotoCJK + Bitstream Vera DejaVu): generated {n_tr} train + {n_va} val PNGs", flush=True)
    return {"source_id": "synthetic_game_ui", "license": "AGPL-3.0-only (rendered via SIL OFL 1.1 NotoSansCJK & Bitstream Vera DejaVu)", "train": n_tr, "val": n_va, "frames": n_tr + n_va}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--synla-dir", default="data/synla_dl/SYNLA-Dataset-master")
    ap.add_argument("--out-dir", default="data/sr_data")
    ap.add_argument("--manifest", default="data/sr_manifest.json")
    ap.add_argument("--ui-count", type=int, default=1200)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_dir)
    entries = []

    video_specs = [
        ("bbb_1080p", raw_dir / "bbb_60fps/bbb_sunflower_1080p_60fps_normal.mp4", 60, 500, "CC-BY 3.0 (Blender Foundation Big Buck Bunny)"),
        ("tos_720p", raw_dir / "tears_of_steel_720p.mov", 40, 350, "CC-BY 3.0 (Blender Foundation Tears of Steel)"),
        ("xiph_parkrun_720p", raw_dir / "720p50_parkrun_ter.y4m", 4, 120, "Xiph / SVT Open Test Media"),
        ("xiph_mobcal_720p", raw_dir / "720p50_mobcal_ter.y4m", 4, 120, "Xiph / SVT Open Test Media"),
        ("xiph_shields_720p", raw_dir / "720p50_shields_ter.y4m", 4, 120, "Xiph / SVT Open Test Media"),
        ("xiph_stockholm_720p", raw_dir / "720p5994_stockholm_ter.y4m", 5, 120, "Xiph / SVT Open Test Media"),
    ]
    for sid, vpath, step_n, max_f, lic in video_specs:
        if vpath.exists():
            stat = extract_video_frames(vpath, sid, out_root, step_n, max_f)
            stat["license"] = lic
            entries.append(stat)
        else:
            print(f"[warn] Missing {vpath}, skipping")

    synla_path = Path(args.synla_dir)
    if synla_path.exists():
        entries.append(ingest_synla(synla_path, out_root))

    entries.append(synthesize_ui_cards(out_root, count=args.ui_count))

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"[done] SR data ready -> {out_root} | manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
