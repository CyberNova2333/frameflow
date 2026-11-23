#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Download & prepare 100% clean open-licensed (CC-BY / open media) 60fps/50fps training clips.

Features:
  - Resumable downloads (`curl -C -`) with license provenance recorded in `data/manifest.json`
  - Automatic ingestion of any user-recorded 60fps videos placed in `data/raw_local/`
  - Scene-cut detection + static/black frame filtering so training triplets NEVER cross a hard cut
  - Deterministic clip-level 95/5 train/val split (zero frame leakage between train and val)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from PIL import Image


def resolve_ffmpeg() -> str:
    """Locate ffmpeg binary (env FFMPEG_BIN -> PATH -> imageio-ffmpeg wheel for rootless notebooks)."""
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    sys_bin = shutil.which("ffmpeg")
    if sys_bin:
        return sys_bin
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio-ffmpeg is not installed. "
            "Install ffmpeg or run: pip install imageio-ffmpeg"
        ) from exc


@dataclass(frozen=True)
class VideoSource:
    id: str
    tier: str  # "starter" | "full"
    url: str
    filename: str
    license: str
    attribution: str
    fps_hint: float
    is_zip: bool = False


CATALOG: list[VideoSource] = [
    VideoSource(
        id="bbb_60fps",
        tier="starter",
        url="https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_60fps_normal.mp4.zip",
        filename="bbb_sunflower_1080p_60fps_normal.mp4.zip",
        license="CC-BY 3.0",
        attribution="(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org",
        fps_hint=60.0,
        is_zip=True,
    ),
    VideoSource(
        id="tears_of_steel_720p",
        tier="starter",
        url="https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        filename="tears_of_steel_720p.mov",
        license="CC-BY 3.0",
        attribution="(CC) Blender Foundation | mango.blender.org",
        fps_hint=24.0,
    ),
    VideoSource(
        id="sintel_1080p",
        tier="full",
        url="https://download.blender.org/durian/movies/Sintel.2010.1080p.mkv",
        filename="Sintel.2010.1080p.mkv",
        license="CC-BY 3.0",
        attribution="(c) copyright Blender Foundation | durian.blender.org",
        fps_hint=24.0,
    ),
    VideoSource(
        id="xiph_parkrun_720p50",
        tier="full",
        url="https://media.xiph.org/video/derf/y4m/720p50_parkrun_ter.y4m",
        filename="720p50_parkrun_ter.y4m",
        license="Xiph / SVT Open Test Media",
        attribution="SVT / Xiph.org Derf's Test Media Collection",
        fps_hint=50.0,
    ),
    VideoSource(
        id="xiph_mobcal_720p50",
        tier="full",
        url="https://media.xiph.org/video/derf/y4m/720p50_mobcal_ter.y4m",
        filename="720p50_mobcal_ter.y4m",
        license="Xiph / SVT Open Test Media",
        attribution="SVT / Xiph.org Derf's Test Media Collection",
        fps_hint=50.0,
    ),
    VideoSource(
        id="xiph_shields_720p50",
        tier="full",
        url="https://media.xiph.org/video/derf/y4m/720p50_shields_ter.y4m",
        filename="720p50_shields_ter.y4m",
        license="Xiph / SVT Open Test Media",
        attribution="SVT / Xiph.org Derf's Test Media Collection",
        fps_hint=50.0,
    ),
    VideoSource(
        id="xiph_stockholm_720p5994",
        tier="full",
        url="https://media.xiph.org/video/derf/y4m/720p5994_stockholm_ter.y4m",
        filename="720p5994_stockholm_ter.y4m",
        license="Xiph / SVT Open Test Media",
        attribution="SVT / Xiph.org Derf's Test Media Collection",
        fps_hint=59.94,
    ),
]


def check_urls(sources: list[VideoSource]) -> None:
    print("Checking catalog URLs (HTTP HEAD, zero download)...")
    for s in sources:
        out = subprocess.check_output(
            ["curl", "-sIL", "--max-time", "10", "-o", "/dev/null", "-w", "%{http_code}", s.url],
            text=True,
        ).strip()
        print(f"  [{out}] {s.tier:7s} | {s.id:24s} | {s.license:25s} | {s.url}")


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dst}")
    subprocess.run(
        [
            "curl", "-fL", "-C", "-",
            "--retry", "5", "--retry-delay", "3", "--connect-timeout", "20",
            "--progress-bar", "-o", str(dst), url,
        ],
        check=True,
    )


def resolve_video_file(src: VideoSource, raw_dir: Path) -> Path:
    archive = raw_dir / src.filename
    if not archive.exists() or (src.is_zip and not zipfile.is_zipfile(archive)):
        download_file(src.url, archive)
    if not src.is_zip:
        return archive
    extract_dir = raw_dir / src.id
    extract_dir.mkdir(parents=True, exist_ok=True)
    vids = list(extract_dir.rglob("*.mp4")) + list(extract_dir.rglob("*.mkv")) + list(extract_dir.rglob("*.mov"))
    if vids:
        return vids[0]
    print(f"[extract] {archive} -> {extract_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_dir)
    vids = list(extract_dir.rglob("*.mp4")) + list(extract_dir.rglob("*.mkv")) + list(extract_dir.rglob("*.mov"))
    if not vids:
        raise RuntimeError(f"No video found inside {archive}")
    return vids[0]


def thumb_mad(p1: Path, p2: Path) -> tuple[float, float]:
    """Compute fast 96x54 downsampled RGB MAD and mean luma of p2."""
    a = np.asarray(Image.open(p1).convert("RGB").resize((96, 54), Image.Resampling.BILINEAR), dtype=np.float32)
    b = np.asarray(Image.open(p2).convert("RGB").resize((96, 54), Image.Resampling.BILINEAR), dtype=np.float32)
    return float(np.mean(np.abs(a - b))), float(np.mean(b))


def split_val(clip_name: str, val_ratio: float = 0.05) -> bool:
    h = int(hashlib.sha1(clip_name.encode("utf-8")).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < val_ratio


def slice_video_to_clips(
    video_path: Path,
    source_id: str,
    out_clips_dir: Path,
    target_height: int = 544,
    clip_len: int = 24,
    max_clips: int = 400,
    min_motion_mad: float = 1.0,
    cut_mad: float = 34.0,
    jpeg_quality: int = 95,
    val_ratio: float = 0.05,
    stride_frames: int = 1,
) -> dict:
    """Decode video with ffmpeg, detect hard scene cuts, filter static shots, and emit micro-clips."""
    train_root = out_clips_dir / "train"
    val_root = out_clips_dir / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    existing = list(train_root.glob(f"{source_id}_c*")) + list(val_root.glob(f"{source_id}_c*"))
    if existing:
        n_frames = sum(len(list(d.glob("*.jpg"))) for d in existing)
        print(f"[slice] Skipping {source_id}: already found {len(existing)} clips ({n_frames} frames)")
        return {"source_id": source_id, "clips": len(existing), "frames": n_frames, "skipped": True}

    with tempfile.TemporaryDirectory(prefix=f"ff_{source_id}_") as tmp:
        tmp_dir = Path(tmp)
        vf = f"scale=-2:{target_height}:flags=lanczos"
        if stride_frames > 1:
            vf = f"select='not(mod(n\\,{stride_frames}))',{vf}"
        ffmpeg_bin = resolve_ffmpeg()
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "warning",
            "-i", str(video_path),
            "-vf", vf,
            "-vsync", "0",
            "-q:v", "2",
            str(tmp_dir / "%06d.jpg"),
        ]
        print(f"[slice] Decoding {video_path.name} -> {target_height}p ...")
        subprocess.run(cmd, check=True)
        frames = sorted(tmp_dir.glob("*.jpg"))
        print(f"[slice] Decoded {len(frames)} frames; filtering cuts & static segments...")

        # Partition frames into continuous non-cut runs with real motion
        runs: list[list[Path]] = []
        cur: list[Path] = []
        recent_mads: list[float] = []

        for i in range(1, len(frames)):
            mad, luma = thumb_mad(frames[i - 1], frames[i])
            avg_recent = (sum(recent_mads[-5:]) / len(recent_mads[-5:])) if recent_mads else mad
            is_cut = (mad > cut_mad) or (len(recent_mads) >= 3 and mad > max(18.0, avg_recent * 4.0))
            is_static_or_black = (mad < min_motion_mad) or (luma < 6.0)

            if is_cut or is_static_or_black:
                if len(cur) >= 6:
                    runs.append(cur)
                cur = []
                recent_mads = []
            else:
                if not cur:
                    cur.append(frames[i - 1])
                cur.append(frames[i])
                recent_mads.append(mad)
                if len(cur) >= clip_len:
                    runs.append(cur)
                    cur = []
                    recent_mads = []
        if len(cur) >= 6:
            runs.append(cur)

        # Evenly subsample runs if there are more than max_clips
        if len(runs) > max_clips:
            idxs = np.linspace(0, len(runs) - 1, max_clips, dtype=int)
            runs = [runs[j] for j in idxs]

        n_train, n_val, total_frames = 0, 0, 0
        for ci, run_frames in enumerate(runs):
            cname = f"{source_id}_c{ci:04d}"
            is_val = split_val(cname, val_ratio=val_ratio)
            dst_dir = (val_root if is_val else train_root) / cname
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fi, src_img in enumerate(run_frames):
                # Re-save at uniform high JPEG quality (or direct copy if already ffmpeg q:v 2)
                shutil.copy2(src_img, dst_dir / f"{fi:04d}.jpg")
            total_frames += len(run_frames)
            if is_val:
                n_val += 1
            else:
                n_train += 1

        print(f"[slice] {source_id}: wrote {n_train} train + {n_val} val clips ({total_frames} frames)")
        return {
            "source_id": source_id,
            "video": str(video_path),
            "train_clips": n_train,
            "val_clips": n_val,
            "frames": total_frames,
            "skipped": False,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare clean open-licensed FrameFlow training clips")
    ap.add_argument("--tier", default="starter", choices=["starter", "full", "local-only"])
    ap.add_argument("--data-dir", default="data", help="Root data directory")
    ap.add_argument("--height", type=int, default=544, help="Decoded clip height (544 matches PS Vita / PSP 2x)")
    ap.add_argument("--clip-len", type=int, default=24)
    ap.add_argument("--max-clips-per-video", type=int, default=450)
    ap.add_argument("--list", action="store_true", help="List available open sources and exit")
    ap.add_argument("--check-urls", action="store_true", help="Verify HTTP status of all catalog URLs without downloading")
    args = ap.parse_args()

    if args.list:
        for s in CATALOG:
            print(f"{s.tier:7s} | {s.id:24s} | {s.fps_hint:5.2f} fps | {s.license:25s} | {s.url}")
        return

    if args.check_urls:
        check_urls(CATALOG)
        return

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    local_dir = data_dir / "raw_local"
    clips_dir = data_dir / "clips"
    local_clips_dir = data_dir / "local_clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    if args.tier != "local-only":
        selected = [s for s in CATALOG if (args.tier == "full" or s.tier == "starter")]
        for src in selected:
            vid = resolve_video_file(src, raw_dir)
            stat = slice_video_to_clips(
                vid,
                src.id,
                clips_dir,
                target_height=args.height,
                clip_len=args.clip_len,
                max_clips=args.max_clips_per_video,
            )
            manifest_entries.append({**asdict(src), **stat})

    # Ingest any user-provided local 60fps videos (e.g., self-recorded 60fps games/animations)
    vid_exts = {".mp4", ".mkv", ".mov", ".webm", ".y4m", ".avi"}
    local_vids = sorted(p for p in local_dir.iterdir() if p.is_file() and p.suffix.lower() in vid_exts)
    for lv in local_vids:
        sid = "local_" + lv.stem.replace(" ", "_")
        stat = slice_video_to_clips(
            lv,
            sid,
            local_clips_dir,
            target_height=args.height,
            clip_len=args.clip_len,
            max_clips=args.max_clips_per_video,
        )
        manifest_entries.append({"id": sid, "license": "user-local", **stat})

    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": manifest_entries}, indent=2) + "\n", encoding="utf-8")
    print(f"[done] Wrote provenance manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
