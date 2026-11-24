#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Package trained FrameFlow exports (GameFI & GameSR) into Release-ready assets, checksums, and release notes.

Produces under `--out` (default `dist/`):
  - gamefi-std-<tag>.tar.gz / gamefi-lite-<tag>.tar.gz  (or gamesr-sr_*-<tag>.tar.gz)
  - gamefi-std-{graph.json,model.json,model.bin}        (direct-fetch files for browser/CDN)
  - gamefi-lite-{graph.json,model.json,model.bin}
  - SHA256SUMS.txt
  - RELEASE_NOTES.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_export_dir(ckpt_root: Path) -> Path | None:
    for sub in ("export_best", "export"):
        cand = ckpt_root / sub
        if (cand / "model.bin").exists() and (cand / "model.json").exists() and (cand / "graph.json").exists():
            return cand
    return None


def read_best_val(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    best = None
    last_train = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "val":
            if best is None or rec.get("val_psnr", -1) >= best.get("val_psnr", -1):
                best = rec
        elif rec.get("kind") == "train":
            last_train = rec
    return best or last_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="weights-v0.1.0")
    ap.add_argument("--presets", nargs="+", default=["std", "lite"])
    ap.add_argument("--dirs", nargs="+", default=None)
    ap.add_argument("--std-dir", default="checkpoints/std")
    ap.add_argument("--lite-dir", default="checkpoints/lite")
    ap.add_argument("--data-manifest", default="data/manifest.json")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--runner-note", default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compliance_files = [
        "LICENSE",
        "LICENSE-ADDITIONAL_TERMS-zh-CN.md",
        "LICENSE-ADDITIONAL_TERMS-en.md",
        "IMPLEMENTATION-GUIDE.zh-en.md",
        "ASSETS_LICENSES.md",
    ]
    data_manifest = Path(args.data_manifest)

    default_dirs = {"std": args.std_dir, "lite": args.lite_dir}
    pair_dirs = args.dirs if args.dirs else [default_dirs.get(p, f"checkpoints/{p}") for p in args.presets]
    packaged = []
    for preset, ckpt_arg in zip(args.presets, pair_dirs):
        ckpt_root = Path(ckpt_arg)
        exp_dir = pick_export_dir(ckpt_root)
        if exp_dir is None:
            print(f"[package] Skipping {preset}: no export found in {ckpt_root}")
            continue

        manifest = json.loads((exp_dir / "model.json").read_text(encoding="utf-8"))
        metrics = read_best_val(ckpt_root / "metrics.jsonl")
        prefix = "gamesr" if manifest["meta"].get("task") == "sr" or preset.startswith("sr_") else "gamefi"

        # Copy standalone direct-fetch files (for GitHub Release / CDN direct loading)
        for fname in ("graph.json", "model.json", "model.bin"):
            shutil.copy2(exp_dir / fname, out_dir / f"{prefix}-{preset}-{fname}")

        # Build clean tar.gz archive with full AGPLv3 + Additional Terms compliance pack + provenance
        tar_name = f"{prefix}-{preset}-{args.tag}.tar.gz"
        tar_path = out_dir / tar_name
        bundle_prefix = f"{prefix}-{preset}"
        with tarfile.open(tar_path, "w:gz") as tf:
            for fname in ("graph.json", "model.json", "model.bin"):
                tf.add(exp_dir / fname, arcname=f"{bundle_prefix}/{fname}")
            for cname in compliance_files:
                cpath = repo_root / cname
                if cpath.exists():
                    tf.add(cpath, arcname=f"{bundle_prefix}/{cname}")
            if data_manifest.exists():
                tf.add(data_manifest, arcname=f"{bundle_prefix}/DATA_PROVENANCE.json")
            if (ckpt_root / "metrics.jsonl").exists():
                tf.add(ckpt_root / "metrics.jsonl", arcname=f"{bundle_prefix}/metrics.jsonl")

        packaged.append({
            "preset": preset,
            "model_name": manifest["meta"].get("model", f"GameFI-{preset}"),
            "train_step": manifest["meta"].get("train_step", 0),
            "n_params": manifest["meta"].get("n_params", 0),
            "bin_bytes": (exp_dir / "model.bin").stat().st_size,
            "tar_name": tar_name,
            "metrics": metrics,
        })

    if not packaged:
        raise RuntimeError("No trained model exports found to package!")

    # Compute SHA256SUMS.txt
    asset_files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name not in ("SHA256SUMS.txt", "RELEASE_NOTES.md"))
    checksum_lines = [f"{sha256_file(p)}  {p.name}" for p in asset_files]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # Detect GPU name if torch is installed
    gpu_desc = args.runner_note
    if not gpu_desc:
        try:
            import torch
            gpu_desc = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({platform.machine()})"
        except Exception:
            gpu_desc = platform.platform()

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# FrameFlow Weights `{args.tag}`",
        "",
        f"- **Built at**: {now_utc}",
        f"- **Compute**: {gpu_desc}",
        "- **License**: GNU AGPLv3 (`AGPL-3.0-only`) + Additional Terms (`v1.6.7-r4`, see `LICENSE-ADDITIONAL_TERMS-zh-CN.md`); 100% self-trained weights on open-licensed media",
        "",
        "## Included Models",
        "",
        "| Model | Preset | Params | `model.bin` (f16) | Step | Best Val PSNR (`t=0.5` / `arb-t`) | HUD MAD | Archive |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for item in packaged:
        m = item["metrics"] or {}
        if "val_gain_db" in m:
            psnr_str = f"{m['val_psnr']:.2f} dB (bilinear {m.get('val_bilinear_psnr', 0):.2f} dB, +{m['val_gain_db']:.2f} dB)"
            hud_str = "UI + SYNLA + OpenMedia"
        elif "val_psnr" in m:
            psnr_str = f"{m['val_psnr']:.2f} dB ({m.get('val_psnr_t05', 0):.2f} / {m.get('val_psnr_arb', 0):.2f})"
            hud_str = f"{m.get('val_hud_mad255', 0):.2f}/255"
        elif "psnr" in m:
            psnr_str = f"train {m['psnr']:.2f} dB"
            hud_str = "—"
        else:
            psnr_str = "—"
            hud_str = "—"
        lines.append(
            f"| `{item['model_name']}` | `{item['preset']}` | {item['n_params']:,} | "
            f"{item['bin_bytes']/1024:.1f} KB | {item['train_step']:,} | {psnr_str} | {hud_str} | `{item['tar_name']}` |"
        )

    if data_manifest.exists():
        dm = json.loads(data_manifest.read_text(encoding="utf-8"))
        entries = dm.get("entries", [])
        if entries:
            lines += ["", "## Data Provenance", ""]
            for e in entries:
                lines.append(
                    f"- **{e.get('id', e.get('source_id'))}** — License: `{e.get('license', 'unknown')}` "
                    f"({e.get('frames', 0)} frames)"
                )

    lines += [
        "",
        "## Checksums (`SHA256SUMS.txt`)",
        "",
        "```text",
        *checksum_lines,
        "```",
        "",
    ]
    (out_dir / "RELEASE_NOTES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[package] Wrote {len(asset_files)} assets + SHA256SUMS.txt + RELEASE_NOTES.md -> {out_dir}")


if __name__ == "__main__":
    main()
