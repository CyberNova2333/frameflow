# Training, Distillation & Release Pipelines

FrameFlow trains purely self-owned `AGPL-3.0-only` weights (`v1.6.7-r4` Additional Terms) strictly on verified open-licensed media — never DIV2K/non-commercial scrapes or copyrighted game captures.

## Open-Licensed Data Sources

- **Interpolation (`scripts/prepare_data.py`)**: Blender Open Movies (`Big Buck Bunny`, `Tears of Steel`, `Sintel`, CC-BY 3.0) + Xiph/SVT sequences, filtered for hard scene cuts and dark frames and deterministically split 95/5 by sequence key. Augmented with `overlay_synthetic_hud()` to teach zero-flow on static HUD widgets.
- **Super-Resolution (`scripts/prepare_sr_data.py`)**: Open movie/test frames paired with SYNLA-Dataset v1 (MIT) procedural line art and synthetic CJK/Latin RPG UI HUD cards rendered via system vector fonts, trained under `GameSRLoss` (Charbonnier + Laplacian + Census + Sobel gradient + optional teacher distillation).

## Release Packaging & Verification (`scripts/package_release.py`)

Every packaged tar archive (`gamefi-<preset>-<tag>.tar.gz` or `gamesr-<preset>-<tag>.tar.gz`) bundles:
1. `(graph.json, model.json, model.bin)` (`format_version = 1`)
2. All 5 root licensing & implementation guide documents (`LICENSE`, `LICENSE-ADDITIONAL_TERMS-zh-CN.md`, `LICENSE-ADDITIONAL_TERMS-en.md`, `IMPLEMENTATION-GUIDE.zh-en.md`, `ASSETS_LICENSES.md`)
3. `DATA_PROVENANCE.json` & `metrics.jsonl`
4. Accompanying `SHA256SUMS.txt` and generated `RELEASE_NOTES.md`

Pre-trained releases can be fetched directly from [GitHub Releases](https://github.com/CyberNova2333/frameflow/releases) (`weights-v0.1.0` & `sr-weights-v0.1.0`) or reproduced on **CNB 48GB GPU** (`.cnb.yml` / `.cnb/web_trigger.yml`), **GitHub Actions** (`.github/workflows/train-and-release.yml`), or local CUDA hardware via `scripts/run_cloud_train.sh`.
