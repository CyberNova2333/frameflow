#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
#
# One-command cloud GPU training & packaging runner (works in rootless Notebooks / SCNet / AutoDL / Linux servers).
#
# Quick start:
#   bash scripts/run_cloud_train.sh
#
# Optional overrides:
#   DATA_TIER=full STD_STEPS=100000 LITE_STEPS=50000 RELEASE_TAG=weights-v0.1.0 bash scripts/run_cloud_train.sh
#   DATA_TIER=synthetic STD_STEPS=100 LITE_STEPS=100 bash scripts/run_cloud_train.sh   # 1-min smoke test
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-$(command -v python3 || command -v python)}"
DATA_TIER="${DATA_TIER:-starter}"           # starter | full | local-only | synthetic
TRAIN_PRESETS="${TRAIN_PRESETS:-both}"      # both | std | lite
STD_STEPS="${STD_STEPS:-100000}"
LITE_STEPS="${LITE_STEPS:-50000}"
RELEASE_TAG="${RELEASE_TAG:-weights-v0.1.0}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "=== [1/6] Inspecting GPU & Python environment ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
else
  VRAM_MB=0
  GPU_NAME="CPU"
fi

NCPU="$(nproc 2>/dev/null || echo 4)"
DEFAULT_WORKERS=$(( NCPU > 2 ? (NCPU - 1 < 6 ? NCPU - 1 : 6) : 2 ))
if [ "${VRAM_MB:-0}" -ge 20000 ]; then
  DEFAULT_BATCH=32
else
  DEFAULT_BATCH=16
fi
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH}"
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"

echo "[cfg] py=$PY | gpu=$GPU_NAME (${VRAM_MB} MiB) | cpus=$NCPU | batch=$BATCH_SIZE | workers=$WORKERS"
echo "[cfg] tier=$DATA_TIER | presets=$TRAIN_PRESETS | std_steps=$STD_STEPS | lite_steps=$LITE_STEPS | tag=$RELEASE_TAG"

echo "=== [2/6] Checking lightweight Python deps & ffmpeg (zero root required) ==="
PIP_ARGS=()
if [ -n "${PIP_MIRROR:-}" ]; then
  PIP_ARGS=(-i "$PIP_MIRROR" --trusted-host "$(echo "$PIP_MIRROR" | awk -F/ '{print $3}')")
fi

$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch not available!'" 2>/dev/null || {
  echo "[warn] CUDA PyTorch not detected in $PY; checking current torch..."
  $PY -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" || {
    echo "[install] Installing PyTorch CUDA 12.4 wheel..."
    $PY -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision
  }
}

MISSING=()
for mod in numpy PIL tqdm; do
  $PY -c "import $mod" 2>/dev/null || MISSING+=("$mod")
done
if ! command -v ffmpeg >/dev/null 2>&1; then
  $PY -c "import imageio_ffmpeg" 2>/dev/null || MISSING+=("imageio-ffmpeg")
fi
if [ "${#MISSING[@]}" -gt 0 ]; then
  PKGS=()
  for m in "${MISSING[@]}"; do
    [ "$m" = "PIL" ] && PKGS+=("pillow") || PKGS+=("$m")
  done
  echo "[install] Installing missing packages: ${PKGS[*]}"
  $PY -m pip install --no-cache-dir -q "${PIP_ARGS[@]}" "${PKGS[@]}" || $PY -m pip install --no-cache-dir -q "${PKGS[@]}"
fi

$PY -c "import torch; print('[ok] PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| bf16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
$PY -m unittest discover -s tests -v

echo "=== [3/6] Preparing clean open-licensed training dataset (tier=$DATA_TIER) ==="
if [ "$DATA_TIER" = "synthetic" ]; then
  DATA_ARGS=(--synthetic)
elif [ "$DATA_TIER" = "local-only" ]; then
  $PY scripts/prepare_data.py --tier local-only --data-dir data --height 544
  DATA_ARGS=(--data data/local_clips/train --val-data data/local_clips/val)
else
  $PY scripts/prepare_data.py --tier "$DATA_TIER" --data-dir data --height 544
  if [ -d data/local_clips/train ] && [ -n "$(ls -A data/local_clips/train 2>/dev/null)" ]; then
    DATA_ARGS=(--data data/clips/train data/local_clips/train --data-weights 0.75 0.25 --val-data data/clips/val)
  else
    DATA_ARGS=(--data data/clips/train --val-data data/clips/val)
  fi
fi

echo "=== [4/6] Training GameFI-Std ==="
if [ "$TRAIN_PRESETS" = "both" ] || [ "$TRAIN_PRESETS" = "std" ]; then
  $PY scripts/train.py \
    --preset std \
    "${DATA_ARGS[@]}" \
    --out checkpoints/std \
    --steps "$STD_STEPS" \
    --batch "$BATCH_SIZE" \
    --crop 256 \
    --lr 3e-4 \
    --hud-prob 0.35 \
    --workers "$WORKERS" \
    --val-every 2500 \
    --save-every 5000 \
    --auto-resume
fi

echo "=== [5/6] Distilling GameFI-Lite ==="
if [ "$TRAIN_PRESETS" = "both" ] || [ "$TRAIN_PRESETS" = "lite" ]; then
  TEACHER_ARGS=()
  if [ -f checkpoints/std/best.pt ]; then
    TEACHER_ARGS=(--teacher checkpoints/std/best.pt)
  elif [ -f checkpoints/std/latest.pt ]; then
    TEACHER_ARGS=(--teacher checkpoints/std/latest.pt)
  fi
  $PY scripts/train.py \
    --preset lite \
    "${TEACHER_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    --out checkpoints/lite \
    --steps "$LITE_STEPS" \
    --batch "$BATCH_SIZE" \
    --crop 256 \
    --lr 4e-4 \
    --hud-prob 0.35 \
    --workers "$WORKERS" \
    --val-every 2500 \
    --save-every 5000 \
    --auto-resume
fi

echo "=== [6/6] Packaging release weights & SHA-256 checksums -> dist/ ==="
$PY scripts/package_release.py \
  --tag "$RELEASE_TAG" \
  --std-dir checkpoints/std \
  --lite-dir checkpoints/lite \
  --data-manifest data/manifest.json \
  --out dist \
  --runner-note "Cloud GPU (${GPU_NAME})"

ls -lh dist/
echo ""
echo "=== DONE! Download dist/ or publish directly with: ==="
echo "  bash scripts/publish_github_release.sh ${RELEASE_TAG} dist CyberNova2333/frameflow"
