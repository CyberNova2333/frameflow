PY := .venv/bin/python
DATA_DIR := data
STD_OUT := checkpoints/std
LITE_OUT := checkpoints/lite

RELEASE_TAG ?= weights-v0.1.0
GITHUB_REPO ?= CyberNova2333/frameflow

.PHONY: help env-check test test-runtime parity verify-runtime data-list data-check data-starter data-full data-local train-std train-lite eval-game package-release publish-github

help:
	@echo "FrameFlow commands:"
	@echo "  make env-check       - Verify Python, PyTorch CUDA/bf16, and ffmpeg"
	@echo "  make test            - Run fast CPU Python + Node runtime unit tests (~5s, battery-safe)"
	@echo "  make test-runtime    - Run zero-GPU Node.js syntax & loader/pacer tests"
	@echo "  make verify-runtime  - Verify self-authored WebGPU & WebGL2 backends vs PyTorch"
	@echo "  make parity          - Verify PyTorch <-> reference FlyHigh WebGPU parity"
	@echo "  make data-list       - List open-licensed video sources & licenses"
	@echo "  make data-check      - Verify catalog URLs via HTTP HEAD (zero download)"
	@echo "  make data-starter    - Download & slice starter CC-BY videos (~727 MB)"
	@echo "  make data-full       - Download & slice full CC-BY + Xiph 50/60fps catalog"
	@echo "  make data-local      - Slice user videos placed in data/raw_local/"
	@echo "  make train-std       - Train GameFI-Std (100K steps, auto-resume, ~1.5h on RTX 5070 Ti)"
	@echo "  make train-lite      - Distill GameFI-Lite from Std (50K steps, auto-resume, ~40m)"
	@echo "  make eval-game       - Render zoom comparison montages on PVSSPW captured battle frames"
	@echo "  make package-release - Package trained exports into dist/ with SHA256SUMS & RELEASE_NOTES"
	@echo "  make publish-github  - Publish dist/ weights directly to GitHub Release (RELEASE_TAG=...)"

env-check:
	$(PY) -c "import torch, torchvision, numpy, PIL, tqdm; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', '| bf16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
	@ffmpeg -version | head -n 1

test:
	$(PY) -m unittest discover -s tests -v
	@$(MAKE) test-runtime

test-runtime:
	@for f in runtime/*.js; do node --check "$$f"; done
	node --test tests/test_runtime_node.mjs

parity:
	$(PY) scripts/verify_parity.py

verify-runtime:
	$(PY) scripts/verify_backends.py

data-list:
	$(PY) scripts/prepare_data.py --list

data-check:
	$(PY) scripts/prepare_data.py --check-urls

data-starter:
	$(PY) scripts/prepare_data.py --tier starter --data-dir $(DATA_DIR) --height 544

data-full:
	$(PY) scripts/prepare_data.py --tier full --data-dir $(DATA_DIR) --height 544

data-local:
	$(PY) scripts/prepare_data.py --tier local-only --data-dir $(DATA_DIR) --height 544

train-std:
	$(PY) scripts/train.py \
		--preset std \
		--data $(DATA_DIR)/clips/train $(DATA_DIR)/local_clips/train \
		--data-weights 0.75 0.25 \
		--val-data $(DATA_DIR)/clips/val \
		--out $(STD_OUT) \
		--steps 100000 --batch 16 --crop 256 --lr 3e-4 --hud-prob 0.35 \
		--val-every 2500 --save-every 5000 --auto-resume

train-lite:
	$(PY) scripts/train.py \
		--preset lite \
		--teacher $$( [ -f $(STD_OUT)/best.pt ] && echo $(STD_OUT)/best.pt || echo $(STD_OUT)/latest.pt ) \
		--data $(DATA_DIR)/clips/train $(DATA_DIR)/local_clips/train \
		--data-weights 0.75 0.25 \
		--val-data $(DATA_DIR)/clips/val \
		--out $(LITE_OUT) \
		--steps 50000 --batch 16 --crop 256 --lr 4e-4 --hud-prob 0.35 \
		--val-every 2500 --save-every 5000 --auto-resume

eval-game:
	$(PY) scripts/eval_pairs.py \
		--ckpts $$( [ -f $(STD_OUT)/best.pt ] && echo $(STD_OUT)/best.pt || echo $(STD_OUT)/latest.pt ) \
		        $$( [ -f $(LITE_OUT)/best.pt ] && echo $(LITE_OUT)/best.pt || echo $(LITE_OUT)/latest.pt ) \
		--pairs-dir ../pvsspw/artifacts/flyhigh-proto/intel-v2 \
		--out eval_outputs --still-test

package-release:
	$(PY) scripts/package_release.py \
		--tag $(RELEASE_TAG) \
		--std-dir $(STD_OUT) \
		--lite-dir $(LITE_OUT) \
		--data-manifest $(DATA_DIR)/manifest.json \
		--out dist

publish-github:
	bash scripts/publish_github_release.sh $(RELEASE_TAG) dist $(GITHUB_REPO)
