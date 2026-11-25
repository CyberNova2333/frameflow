# Architecture & Source Guide

FrameFlow separates model authoring/training (`frameflow/`, PyTorch) from zero-dependency browser execution (`runtime/`, ES modules) using one shared `format_version = 1` contract (`graph.json` + `model.json` + `model.bin`).

## Repository Layout & Source Guides

- **End-to-End Flow**: [Walkthrough](/en/docs/walkthrough) traces export -> `probeCapabilities()` / `createFrameFlow()` -> `parseModel()` / `compilePlan()` -> `pair()` / `mid()` / `upscale()` -> `FrameFlowPacer` and capability rejection boundaries.
- **Source Map**: [Code Map](/en/docs/code-map) lists owning files and verification targets across `frameflow/`, `runtime/`, `scripts/`, and `tests/`.
- **Chinese Source-Truth Ledger**: Full reader-facing source guides live in [`repo-docs/`](https://github.com/CyberNova2333/frameflow/tree/main/repo-docs) (`references/source-evidence.md`, `references/quality-review.md`, `change-log.md`), anchored at verified Git revisions.

## Verification Commands

```sh
# Fast CPU unit tests (PyTorch + Node loader/pacer)
make test

# Headless Edge 157 GPU parity verification (requires Vulkan/WebGPU adapter)
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json .venv/bin/python scripts/verify_backends.py
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json .venv/bin/python scripts/verify_sr_backends.py
```
