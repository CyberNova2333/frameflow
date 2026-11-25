---
layout: page
sidebar: false
title: API & Model Format Reference
description: Reference for FrameFlow probeCapabilities, createFrameFlow, FrameFlowSession, FrameFlowPacer, format_version=1 schema, and release packaging CLI.
---

<script setup lang="ts">
import { apiSpecEn } from '../../.vitepress/data/api-spec'
</script>

<ApiExplorer :spec="apiSpecEn">

### Capability Probing & Engine Creation (`createFrameFlow`) {#create-frameflow}

Selects WebGPU Compute (`shader-f16` with `f32` fallback) first and degrades automatically to WebGL2 `RGBA16F` + MRT; shares `{ device }` or `{ gl }` zero-copy handles without CPU/WASM fallback.

### Zero-Copy Session (`FrameFlowSession`) {#frameflow-session}

Compiles static `pairSteps`/`midSteps` ahead of inference, binds external GPU textures cleanly, locks `GameSR` to `2×`, and reserves `readRgba()` strictly for offline verification.

### Non-Blocking 2× / 3× Pacer (`FrameFlowPacer`) {#frameflow-pacer}

Tracks 4–200 ms EMA deltas, emits `t=0.5` or `t=1/3, 2/3` mid tasks, and drops stalled entries past `factor + 2`.

### Unified `format_version = 1` Schema {#model-format-v1}

Stores `graph.json`, `model.json`, and little-endian `binary16` `model.bin` with multiples-of-4 channels enforced across both families.

### Packaging CLI (`package_release.py`) {#package-release-cli}

Bundles model exports, 5 compliance texts, `DATA_PROVENANCE.json`, and `SHA256SUMS.txt` across CNB OCI and GitHub Releases.

</ApiExplorer>
