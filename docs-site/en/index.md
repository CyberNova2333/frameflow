---
layout: home

hero:
  name: "FrameFlow"
  text: "Cross-Browser Neural Frame Interpolation & SR Suite"
  tagline: "Zero-dependency WebGPU & WebGL2 runtimes paired with GameFI optical-flow interpolation and GameSR 2x super-resolution for web emulators."
  actions:
    - theme: brand
      text: "Getting Started"
      link: /en/docs/
    - theme: alt
      text: "API & Format Spec"
      link: /en/api-reference/
    - theme: alt
      text: "Architecture & Source"
      link: /en/docs/architecture

features:
  - title: "WebGPU -> WebGL2 Automatic Fallback"
    details: "Pure ES modules with zero WASM or npm runtime deps; enables shader-f16 when available and falls back to RGBA16F MRT fragment shaders."
  - title: "GameFI & GameSR Model Families"
    details: "Ships presets for GameFI-Std (775K) & GameFI-Lite (222K) interpolation alongside GameSR-Std (~11.5K) & GameSR-Lite (~6.9K) 2x scaling."
  - title: "Zero-Copy Sharing & Clean Releases"
    details: "Accepts external GPUDevice or WebGL2RenderingContext handles without readback stalls and ships SHA-256 checksummed open-media weights."
---

::: info Delivered Capabilities
- Matches PyTorch exports within **69–72 dB PSNR** (`GameFI`) and **62–63 dB PSNR** (`GameSR-2x`) across WebGPU and WebGL2 in headless Edge 157 (Intel Xe-LPG) verification.
- Supports `2x` (`t=0.5`), `3x` (`t=1/3, 2/3`), arbitrary phase `t ∈ (0, 1)`, synthetic static HUD zero-flow training, and chained `GameFI + GameSR` pipelines under one unified `format_version = 1` artifact contract.
:::

::: warning Explicit Non-Goals & Boundaries
- **Zero binary weights or training footage in Git**: `.bin` / `.pt` artifacts and training media stay gitignored and are distributed via GitHub Releases (`weights-v0.1.0`, `sr-weights-v0.1.0`) or CNB OCI.
- **No CPU / WASM inference fallback**: Devices lacking both WebGPU and float/half-float FBO support report unsupported rather than starving host emulator threads.
- **Fixed 2× scaling & non-blocking pacer**: `GameSR-2x` targets `scale = 2` only, and hot-path pacing never invokes synchronous `mapAsync` readbacks.
:::

## Quick Browser Usage

```js
import { createFrameFlow, loadModelFromUrl } from 'frameflow/runtime';

const engine = await createFrameFlow({ backend: 'auto' }); // Or pass { device } / { gl }
const fiModel = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
const session = engine.createSession(fiModel, {
  width: 960,
  height: 544,
  internal: [480, 288],
  k: 1,
  stillGuard: true,
});

session.pushFrame(0, frameACanvas);
session.pushFrame(1, frameBCanvas);
session.pair(0, 1);
session.mid(0.5, 0);
```
