---
layout: home

hero:
  name: "FrameFlow"
  text: "Cross-Browser Neural Frame Interpolation & SR Suite"
  tagline: "Zero-dependency WebGPU Compute & WebGL2 RGBA16F runtimes paired with GameFI optical-flow interpolation and GameSR 2x super-resolution."
  actions:
    - theme: brand
      text: "Getting Started"
      link: /en/docs/
    - theme: alt
      text: "API & Components"
      link: /en/api-reference/
    - theme: alt
      text: "Model Contracts"
      link: /en/docs/contracts
    - theme: alt
      text: "Architecture & Diagram"
      link: /en/docs/architecture

features:
  - title: "WebGPU -> WebGL2 Fallback"
    details: "Zero npm/WASM runtime dependencies; enables shader-f16 compute when exposed and degrades cleanly to RGBA16F + MRT fragment shaders."
  - title: "GameFI & GameSR Families"
    details: "Ships GameFI-Std (775,811 params) / Lite (222,179) interpolation alongside GameSR-Std (~11.5K) / Lite (~6.9K) 2x scaling."
  - title: "Zero-Copy Sharing & Pacing"
    details: "Shares external GPUDevice (PVSSPW) or WebGL2 contexts (PPSSPW) with FrameFlowPacer and synthetic static-HUD protection."
  - title: "Open-Licensed Compliance Pack"
    details: "Trains solely on CC-BY 3.0, Xiph, SYNLA MIT, and synthetic UI cards, bundling 5 license texts, DATA_PROVENANCE.json, and SHA256SUMS."
---

## Capabilities & Explicit Boundaries

::: info What FrameFlow Delivers
- Runs pure-ES WebGPU Compute (`f16` / `f32`) and WebGL2 (`RGBA16F` + MRT) with **69–72 dB PSNR** (`GameFI`) and **62–63 dB PSNR** (`GameSR-2x`) parity against PyTorch in headless Edge 157 (Intel Xe-LPG).
- Supports shared external `GPUDevice` / `WebGL2RenderingContext`, `2×`/`3×` and arbitrary `t ∈ (0, 1)` phase synthesis, `FrameFlowPacer`, and chainable `GameFI + GameSR-2x` execution — see [API & Component Reference](/en/api-reference/).
:::

::: warning Explicit Non-Goals & Hard Limits
- **Zero binary weights or non-commercial data in Git**: `.bin` / `.pt` artifacts and training footage stay gitignored and publish via GitHub Releases / CNB OCI with provenance and checksums.
- **No CPU / WASM fallback**: Devices lacking WebGPU and float/half-float FBOs report unsupported cleanly instead of starving emulator CPUs.
- **Fixed 2× GameSR & zero hot-path readbacks**: `SrModelConfig` rejects non-2× scales and live loops never invoke blocking `mapAsync` / `readRgba()`.
:::

## Architecture Overview

<DiagramFrame
  src="/diagrams/frameflow-architecture.html"
  title="FrameFlow Export, Dual-Backend Runtime & Host Zero-Copy Architecture (click nodes / pan / zoom)"
  aspect="940 / 330"
  new-tab-label="Open Fullscreen ↗"
/>

## Minimal Usage

```js
import { probeCapabilities, createFrameFlow, loadModelFromUrl } from 'frameflow/runtime';

const caps = await probeCapabilities();
if (caps.supported) {
  const engine = await createFrameFlow({ backend: 'auto' });
  const fiModel = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
  const session = engine.createSession(fiModel, {
    width: 960, height: 544, internal: [480, 288], k: 1, stillGuard: true,
  });
  session.pushFrame(0, frameACanvas);
  session.pushFrame(1, frameBCanvas);
  session.pair(0, 1);
  session.mid(0.5, 0);
}
```

## Documentation Guide

::: tip Quick Navigation by Goal
- **Integrate into Browsers & Emulators**: Follow [Browser & Emulator Integration](/en/docs/integration) and [Integration & Runtime FAQ](/en/docs/troubleshooting) for PVSSPW & PPSSPW wiring;
- **Inspect Model Formats, Budgets & Training**: Read [GameFI / GameSR Contracts & Format](/en/docs/contracts), [Dual-Backend & Pacer Tuning](/en/docs/performance), and [Training & Releases](/en/docs/training);
- **Trace Source Ownership & ADRs**: Explore [Architecture & Interactive Diagram](/en/docs/architecture), [End-to-End Walkthrough](/en/docs/walkthrough), [Code Map](/en/docs/code-map), and [Decisions & Compliance ADRs](/en/docs/decisions).
:::
