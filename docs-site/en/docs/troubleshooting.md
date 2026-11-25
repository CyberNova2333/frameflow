# Integration & Runtime FAQ

## 1. Why does `probeCapabilities()` return `{ supported: false }`?

When a browser lacks both WebGPU and float/half-float WebGL2 FBO rendering (`RGBA16F`), FrameFlow refuses rather than falling back to slow CPU/WASM inference that would starve host emulators. Check `caps.supported` and hide AI toggles when `false`.

## 2. Colors look channel-swapped or upside down after sharing external textures

- **BGR ordering**: Internal tensors expect `[0, 1]` BGR matching `tex_in.wgsl` and `luma.wgsl`. Ensure uploads go through the built-in ingest shaders or match BGR ordering.
- **Y-flip in WebGL2**: WebGL FBOs often store origin at bottom-left; pass `flipSrcY: true` in `createSession()` so sampling matches top-down PyTorch coordinates.

## 3. Errors or 14–24 ms spikes when copying from an OffscreenCanvas

Chromium forbids main-thread `copyExternalImageToTexture` after `transferControlToOffscreen()`, and cross-device copies add significant overhead. Always pass the emulator's `{ device }` into `createFrameFlow()` and bind via `setExternalInputTexture()`.

## 4. `parseModel()` throws on `format_version` or missing tensors

Verify `(graph.json, model.json, model.bin)` come from the same release archive/stem and verify HTTP serving preserves exact binary length (`model.bin` length must be even for `binary16`).
