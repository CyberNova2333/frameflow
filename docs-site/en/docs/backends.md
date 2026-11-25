# WebGPU & WebGL2 Dual-Backend Runtime

All runtime logic lives in plain ES modules (`runtime/`) with zero third-party packages or WebAssembly binaries.

## Backend Selection & Parity Mechanics

1. **`probeCapabilities()` / `createFrameFlow()`** tries **WebGPU Compute** first (automatically requesting `shader-f16` when supported, otherwise aliasing `F = f32`), then falls back to **WebGL2 Fragment Shaders** requiring float/half-float renderable FBO support.
2. When neither is supported, session creation fails explicitly instead of falling back to CPU/WASM compute that would starve host emulators.
3. **`parseModel()` & `compilePlan()`** (`runtime/loader.js`) validate even binary byte length, `format_version === 1`, tensor offsets, graph references, decode little-endian IEEE-754 `f16` weights via `f16ToF32`, pad working sizes with `align32()`, and unroll `repeat` loops into static `pairSteps` + `midSteps`.
4. **`FrameFlowPacer`** (`runtime/pacer.js`) maintains an EMA frame-interval estimate (clamped over 4–200 ms source deltas), queues `2×`/`3×` intermediate slots without calling GPU readback in the hot loop, and drops stale items when queue depth exceeds `factor + 2`.

| Dimension | WebGPU Backend | WebGL2 Backend |
| --- | --- | --- |
| Tensor layout | 1D storage buffers (`array<f16>` / `array<f32>`) | `RGBA16F TEXTURE_2D_ARRAY` (4 channels/layer) |
| Conv dispatch | 3D compute workgroups (`@workgroup_size`) | 4-target MRT `drawBuffers` (16 output ch/draw) |
| External zero-copy | `{ device: GPUDevice }` + `setExternalInputTexture()` | `{ gl: WebGL2RenderingContext, flipSrcY }` |
