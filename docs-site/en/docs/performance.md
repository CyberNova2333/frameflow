# Dual-Backend Compute & Pacer Tuning

`runtime/` isolates browser graphics API variance behind pure ES modules with zero external runtime packages.

## Execution Mechanics

1. **WebGPU First (`runtime/webgpu-backend.js`)**: Requests `'shader-f16'` automatically when supported on the adapter (`alias F = f16`), cleanly aliasing `f32` when absent, and encodes `pair()` or `mid()` stages into compact queue submissions.
2. **WebGL2 Universal Fallback (`runtime/webgl2-backend.js`)**: Packs channels into `RGBA16F TEXTURE_2D_ARRAY` textures and dispatches 4-target `MRT drawBuffers` (up to 16 channels/draw) using top-down coordinates matching `F.grid_sample`.
3. **Explicit No-CPU Policy**: Missing both WebGPU and float/half-float FBO support fails fast rather than degrading to CPU/WASM compute that would stall the emulator main thread.
4. **Non-Blocking `FrameFlowPacer` (`runtime/pacer.js`)**: Estimates `emaIntervalMs` over `4–200 ms` deltas (`maxBufferMs = 120`) and drops queued entries once queue length exceeds `factor + 2`.

## Measured Parity & Timings (Headless Edge 157, Intel Xe-LPG)

| Backend | Model | PyTorch Parity PSNR | Max Pixel Diff | `960×544` (`int 480×288`) / `480->960` | PSP `480×272` / `960->1920` |
| --- | --- | ---: | ---: | ---: | ---: |
| **WebGPU** | `GameFI-Std` | 72.32 / 71.70 dB | 1 / 255 | 20.81 ms | — |
| **WebGPU** | `GameFI-Lite` | 72.12 / 72.56 dB | 1 / 255 | 10.13 ms | 7.02 ms |
| **WebGL2** | `GameFI-Std` | 69.55 / 69.38 dB | 1 / 255 | 40.21 ms | — |
| **WebGL2** | `GameFI-Lite` | 71.05 / 71.33 dB | 1 / 255 | 15.14 ms | 43.27 ms |
| **WebGPU / WebGL2** | `GameSR-Lite` | 63.48 / 63.48 dB | 1 / 255 | 25.6 / 32.4 ms | 45.5 / 54.4 ms |
| **WebGPU / WebGL2** | `GameSR-Std` | 62.52 / 62.52 dB | 1 / 255 | 16.1 / 33.1 ms | 53.0 / 79.8 ms |
