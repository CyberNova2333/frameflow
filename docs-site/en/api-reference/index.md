# API & Model Format Reference

Reference aligned with `runtime/index.js`, `runtime/loader.js`, `runtime/pacer.js`, and `runtime/types.d.ts`.

## Top-Level Runtime Entrypoints (`runtime/index.js`)

| Function | Signature / Options | Behavior |
| --- | --- | --- |
| `probeCapabilities(options?)` | `{ powerPreference?, gl?, canvas? }` | Returns `{ supported, recommended: 'webgpu' \| 'webgl2' \| null, webgpu, webgl2 }` without allocating session textures |
| `createFrameFlow(options?)` | `{ backend?: 'auto' \| 'webgpu' \| 'webgl2', device?, gl?, canvas?, precision?: 'auto' \| 'f16' \| 'f32', powerPreference? }` | Defaults to `webgpu` when `device` is passed, `webgl2` when `gl` is passed, or throws when neither float backend is supported |
| `loadModelFromUrl(engine, baseUrl, stem?)` | Directory URL + optional stem (`'model'`) | Fetches `graph.json`, `{stem}.json`, and `{stem}.bin` concurrently into `engine.loadModel()` |

## Session & Pacer API (`FrameFlowSession` / `FrameFlowPacer`)

| Method | Parameters | Notes & Constraints |
| --- | --- | --- |
| `engine.createSession(model, opts)` | `{ width, height, internal?: 'frame' \| 'auto' \| [w, h], k?, stillGuard?, sampling?, useFilm?, useRefine?, flipSrcY? }` | Pads interpolation dimensions with `align32()`; SR sessions output fixed `2×` size |
| `setExternalInputTexture(slot, tex)` | `slot`, same-device `GPUTexture` or `WebGLTexture` | Zero-copy binding without host RAM round-trips |
| `pushFrame(slot, source)` | Typed array or DOM canvas/bitmap/video source | Uploads source frame when external textures are not bound |
| `pair(slotA, slotB)` / `mid(t?, outSlot?)` | `t ∈ (0, 1)` | Runs feature & coarse stages once per pair, then arbitrary phase synthesis per `mid()` |
| `upscale(srcSlot?, outSlot?)` | `GameSR` 2× sessions only | Renders `(2*width, 2*height)` output texture in `outSlot` |
| `outputTexture(slot?)` | Output slot index | Returns GPU texture handle for downstream rendering or scaling |
| `readRgba(outSlot?)` | `Promise<Uint8Array>` | CPU readback intended strictly for tests/screenshots — never hot-loop pacing |
| `FrameFlowPacer` | `{ factor?: 2 \| 3, maxBufferMs?: 120 }` | Tracks `emaIntervalMs` over `4–200 ms` deltas; drops queued frames beyond `factor + 2` |

## Artifact Format Contract (`format_version = 1`)

- `graph.json`: Static topological `ops` list (`repeat`, `conv`, `sad_volume`, `film`, `render_sr`).
- `model.json`: `{ meta: { model, task, format_version: 1, n_params, cfg }, tensors: { [name]: { shape, offset } } }`.
- `model.bin`: Contiguous little-endian IEEE-754 `binary16` byte stream (even byte length enforced by `parseModel()`).
