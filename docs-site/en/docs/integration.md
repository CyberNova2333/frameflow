# Browser & Emulator Integration

[`runtime/`](https://github.com/CyberNova2333/frameflow/tree/main/runtime) ships zero-dependency, zero-WASM ES modules exposing unified **WebGPU Compute -> WebGL2 Fragment Shader** execution.

## Capability Probing & Backend Selection

```js
import { probeCapabilities, createFrameFlow, loadModelFromUrl } from 'frameflow/runtime';

const caps = await probeCapabilities();
if (!caps.supported) {
  console.info('FrameFlow unsupported on this GPU:', caps);
} else {
  const engine = await createFrameFlow({ backend: 'auto' });
  const fiModel = await loadModelFromUrl(engine, '/assets/models/gamefi-lite/');
  const session = engine.createSession(fiModel, {
    width: 960,
    height: 544,
    internal: [480, 288],
    k: 1,
    stillGuard: true,
  });
}
```

| Backend | Browser Targets | Recommended Model | Working Size |
| --- | --- | --- | --- |
| **WebGPU** (`f16` -> `f32`) | Chrome / Edge, Safari 26+, Win/Mac Firefox | `GameFI-Std` (dGPU) / `GameFI-Lite` (iGPU/mobile) | `[480, 288]` for `960×544`; `'frame'` on dGPUs |
| **WebGL2** (`RGBA16F` + MRT) | Linux/Android Firefox, iOS 15–18 Safari, WebViews, **PPSSPW shared context** | `GameFI-Lite` (~434 KB) | `'frame'` for PSP `480×272`; `[480, 288]` for `960×544` |

## Zero-Copy Emulator Integration

- **PVSSPW (Shared `GPUDevice`)**: Call `createFrameFlow({ backend: 'webgpu', device: vitaslopDevice })`, wrap the guest swapchain in a 3-slot same-device ring (`setExternalInputTexture(slot, texture)`), run `pair(0, 1)` + `mid(t, 0)` before any spatial scaler (FSR 1 or GameSR-2x), and read `session.outputTexture(0)` without CPU readback.
- **PPSSPW (Shared `WebGL2RenderingContext`)**: Pass `{ backend: 'webgl2', gl, flipSrcY: true }` or push the WebGL canvas without triggering `mapAsync` stalls.
- **Standalone GameSR-2x**: Load `gamesr-lite` or `gamesr-std` (`width: 480, height: 272`) and call `srSession.upscale(srcSlot, outSlot)` to obtain sharp `960×544` output.
