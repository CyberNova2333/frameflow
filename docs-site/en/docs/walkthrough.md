# End-to-End Execution Walkthrough

Tracing a full lifecycle from PyTorch export to real-time browser presentation highlights five explicit handoffs and rejection gates:

1. **Export (`frameflow/export.py`, `frameflow/sr_export.py`)**: Produces `(graph.json, model.json, model.bin)` tagged `format_version = 1` with little-endian `f16` weights and `align32`-compatible BGR topology.
2. **Capability Probe & Engine Init (`runtime/index.js`)**: `probeCapabilities()` and `createFrameFlow()` prefer WebGPU (requesting `shader-f16` if present) and fall back to WebGL2 float/half-float FBOs; unsupported browsers fail explicitly.
3. **Model Parse & Plan (`runtime/loader.js`)**: `parseModel()` checks binary alignment, bounds, and tensor references before `f16ToF32()` decoding; `compilePlan()` expands `repeat` ops into `pairSteps` and `midSteps` for a session.
4. **Texture Binding & Encode (`runtime/webgpu-backend.js`, `runtime/webgl2-backend.js`)**: Shared handles (`{ device }` or `{ gl }`) accept external textures (`setExternalInputTexture`) or canvas uploads (`pushFrame`) and encode passes without hot-path `mapAsync` readbacks.
5. **Non-Blocking Pacing (`runtime/pacer.js`)**: `FrameFlowPacer.onSourceFrame()` schedules `t = 0.5` (`factor=2`) or `t = 1/3, 2/3` (`factor=3`) mid tasks, and `tick()` yields source/mid descriptors while shedding overflow (`queue.length > factor + 2`).

Inspect the owning modules in the [Code Map](/en/docs/code-map) or run `make test` to execute the offline CPU + Node loader checks.
