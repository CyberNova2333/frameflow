# Overview & Scope Boundaries

**FrameFlow** unifies neural optical-flow frame interpolation (**GameFI**) and real-time 2× super-resolution (**GameSR**) into a single export specification and zero-dependency browser inference suite (`WebGPU Compute` -> `WebGL2 Fragment Shaders`).

## Component Layers

| Layer | Responsibility | Key Files |
| --- | --- | --- |
| Model & Export (`frameflow/`) | `GameFINet`, `GameSRNet`, shaders-matched operators, losses, and `format_version=1` exporters | `frameflow/model.py`, `frameflow/sr_model.py` |
| Browser Runtime (`runtime/`) | Capability probing, WebGPU/WebGL2 backends, binary16 loader, and zero-stall pacer | `runtime/index.js`, `runtime/loader.js` |
| Pipelines & Scripts (`scripts/`) | Open-media slicing, `bf16` training & KD, Edge parity verification, and release bundling | `scripts/train.py`, `scripts/package_release.py` |
| Fast CPU Tests (`tests/`) | Deterministic PyTorch checks + zero-GPU Node loader & pacer unit tests | `tests/test_frameflow.py`, `tests/test_runtime_node.mjs` |

## Model Presets

| Model | Preset | Task | Params | Binary16 Size | Recommended Target |
| --- | --- | --- | ---: | ---: | --- |
| `GameFI-Std` | `std` | `interp` | 775,811 | ~1.52 MB | Desktop iGPU/dGPU & flagship mobile WebGPU |
| `GameFI-Lite` | `lite` | `interp` | 222,179 | ~434 KB | Universal WebGL2 fallback & mobile targets |
| `GameSR-Std` | `sr_std` | `sr` (2×) | ~11,548 | ~22.6 KB | Higher-detail 2× edge and UI sharpening |
| `GameSR-Lite` | `sr_lite` | `sr` (2×) | ~6,936 | ~13.5 KB | Ultra-lightweight 2× scaling in WebGPU / WebGL2 |

Continue to [Browser & Emulator Integration](/en/docs/integration), [Training, Distillation & Releases](/en/docs/training), or [API & Format Reference](/en/api-reference/).
