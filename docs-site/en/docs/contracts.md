# GameFI / GameSR Contracts & `format_version = 1` Schema

All hidden layer counts across both model families are constrained (`__post_init__`) to positive multiples of 4 so WGSL vector ops and WebGL2 `RGBA16F` 4-channel texture slices align 1:1.

## Preset Specifications

| Family & Preset | Exact Params | Channel Config | Search / Sampling | Binary16 Size | Target Role |
| --- | ---: | --- | --- | ---: | --- |
| `GameFI-Std` (`std`) | 775,811 | `36 / 72 / 96 / 48 / 24` | Coarse `r=3`, fine `r=2`, `bicubic` warp | ~1.52 MB | Desktop iGPU/dGPU & flagship WebGPU |
| `GameFI-Lite` (`lite`) | 222,179 | `20 / 40 / 48 / 24 / 16` | Coarse `r=2`, fine `r=1`, `bilinear` warp | ~434 KB | Universal WebGL2 fallback & mobile targets |
| `GameSR-Std` (`sr_std`) | ~11,548 | `ch=16`, 2 `ResPair`s | Fixed `2× pixel_shuffle` residual | ~22.6 KB | High-detail `2×` edge sharpening |
| `GameSR-Lite` (`sr_lite`) | ~6,936 | `ch=12`, 2 `ResPair`s | Fixed `2× pixel_shuffle` residual | ~13.5 KB | Ultra-fast `2×` UI & cartoon de-aliasing |

## Invariants & Compatibility Aliases

- **Naming & Python Aliases**: `GameFINet = FrameFlowNet` and `GameFILoss = FrameFlowLoss` remain exported from `frameflow/__init__.py`; exported manifests identify `GameFI-Std`/`Lite` or `GameSR-Std`/`Lite`.
- **BGR Color & Luma Alignment**: All tensor inputs follow `[B, 3, H, W]` BGR in `[0, 1]`, with `bgr_to_luma()` coefficient parity (`0.114*B + 0.587*G + 0.299*R`) across PyTorch, WGSL, and GLSL ES 3.00.
- **Arbitrary Phase `t` Conditioning**: `FilmHead` (`film_hidden = 64`) modulates intermediate features continuously via `evalFilmMlp` in JS so a single export supports `2×`, `3×`, or fractional phases.
- **Fixed `2×` Super-Resolution**: `SrModelConfig` validates `scale == 2`; non-2× scales require explicit op/shader additions.
