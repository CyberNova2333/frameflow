# GameFI & GameSR Model Specifications

Both families share channel counts constrained to positive multiples of 4 (`__post_init__` enforced) so WebGL2 packs every intermediate layer cleanly into `RGBA16F` `TEXTURE_2D_ARRAY` slices and 4-target MRT draws.

## Presets & Parameter Counts

| Model | Preset | Hidden Config | Flow Radii & Sampling | Exact Params | Binary16 Size |
| --- | --- | --- | --- | ---: | ---: |
| `GameFI-Std` | `std` | `c8=36, c16=72, cc=96, cr=48, cq=24` | Coarse `r=3`, fine `r=2`, `bicubic` warp | 775,811 | ~1.52 MB |
| `GameFI-Lite` | `lite` | `c8=20, c16=40, cc=48, cr=24, cq=16` | Coarse `r=2`, fine `r=1`, `bilinear` warp | 222,179 | ~434 KB |
| `GameSR-Std` | `sr_std` | `ch=16`, `num_pairs=2`, `scale=2` | Bilinear 2× + `pixel_shuffle` residual | ~11,548 | ~22.6 KB |
| `GameSR-Lite` | `sr_lite` | `ch=12`, `num_pairs=2`, `scale=2` | Bilinear 2× + `pixel_shuffle` residual | ~6,936 | ~13.5 KB |

## Design Invariants

- **BGR `[0, 1]` input normalization**: `bgr_to_luma()` matches WGSL/GLSL (`0.114*B + 0.587*G + 0.299*R`).
- **Continuous time conditioning**: `FilmHead` maps scalar `t ∈ (0, 1)` via a 64-unit MLP (`evalFilmMlp` on JS side), supporting arbitrary `2×`, `3×`, or fractional phase interpolation from a single model.
- **Backward compatibility aliases**: `GameFINet = FrameFlowNet` and `GameFILoss = FrameFlowLoss` remain exported from `frameflow/__init__.py`.
- **Fixed 2× scaling in GameSR**: `SrModelConfig` rejects `scale != 2` at construction time (`render_sr` op assumes 2× `pixel_shuffle`).
