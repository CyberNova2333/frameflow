# Code Map by Responsibility

| Path | Responsibility | Key Files | Main Role |
| --- | --- | --- | --- |
| `frameflow/` | GameFI & GameSR models, ops, losses, datasets, and exporters | `model.py`, `sr_model.py`, `export.py` | Exports `format_version = 1` model artifacts |
| `runtime/` | Zero-dependency WebGPU & WebGL2 runtimes, loader, and pacer | `index.js`, `loader.js`, `pacer.js` | Browser probe, compilation, inference, pacing |
| `scripts/` | Open-media prep, `bf16` training/KD, Edge parity, and packaging | `train.py`, `verify_backends.py`, `package_release.py` | Produces verified weights & compliance tarballs |
| `tests/` | Fast CPU PyTorch checks + Node loader & pacer tests | `test_frameflow.py`, `test_runtime_node.mjs` | Zero-GPU unit regression barrier |

For Chinese line-level anchor tables and audit evidence, see [`repo-docs/code-map.md`](https://github.com/CyberNova2333/frameflow/blob/main/repo-docs/code-map.md).
