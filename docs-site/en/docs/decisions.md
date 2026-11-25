# Decisions & Compliance ADRs

| Decision | Choice Kept | Rejected Alternative & Rationale | Source Anchor |
| --- | --- | --- | --- |
| Runtime footprint | Zero-npm / zero-WASM pure ES modules | Avoids ONNX/WASM heap competition with web emulators and enables direct GPU resource sharing | `package.json`, `runtime/` |
| Unsupported fallback | Explicit rejection (`supported: false`) | CPU/WASM fallback would starve emulator worker threads | `runtime/index.js` |
| Hidden channel sizing | Positive multiples of 4 enforced at init | Maps cleanly onto `RGBA16F` 4-channel layers and 4-target MRT fragment shaders | `frameflow/config.py`, `sr_config.py` |
| Weight distribution | Excluded from Git; shipped via Release / OCI | Keeps repo history clean and attaches `DATA_PROVENANCE.json` + 5 license documents | `.gitignore`, `scripts/package_release.py` |
| Static UI protection | Synthetic HUD overlay + zero-flow loss | Prevents background motion from tearing static game bars, minimaps, and dialogue typography | `frameflow/dataset.py`, `losses.py` |
