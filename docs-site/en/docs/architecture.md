# Architecture & Interactive Diagram

FrameFlow decouples Python authoring/export (`frameflow/`) from zero-dependency browser execution (`runtime/`) across one `format_version = 1` interchange specification.

<DiagramFrame
  src="/diagrams/frameflow-architecture.html"
  title="FrameFlow Export, Dual-Backend Runtime & Host Zero-Copy Architecture"
  aspect="940 / 330"
  new-tab-label="Open Fullscreen ↗"
/>

## Deep-Dive Links & Verification

- [End-to-End Walkthrough](/en/docs/walkthrough): Five-stage execution lifecycle and failure gates.
- [Code Map](/en/docs/code-map): Directory and symbol ownership across `frameflow/`, `runtime/`, `scripts/`, and `tests/`.
- [Decisions & Compliance ADRs](/en/docs/decisions): Key engineering tradeoffs and audit scope.
- Chinese source-truth guides live in [`repo-docs/`](https://github.com/CyberNova2333/frameflow/tree/main/repo-docs).

```sh
make test            # Fast CPU Python + Node loader/pacer unit suite
make verify-runtime  # Headless Edge 157 dual-backend parity vs PyTorch (requires GPU/Vulkan)
```
