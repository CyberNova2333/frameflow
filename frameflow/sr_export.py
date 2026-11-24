# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Exporter for GameSRNet -> (graph.json, model.json, model.bin) in FrameFlow format_version=1."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .sr_config import SrModelConfig
from .sr_model import GameSRNet


def build_sr_graph_json(cfg: SrModelConfig) -> dict:
    ops: list[dict] = [
        {"in": ["A"], "out": ["h0"], "w": ["stem.0.weight", "stem.0.bias", "stem.1.weight"], "stride": 1, "op": "conv"},
    ]
    prev = "h0"
    for idx in range(cfg.num_pairs):
        t1 = f"b{idx}_1"
        t2 = f"b{idx}_2"
        hout = f"h{idx + 1}"
        ops.extend([
            {
                "in": [prev],
                "out": [t1],
                "w": [f"body.{idx}.c1.0.weight", f"body.{idx}.c1.0.bias", f"body.{idx}.c1.1.weight"],
                "stride": 1,
                "op": "conv",
            },
            {
                "in": [t1],
                "out": [t2],
                "w": [f"body.{idx}.c2.0.weight", f"body.{idx}.c2.0.bias", f"body.{idx}.c2.1.weight"],
                "stride": 1,
                "op": "conv",
            },
            {"in": [t2, prev], "out": [hout], "op": "add"},
        ])
        prev = hout

    ops.append({"in": [prev], "out": ["delta_lr"], "w": ["head.weight", "head.bias"], "stride": 1, "op": "conv2d"})
    ops.append({"in": ["A", "delta_lr"], "out": ["render"], "scale": cfg.scale, "output": True, "op": "render_sr"})

    return {
        "meta": {
            "model": cfg.name,
            "task": "sr",
            "scale": cfg.scale,
            "format_version": 1,
            "weights": "model.bin/.json",
            "inputs": {"A": "[B,3,H,W] in [0,1] BGR"},
            "outputs": ["render"],
        },
        "ops": ops,
    }


def export_sr_model(
    model: GameSRNet,
    out_dir: str | Path,
    train_step: int = 0,
    stem: str = "model",
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg

    state = model.state_dict()
    tensors_meta: dict[str, dict] = {}
    chunks: list[np.ndarray] = []
    offset_elems = 0

    for name, tensor in state.items():
        arr = tensor.detach().cpu().contiguous().to(torch.float16).numpy().astype("<f2", copy=False)
        tensors_meta[name] = {
            "shape": list(tensor.shape),
            "offset": offset_elems,
            "dtype": "f16",
        }
        chunks.append(arr.ravel())
        offset_elems += arr.size

    blob = np.concatenate(chunks) if chunks else np.empty((0,), dtype="<f2")
    bin_path = out_path / f"{stem}.bin"
    bin_path.write_bytes(blob.tobytes())

    manifest = {
        "meta": {
            "model": cfg.name,
            "task": "sr",
            "scale": cfg.scale,
            "format_version": 1,
            "train_step": int(train_step),
            "cfg": cfg.to_meta_cfg(),
            "dtype": "f16",
            "bytes_per_elem": 2,
            "endian": "little",
            "n_tensors": len(tensors_meta),
            "n_params": int(offset_elems),
            "render_sampling": "bilinear",
        },
        "tensors": tensors_meta,
    }
    manifest_path = out_path / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    graph = build_sr_graph_json(cfg)
    graph_path = out_path / "graph.json"
    graph_path.write_text(json.dumps(graph, indent=1) + "\n", encoding="utf-8")

    return {
        "dir": str(out_path),
        "bin": str(bin_path),
        "manifest": str(manifest_path),
        "graph": str(graph_path),
        "n_tensors": len(tensors_meta),
        "n_params": int(offset_elems),
        "bin_bytes": int(blob.nbytes),
    }
