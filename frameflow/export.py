# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Exporter for FrameFlow models -> (graph.json, model.json, model.bin).

Produces format_version=1 artifacts compatible with both:
  1. Apache-2.0 FlyHigh WebGPU runtime (`FhEngine.loadModel(graph, manifest, weights)`)
  2. FrameFlow WebGL2 Fragment-Shader universal browser runtime
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .config import ModelConfig
from .model import FrameFlowNet


def build_graph_json(cfg: ModelConfig) -> dict:
    """Dynamically construct graph.json matching ModelConfig channels & radii."""
    ops: list[dict] = [
        {"in": ["A"], "out": ["aq"], "scale": 0.25, "mode": "bilinear", "align_corners": False, "antialias": True, "op": "interpolate"},
        {"in": ["B"], "out": ["bq"], "scale": 0.25, "mode": "bilinear", "align_corners": False, "antialias": True, "op": "interpolate"},
        # Encoder A
        {"in": ["aq"], "out": ["a_e1"], "w": ["enc.e1.0.weight", "enc.e1.0.bias", "enc.e1.1.weight"], "stride": 2, "op": "conv"},
        {"in": ["a_e1"], "out": ["a_e2"], "w": ["enc.e2.0.weight", "enc.e2.0.bias", "enc.e2.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["a_e2", "a_e1"], "out": ["f8a"], "op": "add"},
        {"in": ["f8a"], "out": ["a_e3"], "w": ["enc.e3.0.weight", "enc.e3.0.bias", "enc.e3.1.weight"], "stride": 2, "op": "conv"},
        {"in": ["a_e3"], "out": ["a_e4"], "w": ["enc.e4.0.weight", "enc.e4.0.bias", "enc.e4.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["a_e4", "a_e3"], "out": ["f16a"], "op": "add"},
        # Encoder B
        {"in": ["bq"], "out": ["b_e1"], "w": ["enc.e1.0.weight", "enc.e1.0.bias", "enc.e1.1.weight"], "stride": 2, "op": "conv"},
        {"in": ["b_e1"], "out": ["b_e2"], "w": ["enc.e2.0.weight", "enc.e2.0.bias", "enc.e2.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["b_e2", "b_e1"], "out": ["f8b"], "op": "add"},
        {"in": ["f8b"], "out": ["b_e3"], "w": ["enc.e3.0.weight", "enc.e3.0.bias", "enc.e3.1.weight"], "stride": 2, "op": "conv"},
        {"in": ["b_e3"], "out": ["b_e4"], "w": ["enc.e4.0.weight", "enc.e4.0.bias", "enc.e4.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["b_e4", "b_e3"], "out": ["f16b"], "op": "add"},
        # Image pyramid
        {"in": ["aq"], "out": ["i8a"], "k": 2, "op": "avgpool"},
        {"in": ["bq"], "out": ["i8b"], "k": 2, "op": "avgpool"},
        {"in": ["i8a"], "out": ["i16a"], "k": 2, "op": "avgpool"},
        {"in": ["i8b"], "out": ["i16b"], "k": 2, "op": "avgpool"},
        {"in": ["i16a", "i16b"], "out": ["img16"], "axis": 1, "op": "concat"},
        {"in": ["f16a"], "out": ["side"], "ch": cfg.side_ch, "op": "zeros"},
        {"in": ["f16a"], "out": ["state"], "ch": cfg.state_ch, "op": "zeros"},
        {"in": ["img16"], "out": ["img16a"], "a": 0, "b": 3, "op": "slice"},
        {"in": ["img16"], "out": ["img16b"], "a": 3, "b": 6, "op": "slice"},
        {"in": ["img16a"], "out": ["la"], "op": "luma"},
        {"in": ["img16b"], "out": ["lb"], "op": "luma"},
        {"in": ["la", "lb"], "out": ["cv"], "radius": cfg.cost_radius, "op": "sad_volume"},
        {"in": ["f16a", "f16b", "img16", "side", "state", "cv"], "out": ["cin"], "axis": 1, "op": "concat"},
        # Coarse net
        {"in": ["cin"], "out": ["ch1"], "w": ["coarse.c1.0.weight", "coarse.c1.0.bias", "coarse.c1.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["ch1"], "out": ["ch2"], "w": ["coarse.c2.0.weight", "coarse.c2.0.bias", "coarse.c2.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["ch2"], "out": ["ch3"], "w": ["coarse.c3.0.weight", "coarse.c3.0.bias", "coarse.c3.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["ch3", "ch1"], "out": ["ch3r"], "op": "add"},
        {"in": ["ch3r"], "out": ["hc"], "w": ["coarse.c4.0.weight", "coarse.c4.0.bias", "coarse.c4.1.weight"], "stride": 1, "op": "conv"},
        {"in": ["hc"], "out": ["o"], "w": ["coarse.head.weight", "coarse.head.bias"], "op": "conv2d"},
        {"in": ["o"], "out": ["o04"], "a": 0, "b": 4, "op": "slice"},
        {"in": ["o04"], "out": ["flow16"], "factor": 16.0, "op": "scale"},
        {"in": ["o"], "out": ["vis16"], "a": 4, "b": 7, "op": "slice"},
        {"in": ["o"], "out": ["conf"], "a": 7, "b": 8, "output": True, "op": "slice"},
        {"in": ["o"], "out": ["gate"], "a": 8, "b": 9, "output": True, "op": "slice"},
        {"in": ["o"], "out": ["sal"], "a": 9, "b": 10, "output": True, "op": "slice"},
        {"in": ["flow16"], "out": ["flow"], "scale": 2, "mode": "bilinear", "align_corners": False, "op": "interpolate"},
        {"in": ["vis16"], "out": ["vis"], "scale": 2, "mode": "bilinear", "align_corners": False, "op": "interpolate"},
        {"in": ["flow"], "out": ["h"], "ch": cfg.cr, "op": "zeros"},
        # Fine recurrent refinement (K iterations)
        {
            "times": "K",
            "carry": ["flow", "vis", "h"],
            "body": [
                {"in": ["flow"], "out": ["flow8"], "factor": 0.125, "op": "scale"},
                {"in": ["flow8"], "out": ["flow8f"], "a": 0, "b": 2, "op": "slice"},
                {"in": ["flow8"], "out": ["flow8b"], "a": 2, "b": 4, "op": "slice"},
                {"in": ["f8a", "flow8f"], "out": ["wa"], "op": "warp"},
                {"in": ["f8b", "flow8b"], "out": ["wb"], "op": "warp"},
                {"in": ["i8a", "flow8f"], "out": ["ia"], "op": "warp"},
                {"in": ["i8b", "flow8b"], "out": ["ib"], "op": "warp"},
                {"in": ["ia"], "out": ["fla"], "op": "luma"},
                {"in": ["ib"], "out": ["flb"], "op": "luma"},
                {"in": ["fla", "flb"], "out": ["fcv"], "radius": cfg.fine_cost_radius, "op": "sad_volume"},
                {"in": ["wa", "wb", "ia", "ib", "flow8", "h", "fcv"], "out": ["fx"], "axis": 1, "op": "concat"},
                {"in": ["fx"], "out": ["fr1"], "w": ["fine.r1.0.weight", "fine.r1.0.bias", "fine.r1.1.weight"], "stride": 1, "op": "conv"},
                {"in": ["fr1"], "out": ["fr2"], "w": ["fine.r2.0.weight", "fine.r2.0.bias", "fine.r2.1.weight"], "stride": 1, "op": "conv"},
                {"in": ["fr2"], "out": ["fr3"], "w": ["fine.r3.0.weight", "fine.r3.0.bias", "fine.r3.1.weight"], "stride": 1, "op": "conv"},
                {"in": ["fr3"], "out": ["d"], "w": ["fine.out.weight", "fine.out.bias"], "op": "conv2d"},
                {"in": ["d"], "out": ["d04"], "a": 0, "b": 4, "op": "slice"},
                {"in": ["flow", "d04"], "out": ["flow"], "op": "add"},
                {"in": ["d"], "out": ["d47"], "a": 4, "b": 7, "op": "slice"},
                {"in": ["vis", "d47"], "out": ["vis"], "op": "add"},
                {"in": ["fr3"], "out": ["h"], "op": "copy"},
            ],
            "op": "repeat",
        },
        {"in": ["f16b", "flow16"], "out": ["state_out"], "axis": 1, "output": True, "op": "concat"},
        # Mid-frame stage (phase t)
        {"in": ["flow"], "out": ["flowf"], "a": 0, "b": 2, "op": "slice"},
        {"in": ["flow"], "out": ["flowb"], "a": 2, "b": 4, "op": "slice"},
        {"in": ["flowf"], "out": ["linf"], "mode": "fwd", "op": "mul_t"},
        {"in": ["flowb"], "out": ["linb"], "mode": "bwd", "op": "mul_t"},
        {"in": ["linf", "linb"], "out": ["lin"], "axis": 1, "op": "concat"},
        {"in": ["t"], "out": ["g"], "w": ["film.film.0.weight", "film.film.0.bias", "film.film.2.weight", "film.film.2.bias"], "cond": "use_film", "op": "film_mlp"},
        {"in": ["h", "g"], "out": ["hm"], "cr": cfg.cr, "cond": "use_film", "op": "film_modulate"},
        {"in": ["hm"], "out": ["fh1"], "w": ["film.h1.0.weight", "film.h1.0.bias", "film.h1.1.weight"], "stride": 1, "cond": "use_film", "op": "conv"},
        {"in": ["fh1"], "out": ["fh2"], "w": ["film.h2.0.weight", "film.h2.0.bias", "film.h2.1.weight"], "stride": 1, "cond": "use_film", "op": "conv"},
        {"in": ["fh2"], "out": ["dfilm"], "w": ["film.out.weight", "film.out.bias"], "cond": "use_film", "op": "conv2d"},
        {"in": ["dfilm"], "out": ["d_flow"], "a": 0, "b": 4, "cond": "use_film", "op": "slice"},
        {"in": ["dfilm"], "out": ["d_mask"], "a": 4, "b": 5, "cond": "use_film", "op": "slice"},
        {"in": ["lin", "d_flow"], "out": ["flow_t"], "cond": "use_film", "else_alias": {"flow_t": "lin"}, "op": "add"},
        {"in": ["d_mask"], "out": ["nd"], "cond": "use_film", "op": "neg"},
        {"in": ["d_mask"], "out": ["zd"], "cond": "use_film", "op": "zeros_like"},
        {"in": ["d_mask", "nd", "zd"], "out": ["dvis"], "axis": 1, "cond": "use_film", "op": "concat"},
        {"in": ["vis", "dvis"], "out": ["vis_t"], "cond": "use_film", "else_alias": {"vis_t": "vis"}, "op": "add"},
        {"in": ["flow_t", "vis_t"], "out": ["ft_vt"], "axis": 1, "op": "concat"},
        {"in": ["h", "ft_vt"], "out": ["up7"], "w": ["up.w.weight", "up.w.bias"], "op": "convex_upsample"},
        {"in": ["up7"], "out": ["flow_tq"], "a": 0, "b": 4, "output": True, "op": "slice"},
        {"in": ["up7"], "out": ["vis_tq"], "a": 4, "b": 7, "output": True, "op": "slice"},
        # Quarter-res residual refine head
        {"in": ["flow_tq"], "out": ["flow_tq4"], "factor": 0.25, "cond": "use_refine", "op": "scale"},
        {"in": ["flow_tq4"], "out": ["ftq4f"], "a": 0, "b": 2, "cond": "use_refine", "op": "slice"},
        {"in": ["flow_tq4"], "out": ["ftq4b"], "a": 2, "b": 4, "cond": "use_refine", "op": "slice"},
        {"in": ["aq", "ftq4f"], "out": ["rwa"], "cond": "use_refine", "op": "warp"},
        {"in": ["bq", "ftq4b"], "out": ["rwb"], "cond": "use_refine", "op": "warp"},
        {"in": ["vis_tq"], "out": ["bwA", "bwB"], "cond": "use_refine", "op": "blend_weights"},
        {"in": ["rwa", "bwA"], "out": ["bt1"], "cond": "use_refine", "op": "mul"},
        {"in": ["rwb", "bwB"], "out": ["bt2"], "cond": "use_refine", "op": "mul"},
        {"in": ["bt1", "bt2"], "out": ["blend"], "cond": "use_refine", "op": "add"},
        {"in": ["flow_tq"], "out": ["flow_tq20"], "factor": 0.05, "cond": "use_refine", "op": "scale"},
        {"in": ["blend"], "out": ["lg"], "cond": "use_refine", "op": "luma_grad"},
        {"in": ["rwa", "rwb", "vis_tq", "flow_tq20", "lg"], "out": ["rx"], "axis": 1, "cond": "use_refine", "op": "concat"},
        {"in": ["rx"], "out": ["rq1"], "w": ["refine.q1.0.weight", "refine.q1.0.bias", "refine.q1.1.weight"], "stride": 1, "cond": "use_refine", "op": "conv"},
        {"in": ["rq1"], "out": ["rq2"], "w": ["refine.q2.0.weight", "refine.q2.0.bias", "refine.q2.1.weight"], "stride": 1, "cond": "use_refine", "op": "conv"},
        {"in": ["rq2"], "out": ["rq3"], "w": ["refine.q3.0.weight", "refine.q3.0.bias", "refine.q3.1.weight"], "stride": 1, "cond": "use_refine", "op": "conv"},
        {"in": ["rq3"], "out": ["rr"], "w": ["refine.q4.weight", "refine.q4.bias"], "cond": "use_refine", "op": "conv2d"},
        {"in": ["rr"], "out": ["R_t"], "output": True, "cond": "use_refine", "op": "affine_out"},
        {"in": ["A", "B", "flow_tq", "vis_tq", "R_t"], "out": ["render"], "output": True, "op": "render_pix"},
    ]
    return {
        "meta": {
            "model": cfg.name,
            "format_version": 1,
            "weights": "model.bin/.json",
            "inputs": {
                "A": "[B,3,H,W] in [0,1] BGR, H,W % 32 == 0",
                "B": "same",
                "t": "[B] in (0,1)",
                "K": "int in [1,3]",
            },
            "params": ["use_film (default true)", "use_refine (default true)"],
            "outputs": ["render", "flow_tq", "vis_tq", "R_t", "conf", "gate", "sal", "state_out"],
        },
        "ops": ops,
    }


def export_model(
    model: FrameFlowNet,
    out_dir: str | Path,
    train_step: int = 0,
    stem: str = "model",
) -> dict:
    """Export FrameFlowNet to `(graph.json, <stem>.json, <stem>.bin)` (f16 little-endian)."""
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
            "format_version": 1,
            "train_step": int(train_step),
            "cfg": cfg.to_meta_cfg(),
            "default_K": cfg.k_default,
            "K_range": [1, 3],
            "side_ch": cfg.side_ch,
            "cost_radius": cfg.cost_radius,
            "dtype": "f16",
            "bytes_per_elem": 2,
            "endian": "little",
            "n_tensors": len(tensors_meta),
            "n_params": int(offset_elems),
            "render_sampling": cfg.render_sampling,
        },
        "tensors": tensors_meta,
    }
    manifest_path = out_path / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    graph = build_graph_json(cfg)
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
