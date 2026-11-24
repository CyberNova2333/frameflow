# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Model architecture presets for GameFI (FrameFlow optical-flow frame interpolation).

All hidden channel counts are multiples of 4 to enable optimal RGBA16F vec4
texture packing in WebGL2 fragment shaders as well as vectorized WebGPU kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelConfig:
    name: str
    c8: int
    c16: int
    cc: int
    cr: int
    cq: int
    side_ch: int = 4
    k_default: int = 2
    cost_radius: int = 3
    fine_cost_radius: int = 2
    film_hidden: int = 64  # Fixed at 64 for 1:1 compatibility with FlyHigh WebGPU runtime
    coarse_head_ch: int = 12
    render_sampling: str = "bicubic"

    def __post_init__(self) -> None:
        for field_name in ("c8", "c16", "cc", "cr", "cq", "side_ch"):
            val = getattr(self, field_name)
            if val % 4 != 0 or val <= 0:
                raise ValueError(f"{field_name}={val} must be a positive multiple of 4")

    @property
    def state_ch(self) -> int:
        return self.c16 + 4

    @property
    def coarse_cost_ch(self) -> int:
        return (2 * self.cost_radius + 1) ** 2

    @property
    def fine_cost_ch(self) -> int:
        return (2 * self.fine_cost_radius + 1) ** 2

    @property
    def coarse_in_ch(self) -> int:
        # f16a + f16b + img16(6) + side + state + cv
        return self.c16 * 2 + 6 + self.side_ch + self.state_ch + self.coarse_cost_ch

    @property
    def fine_in_ch(self) -> int:
        # wa + wb + ia(3) + ib(3) + flow8(4) + h(cr) + fcv
        return self.c8 * 2 + 6 + 4 + self.cr + self.fine_cost_ch

    def to_meta_cfg(self) -> dict:
        d = asdict(self)
        return {
            "c8": d["c8"],
            "c16": d["c16"],
            "cc": d["cc"],
            "cr": d["cr"],
            "cq": d["cq"],
            "side_ch": d["side_ch"],
            "k_default": d["k_default"],
            "cost_radius": d["cost_radius"],
            "fine_cost_radius": d["fine_cost_radius"],
        }


PRESETS: dict[str, ModelConfig] = {
    # 1:1 isomorphic to FlyHighFG (775,811 params, ~1.52 MB f16)
    "std": ModelConfig(
        name="GameFI-Std",
        c8=36,
        c16=72,
        cc=96,
        cr=48,
        cq=24,
        side_ch=4,
        k_default=2,
        cost_radius=3,
        fine_cost_radius=2,
        render_sampling="bicubic",
    ),
    # Distilled lightweight preset (222,179 params, ~434 KB f16) for WebGL2 & mobile GPUs
    "lite": ModelConfig(
        name="GameFI-Lite",
        c8=20,
        c16=40,
        cc=48,
        cr=24,
        cq=16,
        side_ch=4,
        k_default=1,
        cost_radius=2,
        fine_cost_radius=1,
        render_sampling="bilinear",
    ),
}


def get_preset(name: str) -> ModelConfig:
    key = name.lower()
    if key == "standard":
        key = "std"
    if key not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; expected one of {sorted(PRESETS)}")
    return PRESETS[key]
