# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Model architecture presets for GameSR-2x (real-time browser 2x AI super-resolution).

All feature channel counts are multiples of 4 (16 / 24 / 32) so WebGL2 packs 4 channels
per RGBA16F texture layer and executes up to 4 layers (16 channels) in a single MRT draw call.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SrModelConfig:
    name: str
    ch: int
    num_pairs: int  # Each pair = 2 residual conv3x3+PReLU layers with residual skip
    scale: int = 2

    def __post_init__(self) -> None:
        if self.ch % 4 != 0 or self.ch <= 0:
            raise ValueError(f"ch={self.ch} must be a positive multiple of 4")
        if self.num_pairs < 1:
            raise ValueError(f"num_pairs={self.num_pairs} must be >= 1")
        if self.scale != 2:
            raise ValueError(f"Only scale=2 is supported cleanly by render_sr (got {self.scale})")

    def to_meta_cfg(self) -> dict:
        return asdict(self)


SR_PRESETS: dict[str, SrModelConfig] = {
    # Ultra-fast WebGPU/WebGL2 2x preset: stem + 2 res pairs (4 body convs) + head = 6 convs (~6.9K params, ~13.5 KB f16)
    "sr_lite": SrModelConfig(
        name="GameSR-Lite",
        ch=12,
        num_pairs=2,
        scale=2,
    ),
    # Higher-detail preset: stem + 2 res pairs (4 body convs) + head = 6 convs (~11.5K params, ~22.6 KB f16)
    "sr_std": SrModelConfig(
        name="GameSR-Std",
        ch=16,
        num_pairs=2,
        scale=2,
    ),
}


def get_sr_preset(name: str) -> SrModelConfig:
    key = name.lower().replace("-", "_")
    aliases = {"lite": "sr_lite", "std": "sr_std", "standard": "sr_std"}
    key = aliases.get(key, key)
    if key not in SR_PRESETS:
        raise ValueError(f"Unknown SR preset {name!r}; expected one of {sorted(SR_PRESETS)}")
    return SR_PRESETS[key]
