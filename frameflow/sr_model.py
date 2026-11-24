# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""GameSRNet: Lightweight browser-native 2x AI super-resolution model.

Input tensor `x` is BGR in [0, 1], shape [B, 3, H, W] (any H, W >= 8).
Outputs `render` in [0, 1], shape [B, 3, 2H, 2W], as:
    clamp(bilinear_2x(x) + pixel_shuffle_2x(head(backbone(x))), 0, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sr_config import SrModelConfig, get_sr_preset


def _conv_prelu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=True),
        nn.PReLU(num_parameters=cout, init=0.25),
    )


class ResPair(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = _conv_prelu(ch, ch)
        self.c2 = _conv_prelu(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2(self.c1(x)) + x


class GameSRNet(nn.Module):
    def __init__(self, cfg: SrModelConfig | str = "sr_lite") -> None:
        super().__init__()
        self.cfg = get_sr_preset(cfg) if isinstance(cfg, str) else cfg
        self.stem = _conv_prelu(3, self.cfg.ch)
        self.body = nn.ModuleList([ResPair(self.cfg.ch) for _ in range(self.cfg.num_pairs)])
        self.head = nn.Conv2d(self.cfg.ch, 3 * (self.cfg.scale ** 2), kernel_size=3, stride=1, padding=1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        # Near-zero head init starts smoothly from bilinear 2x baseline
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f0 = self.stem(x)
        h = f0
        for pair in self.body:
            h = pair(h)
        delta_lr = self.head(h)
        delta_hr = F.pixel_shuffle(delta_lr, self.cfg.scale)
        base_hr = F.interpolate(x, scale_factor=self.cfg.scale, mode="bilinear", align_corners=False)
        render = (base_hr + delta_hr).clamp(0.0, 1.0)
        return {
            "render": render,
            "delta_lr": delta_lr,
            "delta_hr": delta_hr,
            "base_hr": base_hr,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
