# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Loss functions for GameSR-2x super-resolution (crisp edges, de-pixelation, zero blur).

Combines:
  - Charbonnier pixel fidelity loss
  - Multi-scale Laplacian pyramid reconstruction loss
  - Differentiable 3x3 Census structure loss on luma (sharpens anime/UI lines without overshoot)
  - Spatial edge-gradient Sobel L1 loss (enforces crisp restored contours & text strokes)
  - Optional Student <- Teacher knowledge distillation loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import census_loss, charbonnier, laplacian_loss
from .ops import bgr_to_luma


def edge_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pl = bgr_to_luma(pred.float())
    tl = bgr_to_luma(target.float())
    pdx = pl[:, :, :, 1:] - pl[:, :, :, :-1]
    pdy = pl[:, :, 1:, :] - pl[:, :, :-1, :]
    tdx = tl[:, :, :, 1:] - tl[:, :, :, :-1]
    tdy = tl[:, :, 1:, :] - tl[:, :, :-1, :]
    return 0.5 * (charbonnier(pdx - tdx).mean() + charbonnier(pdy - tdy).mean())


class GameSRLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 1.0,
        w_lap: float = 0.45,
        w_census: float = 0.30,
        w_grad: float = 0.35,
        w_kd: float = 0.25,
    ) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_lap = w_lap
        self.w_census = w_census
        self.w_grad = w_grad
        self.w_kd = w_kd

    def forward(
        self,
        out: dict[str, torch.Tensor],
        gt_hr: torch.Tensor,
        teacher_out: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = out["render"].float()
        gt = gt_hr.float()

        l_l1 = charbonnier(pred - gt).mean()
        l_lap = laplacian_loss(pred, gt, levels=3)
        l_cen = census_loss(pred, gt)
        l_grad = edge_gradient_loss(pred, gt)

        total = (
            self.w_l1 * l_l1
            + self.w_lap * l_lap
            + self.w_census * l_cen
            + self.w_grad * l_grad
        )
        metrics = {
            "l1": float(l_l1.item()),
            "lap": float(l_lap.item()),
            "census": float(l_cen.item()),
            "grad": float(l_grad.item()),
        }
        if teacher_out is not None:
            l_kd = charbonnier(pred - teacher_out["render"].float().detach()).mean()
            total = total + self.w_kd * l_kd
            metrics["kd"] = float(l_kd.item())
        metrics["total"] = float(total.item())
        return total, metrics
