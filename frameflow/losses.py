# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Loss functions for GameFI training and GameFI-Std -> GameFI-Lite distillation.

Combines:
  - Multi-scale Charbonnier + Laplacian pyramid reconstruction loss
  - Differentiable Census (ternary) structure loss for crisp edges
  - Pre-residual warped blend supervision (forces optical flow to align motion)
  - Edge-aware flow smoothness & residual L1 sparsity
  - Explicit static HUD zero-flow + pixel fidelity loss
  - Optional Knowledge Distillation loss from a frozen Standard teacher
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import backward_warp, bgr_to_luma, blend_weights


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def _gauss_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k2 = torch.outer(k, k)
    k2 = k2 / k2.sum()
    return k2.view(1, 1, 5, 5).repeat(3, 1, 1, 1)


def _lap_pyr(img: torch.Tensor, levels: int = 4) -> list[torch.Tensor]:
    kernel = _gauss_kernel(img.device, img.dtype)
    pyr = []
    cur = img
    for _ in range(levels):
        blurred = F.conv2d(F.pad(cur, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)
        down = blurred[:, :, ::2, ::2]
        up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        pyr.append(cur - up)
        cur = down
    pyr.append(cur)
    return pyr


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor, levels: int = 4) -> torch.Tensor:
    p_pyr = _lap_pyr(pred.float(), levels)
    t_pyr = _lap_pyr(target.float(), levels)
    loss = pred.new_zeros(())
    for i, (p, t) in enumerate(zip(p_pyr, t_pyr)):
        loss = loss + (2.0**i) * charbonnier(p - t).mean()
    return loss / (2.0**levels)


def census_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft 3x3 Census transform loss on luma (sharpens anime/3D edges without blur)."""
    p = bgr_to_luma(pred.float())
    t = bgr_to_luma(target.float())
    # Compare each pixel with its 3x3 neighborhood
    p_unf = F.unfold(p, kernel_size=3, padding=1)  # [B, 9, H*W]
    t_unf = F.unfold(t, kernel_size=3, padding=1)
    p_diff = p_unf - p.flatten(2)
    t_diff = t_unf - t.flatten(2)
    # Smooth sign normalization
    p_c = p_diff / torch.sqrt(p_diff * p_diff + 1e-4)
    t_c = t_diff / torch.sqrt(t_diff * t_diff + 1e-4)
    return charbonnier(p_c - t_c, eps=1e-2).mean()


def edge_aware_smoothness(flow: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """First-order edge-weighted TV smoothness on 4-channel flow."""
    img_s = F.interpolate(img.float(), size=flow.shape[-2:], mode="bilinear", align_corners=False)
    dx_img = (img_s[:, :, :, 1:] - img_s[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    dy_img = (img_s[:, :, 1:, :] - img_s[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    wx = torch.exp(-10.0 * dx_img)
    wy = torch.exp(-10.0 * dy_img)
    dx_f = charbonnier(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dy_f = charbonnier(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    return (wx * dx_f).mean() + (wy * dy_f).mean()


class FrameFlowLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 1.0,
        w_lap: float = 0.5,
        w_census: float = 0.25,
        w_warp: float = 0.35,
        w_coarse: float = 0.15,
        w_res_reg: float = 0.02,
        w_smooth: float = 0.01,
        w_hud: float = 0.5,
        w_kd_flow: float = 0.08,
        w_kd_render: float = 0.3,
    ) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_lap = w_lap
        self.w_census = w_census
        self.w_warp = w_warp
        self.w_coarse = w_coarse
        self.w_res_reg = w_res_reg
        self.w_smooth = w_smooth
        self.w_hud = w_hud
        self.w_kd_flow = w_kd_flow
        self.w_kd_render = w_kd_render

    def forward(
        self,
        out: dict[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        gt: torch.Tensor,
        t: torch.Tensor,
        hud_mask: torch.Tensor | None = None,
        teacher_out: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = out["render"].float()
        gt_f = gt.float()

        l_l1 = charbonnier(pred - gt_f).mean()
        l_lap = laplacian_loss(pred, gt_f)
        l_cen = census_loss(pred, gt_f)

        # Pre-residual warped blend loss at quarter-res (keeps optical flow honest)
        flow_tq = out["flow_tq"].float()
        vis_tq = out["vis_tq"].float()
        aq = F.interpolate(a.float(), size=flow_tq.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
        bq = F.interpolate(b.float(), size=flow_tq.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
        gtq = F.interpolate(gt_f, size=flow_tq.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
        wa_q = backward_warp(aq, flow_tq[:, 0:2] * 0.25, mode="bilinear")
        wb_q = backward_warp(bq, flow_tq[:, 2:4] * 0.25, mode="bilinear")
        bw_a, bw_b = blend_weights(vis_tq)
        blend_q = wa_q * bw_a + wb_q * bw_b
        l_warp = charbonnier(blend_q - gtq).mean()

        # Coarse flow photometric supervision at 1/16 res
        t_4d = t.float().view(-1, 1, 1, 1)
        c_flow = out["coarse_flow"].float()
        c_vis = out["coarse_vis"].float()
        a16 = F.interpolate(a.float(), size=c_flow.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
        b16 = F.interpolate(b.float(), size=c_flow.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
        gt16 = F.interpolate(gt_f, size=c_flow.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
        cf_f = c_flow[:, 0:2] * (2.0 * t_4d) * (1.0 / 16.0)
        cf_b = c_flow[:, 2:4] * (2.0 * (1.0 - t_4d)) * (1.0 / 16.0)
        cwa = backward_warp(a16, cf_f, mode="bilinear")
        cwb = backward_warp(b16, cf_b, mode="bilinear")
        cbw_a, cbw_b = blend_weights(c_vis)
        l_coarse = charbonnier(cwa * cbw_a + cwb * cbw_b - gt16).mean()

        # Keep residual small so flow carries motion
        l_res = out["R_t"].float().abs().mean()
        # Edge-aware smoothness on quarter-res flow (scaled down to pixel units)
        l_smooth = edge_aware_smoothness(flow_tq * 0.1, gt_f)

        total = (
            self.w_l1 * l_l1
            + self.w_lap * l_lap
            + self.w_census * l_cen
            + self.w_warp * l_warp
            + self.w_coarse * l_coarse
            + self.w_res_reg * l_res
            + self.w_smooth * l_smooth
        )

        metrics = {
            "l1": float(l_l1.detach()),
            "lap": float(l_lap.detach()),
            "cen": float(l_cen.detach()),
            "warp": float(l_warp.detach()),
            "res": float(l_res.detach()),
        }

        # Explicit static HUD supervision: zero flow + ultra-sharp reconstruction on HUD mask
        if hud_mask is not None and hud_mask.sum() > 0:
            hm = hud_mask.float()
            hm_q = F.interpolate(hm, size=flow_tq.shape[-2:], mode="area")
            hud_pix = (charbonnier(pred - gt_f) * hm).sum() / (hm.sum() * 3.0 + 1e-6)
            hud_flow = (flow_tq.abs() * hm_q).sum() / (hm_q.sum() * 4.0 + 1e-6)
            l_hud = hud_pix + 0.05 * hud_flow
            total = total + self.w_hud * l_hud
            metrics["hud"] = float(l_hud.detach())

        # Optional Standard -> Lite Knowledge Distillation
        if teacher_out is not None:
            t_flow = teacher_out["flow_tq"].detach().float()
            t_vis = teacher_out["vis_tq"].detach().float()
            t_ren = teacher_out["render"].detach().float()
            l_kd_f = charbonnier(flow_tq - t_flow).mean() + 0.2 * charbonnier(vis_tq - t_vis).mean()
            l_kd_r = charbonnier(pred - t_ren).mean()
            total = total + self.w_kd_flow * l_kd_f + self.w_kd_render * l_kd_r
            metrics["kd_flow"] = float(l_kd_f.detach())
            metrics["kd_ren"] = float(l_kd_r.detach())

        metrics["total"] = float(total.detach())
        return total, metrics


GameFILoss = FrameFlowLoss
