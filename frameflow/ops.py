# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""Exact differentiable PyTorch operators matching the WebGPU & WebGL2 graph spec.

All image tensors use BGR channel order in [0, 1] (`[B, 3, H, W]`), matching
`tex_in.wgsl` and `luma.wgsl`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bgr_to_luma(x: torch.Tensor) -> torch.Tensor:
    """BGR [B, 3, H, W] -> Luma [B, 1, H, W]. Matches luma.wgsl exactly."""
    return 0.114 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.299 * x[:, 2:3]


def luma_grad(x: torch.Tensor) -> torch.Tensor:
    """Forward-difference edge magnitude of luma [B, 1, H, W]. Matches luma.wgsl `grad`."""
    lum = bgr_to_luma(x)
    gx = torch.zeros_like(lum)
    gy = torch.zeros_like(lum)
    gx[:, :, :, :-1] = lum[:, :, :, 1:] - lum[:, :, :, :-1]
    gy[:, :, :-1, :] = lum[:, :, 1:, :] - lum[:, :, :-1, :]
    return gx.abs() + gy.abs()


def sad_volume(la: torch.Tensor, lb: torch.Tensor, radius: int) -> torch.Tensor:
    """Luma SAD cost volume: (2r+1)^2 channels, 3x3 zero-padded box avg of |la - shift(lb)|.

    Matches `sad.wgsl` exactly:
      c = (dy + r) * (2r + 1) + (dx + r), dy in [-r, r], dx in [-r, r],
      lb edge-clamped, then 3x3 sum / 9.0 with zero padding outside image bounds.
    """
    b, _, h, w = la.shape
    lb_pad = F.pad(lb, (radius, radius, radius, radius), mode="replicate")
    diffs = []
    for dy in range(-radius, radius + 1):
        y0 = radius + dy
        for dx in range(-radius, radius + 1):
            x0 = radius + dx
            shifted = lb_pad[:, :, y0 : y0 + h, x0 : x0 + w]
            diffs.append((la - shifted).abs())
    vol = torch.cat(diffs, dim=1)  # [B, (2r+1)^2, H, W]
    return F.avg_pool2d(vol, kernel_size=3, stride=1, padding=1, count_include_pad=True)


def backward_warp(
    img: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Backward warp `out[y, x] = img[y + fy, x + fx]` with border clamping.

    Matches `warp.wgsl` (bilinear) and `render.wgsl` (bilinear/bicubic Keys a=-0.75)
    which correspond to `F.grid_sample(..., align_corners=True, padding_mode='border')`.
    """
    b, _, h, w = img.shape
    device, dtype = img.device, img.dtype
    yy = torch.arange(h, device=device, dtype=torch.float32)
    xx = torch.arange(w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    gx = grid_x.unsqueeze(0) + flow[:, 0].float()
    gy = grid_y.unsqueeze(0) + flow[:, 1].float()
    # Map pixel coordinate [0, W-1] -> [-1, 1] for align_corners=True
    nx = 2.0 * gx / max(w - 1, 1) - 1.0
    ny = 2.0 * gy / max(h - 1, 1) - 1.0
    grid = torch.stack((nx, ny), dim=-1)
    out = F.grid_sample(
        img.float(),
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    return out.to(dtype)


def convex_upsample_2x(wt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """2x convex upsampling matching `convex.wgsl`.

    wt: [B, 36, H, W] indexed as channel `k9 * 4 + sy * 2 + sx` (k9 in 0..8, sy,sx in 0..1)
    x:  [B, C, H, W]
    returns: [B, C, 2*H, 2*W]
    """
    b, c, h, w = x.shape
    # Reshape weights to [B, 9, 2, 2, H, W] and softmax over the 9 neighborhood taps
    weights = wt.float().view(b, 9, 2, 2, h, w)
    weights = torch.softmax(weights, dim=1)  # [B, 9, 2, 2, H, W]

    # Zero-padded 3x3 neighborhood of x: [B, C*9, H*W] -> [B, C, 9, 1, 1, H, W]
    x_unf = F.unfold(x.float(), kernel_size=3, padding=1).view(b, c, 9, 1, 1, h, w)
    # Weighted sum over k9 -> [B, C, 2, 2, H, W]
    up = (weights.unsqueeze(1) * x_unf).sum(dim=2)
    # Interleave (H, sy=2) and (W, sx=2) -> [B, C, 2H, 2W]
    up = up.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, c, h * 2, w * 2)
    return up.to(x.dtype)


def blend_weights(vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """3-logit visibility softmax -> (wA, wB). Matches `blend.wgsl` exactly."""
    p = torch.softmax(vis.float(), dim=1)
    w_a = (p[:, 0:1] + 0.5 * p[:, 2:3]).to(vis.dtype)
    w_b = (p[:, 1:2] + 0.5 * p[:, 2:3]).to(vis.dtype)
    return w_a, w_b


def affine_out(x: torch.Tensor) -> torch.Tensor:
    """2 * sigmoid(x) - 1 in (-1, 1). Matches `unary.wgsl` op=2."""
    return 2.0 * torch.sigmoid(x) - 1.0


def render_pix(
    a: torch.Tensor,
    b: torch.Tensor,
    flow_tq: torch.Tensor,
    vis_tq: torch.Tensor,
    r_t: torch.Tensor,
    sampling: str = "bicubic",
    still_test: bool = False,
) -> torch.Tensor:
    """Full-resolution differentiable render stage matching `render.wgsl`.

    `still_test=False` matches reference training / `NetOpts::new` (`decisive=0`).
    `still_test=True` matches browser `FhSession` default (`decisive=2`, HUD still guard).
    """
    _, _, h, w = a.shape
    flow_full = F.interpolate(flow_tq, size=(h, w), mode="bilinear", align_corners=False)
    vis_full = F.interpolate(vis_tq, size=(h, w), mode="bilinear", align_corners=False)
    r_full = F.interpolate(r_t, size=(h, w), mode="bilinear", align_corners=False)

    wa_w, wb_w = blend_weights(vis_full)
    wa = backward_warp(a, flow_full[:, 0:2], mode=sampling)
    wb = backward_warp(b, flow_full[:, 2:4], mode=sampling)
    out = wa * wa_w + wb * wb_w + r_full
    if still_test:
        ab_diff = (a.float() - b.float()).abs().mean(dim=1, keepdim=True)
        es = F.avg_pool2d(F.pad(ab_diff, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        ef = (wa.float() - wb.float()).abs().mean(dim=1, keepdim=True)
        stillw = ((ef - es) / 0.04).clamp(0.0, 1.0) * (es < 0.03).float()
        own = 0.5 * (a.float() + b.float())
        out = (out.float() * (1.0 - stillw) + own * stillw).to(out.dtype)
    return out.clamp(0.0, 1.0)
