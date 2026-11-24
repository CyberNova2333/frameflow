# SPDX-License-Identifier: AGPL-3.0-only
# Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
"""GameFINet (FrameFlowNet): 1:1 graph-compatible PyTorch implementation supporting GameFI-Std and GameFI-Lite.

Input frames A, B are BGR in [0, 1], shape [B, 3, H, W] with H, W divisible by 32.
Parameter names match the `graph.json` / manifest specification so exported weights load
directly into both the WebGPU engine and the WebGL2 Fragment Shader runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, get_preset
from .ops import (
    affine_out,
    backward_warp,
    bgr_to_luma,
    blend_weights,
    convex_upsample_2x,
    luma_grad,
    render_pix,
    sad_volume,
)


def conv_prelu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(num_parameters=cout, init=0.25),
    )


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.e1 = conv_prelu(3, cfg.c8, stride=2)
        self.e2 = conv_prelu(cfg.c8, cfg.c8, stride=1)
        self.e3 = conv_prelu(cfg.c8, cfg.c16, stride=2)
        self.e4 = conv_prelu(cfg.c16, cfg.c16, stride=1)

    def forward(self, xq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.e1(xq)
        f8 = self.e2(e1) + e1
        e3 = self.e3(f8)
        f16 = self.e4(e3) + e3
        return f8, f16


class CoarseNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.c1 = conv_prelu(cfg.coarse_in_ch, cfg.cc, stride=1)
        self.c2 = conv_prelu(cfg.cc, cfg.cc, stride=1)
        self.c3 = conv_prelu(cfg.cc, cfg.cc, stride=1)
        self.c4 = conv_prelu(cfg.cc, cfg.cc, stride=1)
        self.head = nn.Conv2d(cfg.cc, cfg.coarse_head_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(
        self,
        f16a: torch.Tensor,
        f16b: torch.Tensor,
        i16a: torch.Tensor,
        i16b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = f16a.shape
        device, dtype = f16a.device, f16a.dtype
        img16 = torch.cat([i16a, i16b], dim=1)
        side = torch.zeros((b, self.cfg.side_ch, h, w), device=device, dtype=dtype)
        state = torch.zeros((b, self.cfg.state_ch, h, w), device=device, dtype=dtype)
        la = bgr_to_luma(i16a)
        lb = bgr_to_luma(i16b)
        cv = sad_volume(la, lb, self.cfg.cost_radius)
        cin = torch.cat([f16a, f16b, img16, side, state, cv], dim=1)
        h1 = self.c1(cin)
        h2 = self.c2(h1)
        h3 = self.c3(h2) + h1
        hc = self.c4(h3)
        o = self.head(hc)
        flow16 = o[:, 0:4] * 16.0
        vis16 = o[:, 4:7]
        conf = o[:, 7:8]
        gate = o[:, 8:9]
        sal = o[:, 9:10]
        return flow16, vis16, conf, gate, sal


class FineNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.r1 = conv_prelu(cfg.fine_in_ch, cfg.cr, stride=1)
        self.r2 = conv_prelu(cfg.cr, cfg.cr, stride=1)
        self.r3 = conv_prelu(cfg.cr, cfg.cr, stride=1)
        self.out = nn.Conv2d(cfg.cr, 7, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(
        self,
        f8a: torch.Tensor,
        f8b: torch.Tensor,
        i8a: torch.Tensor,
        i8b: torch.Tensor,
        flow: torch.Tensor,
        vis: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flow8 = flow * 0.125
        wa = backward_warp(f8a, flow8[:, 0:2], mode="bilinear")
        wb = backward_warp(f8b, flow8[:, 2:4], mode="bilinear")
        ia = backward_warp(i8a, flow8[:, 0:2], mode="bilinear")
        ib = backward_warp(i8b, flow8[:, 2:4], mode="bilinear")
        fcv = sad_volume(bgr_to_luma(ia), bgr_to_luma(ib), self.cfg.fine_cost_radius)
        fx = torch.cat([wa, wb, ia, ib, flow8, h, fcv], dim=1)
        r1 = self.r1(fx)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        d = self.out(r3)
        flow = flow + d[:, 0:4]
        vis = vis + d[:, 4:7]
        return flow, vis, r3


class ConvexUp(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w = nn.Conv2d(cfg.cr, 36, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return convex_upsample_2x(self.w(h), x)


class FilmHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cr = cfg.cr
        self.film = nn.Sequential(
            nn.Linear(1, cfg.film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.film_hidden, cfg.cr * 2),
        )
        self.h1 = conv_prelu(cfg.cr, cfg.cr, stride=1)
        self.h2 = conv_prelu(cfg.cr, cfg.cr, stride=1)
        self.out = nn.Conv2d(cfg.cr, 5, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # t: [B] or [B, 1] in (0, 1)
        t_in = t.view(-1, 1).to(dtype=h.dtype, device=h.device)
        g = self.film(t_in)  # [B, 2 * cr]
        sc = g[:, : self.cr].unsqueeze(-1).unsqueeze(-1)
        bi = g[:, self.cr :].unsqueeze(-1).unsqueeze(-1)
        hm = h * (1.0 + sc) + bi
        fh1 = self.h1(hm)
        fh2 = self.h2(fh1)
        d = self.out(fh2)
        return d[:, 0:4], d[:, 4:5]


class RefineHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.q1 = conv_prelu(14, cfg.cq, stride=1)
        self.q2 = conv_prelu(cfg.cq, cfg.cq, stride=1)
        self.q3 = conv_prelu(cfg.cq, cfg.cq, stride=1)
        self.q4 = nn.Conv2d(cfg.cq, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(
        self,
        aq: torch.Tensor,
        bq: torch.Tensor,
        flow_tq: torch.Tensor,
        vis_tq: torch.Tensor,
    ) -> torch.Tensor:
        flow_tq4 = flow_tq * 0.25
        rwa = backward_warp(aq, flow_tq4[:, 0:2], mode="bilinear")
        rwb = backward_warp(bq, flow_tq4[:, 2:4], mode="bilinear")
        bw_a, bw_b = blend_weights(vis_tq)
        blend = rwa * bw_a + rwb * bw_b
        lg = luma_grad(blend)
        rx = torch.cat([rwa, rwb, vis_tq, flow_tq * 0.05, lg], dim=1)
        q1 = self.q1(rx)
        q2 = self.q2(q1)
        q3 = self.q3(q2)
        return affine_out(self.q4(q3))


@dataclass
class PairState:
    a: torch.Tensor
    b: torch.Tensor
    aq: torch.Tensor
    bq: torch.Tensor
    flow: torch.Tensor
    vis: torch.Tensor
    h: torch.Tensor
    coarse_flow: torch.Tensor
    coarse_vis: torch.Tensor
    conf: torch.Tensor
    gate: torch.Tensor
    sal: torch.Tensor
    state_out: torch.Tensor


class FrameFlowNet(nn.Module):
    """Full optical-flow neural frame interpolator (GameFI-Std / GameFI-Lite)."""

    def __init__(self, cfg: ModelConfig | str = "std") -> None:
        super().__init__()
        self.cfg = get_preset(cfg) if isinstance(cfg, str) else cfg
        self.enc = Encoder(self.cfg)
        self.coarse = CoarseNet(self.cfg)
        self.fine = FineNet(self.cfg)
        self.up = ConvexUp(self.cfg)
        self.film = FilmHead(self.cfg)
        self.refine = RefineHead(self.cfg)
        self._init_weights()

    def _init_weights(self) -> None:
        # Mild zero-near init on residual heads so early training starts near clean blend
        nn.init.zeros_(self.coarse.head.weight)
        nn.init.zeros_(self.coarse.head.bias)
        nn.init.zeros_(self.fine.out.weight)
        nn.init.zeros_(self.fine.out.bias)
        nn.init.zeros_(self.film.out.weight)
        nn.init.zeros_(self.film.out.bias)
        nn.init.zeros_(self.refine.q4.weight)
        nn.init.zeros_(self.refine.q4.bias)

    def forward_pair(self, a: torch.Tensor, b: torch.Tensor, k: int | None = None) -> PairState:
        steps = self.cfg.k_default if k is None else k
        aq = F.interpolate(a, scale_factor=0.25, mode="bilinear", align_corners=False, antialias=True)
        bq = F.interpolate(b, scale_factor=0.25, mode="bilinear", align_corners=False, antialias=True)

        f8a, f16a = self.enc(aq)
        f8b, f16b = self.enc(bq)

        i8a = F.avg_pool2d(aq, kernel_size=2, stride=2)
        i8b = F.avg_pool2d(bq, kernel_size=2, stride=2)
        i16a = F.avg_pool2d(i8a, kernel_size=2, stride=2)
        i16b = F.avg_pool2d(i8b, kernel_size=2, stride=2)

        flow16, vis16, conf, gate, sal = self.coarse(f16a, f16b, i16a, i16b)
        flow = F.interpolate(flow16, scale_factor=2, mode="bilinear", align_corners=False)
        vis = F.interpolate(vis16, scale_factor=2, mode="bilinear", align_corners=False)
        h = torch.zeros((a.shape[0], self.cfg.cr, flow.shape[2], flow.shape[3]), device=a.device, dtype=a.dtype)

        for _ in range(steps):
            flow, vis, h = self.fine(f8a, f8b, i8a, i8b, flow, vis, h)

        state_out = torch.cat([f16b, flow16], dim=1)
        return PairState(
            a=a,
            b=b,
            aq=aq,
            bq=bq,
            flow=flow,
            vis=vis,
            h=h,
            coarse_flow=flow16,
            coarse_vis=vis16,
            conf=conf,
            gate=gate,
            sal=sal,
            state_out=state_out,
        )

    def forward_mid(
        self,
        pair: PairState,
        t: torch.Tensor | float,
        use_film: bool = True,
        use_refine: bool = True,
        sampling: str | None = None,
        still_test: bool = False,
    ) -> dict[str, torch.Tensor]:
        bsz = pair.a.shape[0]
        if not isinstance(t, torch.Tensor):
            t_vec = torch.full((bsz,), float(t), device=pair.a.device, dtype=pair.a.dtype)
        else:
            t_vec = t.to(device=pair.a.device, dtype=pair.a.dtype).view(bsz)
        t_4d = t_vec.view(bsz, 1, 1, 1)

        linf = pair.flow[:, 0:2] * (2.0 * t_4d)
        linb = pair.flow[:, 2:4] * (2.0 * (1.0 - t_4d))
        lin = torch.cat([linf, linb], dim=1)

        if use_film:
            d_flow, d_mask = self.film(pair.h, t_vec)
            flow_t = lin + d_flow
            dvis = torch.cat([d_mask, -d_mask, torch.zeros_like(d_mask)], dim=1)
            vis_t = pair.vis + dvis
        else:
            flow_t = lin
            vis_t = pair.vis

        up7 = self.up(pair.h, torch.cat([flow_t, vis_t], dim=1))
        flow_tq = up7[:, 0:4]
        vis_tq = up7[:, 4:7]

        if use_refine:
            r_t = self.refine(pair.aq, pair.bq, flow_tq, vis_tq)
        else:
            r_t = torch.zeros_like(pair.aq)

        samp = self.cfg.render_sampling if sampling is None else sampling
        rendered = render_pix(pair.a, pair.b, flow_tq, vis_tq, r_t, sampling=samp, still_test=still_test)
        return {
            "render": rendered,
            "flow_tq": flow_tq,
            "vis_tq": vis_tq,
            "R_t": r_t,
            "flow_t": flow_t,
            "vis_t": vis_t,
            "flow": pair.flow,
            "vis": pair.vis,
            "coarse_flow": pair.coarse_flow,
            "coarse_vis": pair.coarse_vis,
            "conf": pair.conf,
            "gate": pair.gate,
            "sal": pair.sal,
            "state_out": pair.state_out,
        }

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor | float = 0.5,
        k: int | None = None,
        use_film: bool = True,
        use_refine: bool = True,
        still_test: bool = False,
    ) -> dict[str, torch.Tensor]:
        pair = self.forward_pair(a, b, k=k)
        return self.forward_mid(pair, t, use_film=use_film, use_refine=use_refine, still_test=still_test)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


GameFINet = FrameFlowNet
