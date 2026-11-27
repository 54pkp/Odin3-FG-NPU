"""Border-safe 3x3 symmetric bilateral cost volume (`CLAMP_TO_EDGE` replication, zero border chroma bleed) for OmniShift."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def replicate_pad_2d(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="replicate")


class BorderSafeCostVolume25(nn.Module):
    """3x3 symmetric bilateral cost volume with explicit edge-replicated padding (`padding=0` on Conv2d)."""

    def __init__(self, c_in: int, c_match: int = 12, direct_color_ch: int = 0):
        super().__init__()
        self.proj_a = nn.Conv2d(c_in, c_match, 1, bias=True)
        self.proj_b = nn.Conv2d(c_in, c_match, 1, bias=True)
        offs = [(dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)]

        self.rep_a = nn.Conv2d(c_match, c_match * 25, 1, bias=True)
        self.rep_b = nn.Conv2d(c_match, c_match * 25, 1, bias=True)
        self.shift_a = nn.Conv2d(c_match, c_match * 25, 3, padding=0, bias=True)
        self.shift_b = nn.Conv2d(c_match, c_match * 25, 3, padding=0, bias=True)
        self.pool_m = nn.Conv2d(c_match * 25, 25, 1, bias=True)
        self.pool_n = nn.Conv2d(c_match * 25, 25, 1, bias=True)
        self.score_head = nn.Conv2d(50, 25, 1, bias=True)

        def tap_1d(delta_cell: float) -> tuple[float, float, float]:
            if delta_cell <= -0.99:
                return (1.0, 0.0, 0.0)
            if delta_cell <= -0.49:
                return (0.5, 0.5, 0.0)
            if delta_cell < 0.49:
                return (0.0, 1.0, 0.0)
            if delta_cell < 0.99:
                return (0.0, 0.5, 0.5)
            return (0.0, 0.0, 1.0)

        with torch.no_grad():
            nn.init.orthogonal_(self.proj_a.weight, gain=2.2)
            self.proj_a.bias.zero_()
            if direct_color_ch > 0:
                base_c = c_in - direct_color_ch
                self.proj_a.weight[:6, :].zero_()
                for k_i, src_k in enumerate((0, 16, 17, 1, 2, 18)):
                    if src_k < direct_color_ch:
                        self.proj_a.weight[k_i, base_c + src_k, 0, 0] = 6.0
                        self.proj_a.bias[k_i] = -3.0
            self.proj_b.weight.copy_(self.proj_a.weight)
            self.proj_b.bias.copy_(self.proj_a.bias)

            for m in (self.rep_a, self.rep_b, self.shift_a, self.shift_b, self.pool_m, self.pool_n):
                m.weight.zero_()
                m.bias.fill_(1e-6)
            ch_weights = np.ones((c_match,), dtype=np.float32)
            if direct_color_ch > 0:
                ch_weights[:6] = 2.5
            ch_weights /= ch_weights.sum()
            for idx, (dy, dx) in enumerate(offs):
                wya = tap_1d(-0.5 * dy)
                wxa = tap_1d(-0.5 * dx)
                wyb = tap_1d(0.5 * dy)
                wxb = tap_1d(0.5 * dx)
                for c in range(c_match):
                    ch_o = idx * c_match + c
                    self.rep_a.weight[ch_o, c, 0, 0] = 1.0
                    self.rep_b.weight[ch_o, c, 0, 0] = 1.0
                    for ky in range(3):
                        for kx in range(3):
                            self.shift_a.weight[ch_o, c, ky, kx] = wya[ky] * wxa[kx]
                            self.shift_b.weight[ch_o, c, ky, kx] = wyb[ky] * wxb[kx]
                    self.pool_m.weight[idx, ch_o, 0, 0] = float(ch_weights[c])
                    self.pool_n.weight[idx, ch_o, 0, 0] = float(ch_weights[c])

            self.score_head.weight.zero_()
            self.score_head.bias.zero_()
            for idx in range(25):
                self.score_head.weight[idx, idx, 0, 0] = -4.0
                self.score_head.weight[idx, 25 + idx, 0, 0] = 2.0
                self.score_head.bias[idx] = 1.0

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        qa = torch.tanh(self.proj_a(fa))
        qb = torch.tanh(self.proj_b(fb))
        sa = self.shift_a(replicate_pad_2d(qa, 1, 1))
        sb = self.shift_b(replicate_pad_2d(qb, 1, 1))
        diff = torch.abs(sa - sb)
        norm = torch.abs(self.rep_a(qa)) + torch.abs(self.rep_b(qb))
        m25 = self.pool_m(diff)
        n25 = self.pool_n(norm)
        return F.relu(self.score_head(torch.cat([m25, n25], dim=1)))
