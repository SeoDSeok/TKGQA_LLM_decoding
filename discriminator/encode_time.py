"""Bochner (TGAT-style) functional time encoding Φ(t) — the value-space signal (R1).

The core Phase-0.5 finding is that the LLM encodes dates as base-10 digit
fragments, not as a comparable scalar, so it cannot reliably do first/last (raw
model sits at chance, 46.2 %). Φ maps a *real-valued* time to a continuous vector
whose inner product is a smooth function of the time difference — i.e. it puts
time back in value space so a small set-attention net can compare candidates.

Two channels, concatenated:
  * relative  — time rescaled to [0,1] within the candidate set
                (t̃ = (t−min)/(max−min)); first/last are pure relative operators,
                so this channel is what makes zero-shot transfer across absolute
                epochs possible.
  * absolute  — fractional year on a fixed decade scale; carries the anchor
                comparison needed for before/after.

Note on leakage: relative rescaling is a *value* representation (true rescaled
times), NOT an is_min/is_max flag. The model must still learn, from the question,
which end (min vs max) the operator wants — that mapping is exactly what Split-A
tests. Explicit rank / is_min / is_max flags are deliberately excluded from the
base config (see dataset.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def date_to_fracyear(tp) -> float:
    """TimePoint -> fractional year (monotone real scalar)."""
    return tp.year + (tp.month - 1) / 12.0 + (tp.day - 1) / 372.0


class BochnerTime(nn.Module):
    """Φ(t) = [cos(ω_1 t + b_1), …, cos(ω_d t + b_d)], ω and b learnable."""

    def __init__(self, dim: int = 64, init_scale: float = 10.0):
        super().__init__()
        # log-spaced initial frequencies span coarse (decades) to fine (days)
        omega = torch.logspace(-2.0, 1.0, dim) / init_scale
        self.omega = nn.Parameter(omega)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,) real values -> (..., dim)
        return torch.cos(t.unsqueeze(-1) * self.omega + self.bias)


class TimeEncoder(nn.Module):
    """Relative + absolute Bochner channels -> (..., 2*dim)."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.rel = BochnerTime(dim, init_scale=1.0)     # input already in [0,1]
        self.abs = BochnerTime(dim, init_scale=10.0)    # input in decades
        self.out_dim = 2 * dim

    def forward(self, t_rel: torch.Tensor, t_abs: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.rel(t_rel), self.abs(t_abs)], dim=-1)
