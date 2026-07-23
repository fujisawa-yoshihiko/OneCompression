"""Shared straight-through estimators for binary (sign) quantization.

A single canonical SmoothSign STE used by both the LoRDBA QAT post-process
(:mod:`onecomp.post_process.post_process_lordba`) and the block-wise sign
optimizers (:mod:`onecomp.post_process._blockwise`).  Keeping one definition
avoids the silent divergence that arises when the same estimator is copied
into several modules.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import torch


def smooth_sign_ste(x: torch.Tensor, k: float = 100.0) -> torch.Tensor:
    """SmoothSign straight-through estimator.

    Forward:  ``sign(x)`` in ``{-1, +1}`` (zeros mapped to ``+1``).
    Backward: ``d/dx tanh(k*x) = k * (1 - tanh^2(k*x))``.

    The hard ``sign`` is produced in the forward pass while the tanh surrogate
    carries the gradient, which is more stable than clipped-identity STE,
    especially at ultra-low bit.  Set ``k <= 0`` to fall back to vanilla
    clipped-identity STE (gradient ``1`` everywhere through the clip).
    """
    y = x.sign()
    y = torch.where(y == 0, torch.ones_like(y), y)
    if k <= 0:
        return x + (y - x).detach()
    t = torch.tanh(k * x)
    return t + (y - t).detach()
