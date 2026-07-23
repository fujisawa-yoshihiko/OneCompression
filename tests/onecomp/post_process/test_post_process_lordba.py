"""
Unit tests for PostProcessLoRDBA (binary DBF LoRA, QAT).

CPU-only, no model download: exercises the LoRDBA building blocks directly.
Covers the SmoothSign STE, BPW-aware mid_dim sizing, the no-op cold-start
initialization, the trainable->bit-packed materialization path (including the
DoubleBinaryLinear branch of _collect_lora_state), and the re-injection guard.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    pytest tests/onecomp/post_process/test_post_process_lordba.py -v
"""

import logging

import torch
import torch.nn as nn

import pytest

from onecomp.post_process._ste import smooth_sign_ste
from onecomp.post_process.post_process_lordba import (
    LoRDBALinear,
    PostProcessLoRDBA,
    TrainableDBFAdapter,
    _WarmupLoRALinear,
    estimate_dbf_bpw,
    solve_mid_dim_for_bpw,
)
from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear


# ---------------------------------------------------------------------------
# SmoothSign STE
# ---------------------------------------------------------------------------


def test_smooth_sign_ste_forward_is_pm_one():
    """Forward value is a hard ±1 sign, with zeros mapped to +1."""
    x = torch.tensor([-2.0, -0.01, 0.0, 0.01, 5.0])
    y = smooth_sign_ste(x, k=100.0)
    assert torch.equal(y, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))


def test_smooth_sign_ste_gradient_near_zero_is_positive():
    """Near zero the tanh surrogate gives a positive (non-vanishing) gradient."""
    x = torch.linspace(-0.005, 0.005, 5, requires_grad=True)
    smooth_sign_ste(x, k=100.0).sum().backward()
    assert torch.all(x.grad > 0)


def test_smooth_sign_ste_k_zero_is_identity_grad():
    """k<=0 falls back to clipped-identity STE (gradient 1 everywhere)."""
    x = torch.tensor([-3.0, 0.2, 4.0], requires_grad=True)
    smooth_sign_ste(x, k=0.0).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad))


# ---------------------------------------------------------------------------
# BPW-aware mid_dim sizing
# ---------------------------------------------------------------------------


def test_solve_mid_dim_respects_budget():
    """For a feasible target, the solved mid_dim stays within target_bpw."""
    mid = solve_mid_dim_for_bpw(4096, 4096, target_bpw=1.5)
    assert mid % 8 == 0
    assert estimate_dbf_bpw(4096, 4096, mid) <= 1.5


def test_solve_mid_dim_warns_when_below_minimum(caplog):
    """An infeasibly small target_bpw clamps to mid_dim=8 and logs a warning."""
    with caplog.at_level(logging.WARNING):
        mid = solve_mid_dim_for_bpw(256, 256, target_bpw=0.05)
    assert mid == 8
    assert estimate_dbf_bpw(256, 256, 8) > 0.05  # genuinely over budget
    assert any("below the minimum achievable" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Cold-start initialization
# ---------------------------------------------------------------------------


def test_adapter_is_noop_at_init():
    """scale_out=0 makes the adapter output exactly zero at initialization."""
    adapter = TrainableDBFAdapter(in_features=32, out_features=16, mid_dim=8)
    x = torch.randn(4, 32)
    assert torch.count_nonzero(adapter(x)) == 0


def test_cold_start_scales_are_unit():
    """scale_in / scale_mid start at 1.0 so they are not frozen by a tiny init."""
    adapter = TrainableDBFAdapter(in_features=32, out_features=16, mid_dim=8)
    assert torch.allclose(adapter.scale_in, torch.ones_like(adapter.scale_in))
    assert torch.allclose(adapter.scale_mid, torch.ones_like(adapter.scale_mid))
    assert torch.count_nonzero(adapter.scale_out) == 0


# ---------------------------------------------------------------------------
# QAT-Full SVD initialization (Eq. 7)
# ---------------------------------------------------------------------------


def test_init_from_delta_rank1_reconstructs_exactly():
    """A rank-1 target is reconstructed exactly: the binary signs carry the
    sign pattern and the one-sweep least-squares scales recover |u|, |v|."""
    torch.manual_seed(0)
    in_f, out_f, mid = 16, 12, 8
    u = torch.randn(in_f)
    v = torch.randn(out_f)
    delta = 3.0 * torch.outer(u, v)  # (in, out), rank 1

    adapter = TrainableDBFAdapter(in_f, out_f, mid)
    adapter.init_from_delta(delta)

    x = torch.randn(5, in_f)
    assert torch.allclose(adapter(x), x @ delta, atol=1e-3, rtol=1e-3)


def test_init_from_delta_sets_scale_mid_to_singular_values():
    """scale_mid (beta) is initialized to the top singular values of the delta."""
    torch.manual_seed(1)
    in_f, out_f, mid = 24, 20, 8
    delta = torch.randn(in_f, out_f)
    svals = torch.linalg.svdvals(delta)[:mid]

    adapter = TrainableDBFAdapter(in_f, out_f, mid)
    adapter.init_from_delta(delta)
    assert torch.allclose(adapter.scale_mid.detach(), svals, atol=1e-3, rtol=1e-3)


def test_init_from_delta_improves_over_random():
    """SVD init reconstructs a low-rank delta far better than random init."""
    torch.manual_seed(2)
    in_f, out_f, mid = 32, 28, 8
    a = torch.randn(in_f, mid)
    b = torch.randn(mid, out_f)
    delta = a @ b  # rank-8 target, matches mid_dim
    x = torch.randn(64, in_f)
    target = x @ delta

    random_adapter = TrainableDBFAdapter(in_f, out_f, mid)
    with torch.no_grad():
        random_adapter.scale_out.fill_(0.1)  # random init is a no-op otherwise
    err_random = (random_adapter(x) - target).norm()

    svd_adapter = TrainableDBFAdapter(in_f, out_f, mid)
    svd_adapter.init_from_delta(delta)
    err_svd = (svd_adapter(x) - target).norm()

    # Sign-binarizing mid_dim components with only per-axis scales cannot match
    # a rank-mid Gaussian delta exactly (entries straddle zero; cf. Thm 4.1),
    # but the task-informed init is still clearly better than random signs.
    assert err_svd < 0.8 * err_random


def test_init_from_delta_noop_on_zero_delta():
    """A near-zero delta leaves the random initialization untouched."""
    adapter = TrainableDBFAdapter(16, 12, 8)
    before = adapter.b1_latent.detach().clone()
    adapter.init_from_delta(torch.zeros(16, 12))
    assert torch.equal(adapter.b1_latent.detach(), before)


def test_init_from_delta_after_freeze_raises():
    """init_from_delta must run before the signs are hardened."""
    adapter = TrainableDBFAdapter(16, 12, 8)
    adapter.freeze_signs()
    with pytest.raises(RuntimeError):
        adapter.init_from_delta(torch.randn(16, 12))


def test_warmup_lora_delta_zero_at_init():
    """The fp16 warm-up adapter starts as a no-op (B=0 LoRA convention)."""
    base = nn.Linear(32, 16, bias=False)
    warm = _WarmupLoRALinear(base, in_features=32, out_features=16, rank=8, scaling=2.0)
    dw = warm.delta_weight()
    assert dw.shape == (16, 32)  # (out, in)
    assert torch.count_nonzero(dw) == 0


# ---------------------------------------------------------------------------
# Materialization + state collection (DoubleBinaryLinear branch)
# ---------------------------------------------------------------------------


def test_freeze_to_inference_swaps_in_double_binary_linear():
    """After freezing, the adapter is a bit-packed DoubleBinaryLinear and the
    layer output is preserved (signs identical; scales rounded to fp16)."""
    torch.manual_seed(0)
    base = nn.Linear(32, 16, bias=False)
    layer = LoRDBALinear(base, in_features=32, out_features=16, mid_dim=8, scaling=2.0)
    # Move scale_out off zero so the adapter actually contributes.
    with torch.no_grad():
        layer.adapter.scale_out.fill_(0.1)

    x = torch.randn(4, 32)
    before = layer(x)
    layer.freeze_to_inference()
    after = layer(x)

    assert layer.is_frozen
    assert isinstance(layer.adapter, DoubleBinaryLinear)
    # fp16 rounding of the scale vectors only -> small relative difference.
    assert torch.allclose(before, after, atol=1e-2, rtol=1e-2)


def test_collect_lora_state_double_binary_branch():
    """_collect_lora_state handles a materialized (DoubleBinaryLinear) adapter,
    exercising the post-materialization branch end-to-end."""
    base = nn.Linear(32, 16, bias=False)
    layer = LoRDBALinear(base, in_features=32, out_features=16, mid_dim=8, scaling=2.0)
    layer.freeze_to_inference()
    assert isinstance(layer.adapter, DoubleBinaryLinear)

    model = nn.Module()
    model.add_module("q_proj", layer)

    pp = PostProcessLoRDBA()
    state = pp._collect_lora_state(model)  # pylint: disable=protected-access

    assert "q_proj" in state
    entry = state["q_proj"]
    assert entry["in_features"] == 32
    assert entry["out_features"] == 16
    assert entry["mid_dim"] == 8
    # bp1 packs the (mid x in) sign matrix = 8*32 = 256 signs -> 32 bytes.
    assert entry["bp1"].dtype == torch.uint8
    assert entry["bp3"].dtype == torch.uint8
    assert entry["scale_in"].dtype == torch.float16


# ---------------------------------------------------------------------------
# Re-injection guard
# ---------------------------------------------------------------------------


def test_find_targets_skips_trainable_adapter_children():
    """_find_target_layer_names must never return a TrainableDBFAdapter, so a
    second injection pass cannot double-wrap an already-wrapped layer."""
    base = nn.Linear(32, 16, bias=False)
    layer = LoRDBALinear(base, in_features=32, out_features=16, mid_dim=8, scaling=2.0)
    model = nn.Module()
    model.add_module("q_proj", layer)

    pp = PostProcessLoRDBA()
    names = pp._find_target_layer_names(model)  # pylint: disable=protected-access

    for name in names:
        resolved = model.get_submodule(name)
        assert not isinstance(resolved, (TrainableDBFAdapter, LoRDBALinear))
