"""Coordinate Descent verification for binary weight updates.

Provides loss-verified bit flipping for DBF and MDBF binary weights.
Instead of committing all STE-suggested bit flips at once (which may
include "false flips" that increase discrete loss), this module:

  1. Ranks flip candidates by STE divergence score.
  2. Attempts bulk flips and accepts only if loss decreases.
  3. Falls back to individual flips for top candidates on bulk failure.

This is the key component of the Hybrid STE+CD method, which achieves
better generation quality than pure STE by preventing false flips from
degrading the model in the discrete binary space.

Copyright 2025-2026 Fujitsu Ltd.

"""

from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate extraction from STE shadow weights
# ---------------------------------------------------------------------------


def _get_dbf_candidates(
    dbf_modules: List[Tuple[str, nn.Module]],
    cd_k: int,
) -> List[Tuple[str, nn.Module, str, str, torch.Tensor]]:
    """Extract top-K flip candidates from DBF modules.

    For each binary matrix (bp1/bp3), candidates are positions where the
    STE shadow weight disagrees with the current binary value AND flipping
    would reduce the STE loss (score > 0).

    Returns:
        List of (name, module, bp_attr, opt_attr, flat_indices).
    """
    from onecomp.quantizer.dbf.dbf_layer import unpack_binary

    _BINARY_ATTRS = ("bp1", "bp3")
    _BINARY_SHAPES = ("_bp1_shape", "_bp3_shape")
    _OPT_BINARY_ATTRS = ("_opt_bp1", "_opt_bp3")

    candidates = []
    for mod_name, mod in dbf_modules:
        for bp_attr, shape_attr, opt_attr in zip(
            _BINARY_ATTRS, _BINARY_SHAPES, _OPT_BINARY_ATTRS,
        ):
            if not hasattr(mod, opt_attr):
                continue
            w = getattr(mod, opt_attr)
            shape = getattr(mod, shape_attr)
            bp_packed = getattr(mod, bp_attr)
            Wq = unpack_binary(bp_packed)[:shape[0] * shape[1]].reshape(shape).float()

            score = (Wq - w.detach()) * Wq
            score = score.clamp(min=0)
            if score.max() == 0:
                continue

            K_actual = min(cd_k, int((score > 0).sum().item()))
            if K_actual == 0:
                continue
            top_vals, top_idx = score.flatten().topk(K_actual)
            valid = top_vals > 0
            if valid.any():
                candidates.append(
                    (mod_name, mod, bp_attr, opt_attr, top_idx[valid])
                )
    return candidates


def _get_mdbf_candidates(
    mdbf_modules: List[Tuple[str, nn.Module]],
    cd_k: int,
) -> List[Tuple[str, nn.Module, int, str, torch.Tensor]]:
    """Extract top-K flip candidates from MDBF modules.

    For each path's sign matrices (A_sign, B_sign), candidates are
    positions where the STE shadow weight has drifted from the current
    packed sign value.

    Returns:
        List of (name, module, path_idx, which, flat_indices).
    """
    from onecomp.quantizer.mdbf.mdbf_layer import unpack_binary

    _BINARY_SIGN_NAMES = ("A", "B")

    candidates = []
    for mod_name, mod in mdbf_modules:
        for p, path in enumerate(mod.paths):
            for which in _BINARY_SIGN_NAMES:
                opt_attr = f"_opt_{which}_sign"
                if not hasattr(path, opt_attr):
                    continue
                w = getattr(path, opt_attr)
                shape = (path.n, path.r) if which == "A" else (path.r, path.m)

                packed_key = f"{which}_sign_packed"
                packed = path._buffers.get(packed_key)
                if packed is None:
                    packed = path._packed_cpu.get(which)
                if packed is None:
                    continue

                Wq = unpack_binary(packed, shape).float().to(w.device)

                score = (Wq - w.detach()) * Wq
                score = score.clamp(min=0)
                if score.max() == 0:
                    continue

                K_actual = min(cd_k, int((score > 0).sum().item()))
                if K_actual == 0:
                    continue
                top_vals, top_idx = score.flatten().topk(K_actual)
                valid = top_vals > 0
                if valid.any():
                    candidates.append(
                        (mod_name, mod, p, which, top_idx[valid])
                    )
    return candidates


# ---------------------------------------------------------------------------
# Flip / revert helpers
# ---------------------------------------------------------------------------


def _flip_dbf_bits(
    mod: nn.Module,
    bp_attr: str,
    opt_attr: str,
    flat_indices: torch.Tensor,
) -> None:
    """Flip specified bits in a DBF binary matrix (both packed and shadow)."""
    from onecomp.quantizer.dbf.dbf_layer import pack_binary, unpack_binary

    shape_attr = "_bp1_shape" if bp_attr == "bp1" else "_bp3_shape"
    shape = getattr(mod, shape_attr)
    bp_packed = getattr(mod, bp_attr)
    W_flat = unpack_binary(bp_packed)[:shape[0] * shape[1]].float()
    W_flat[flat_indices] *= -1
    repacked = pack_binary(W_flat.sign().to(torch.int8).reshape(shape))
    bp_packed.copy_(repacked.to(bp_packed.device))

    if hasattr(mod, opt_attr):
        w = getattr(mod, opt_attr)
        w_flat = w.data.flatten()
        w_flat[flat_indices] *= -1


def _flip_mdbf_bits(
    mod: nn.Module,
    path_idx: int,
    which: str,
    flat_indices: torch.Tensor,
) -> None:
    """Flip specified bits in an MDBF sign matrix (both packed and shadow)."""
    from onecomp.quantizer.mdbf.mdbf_layer import pack_binary, unpack_binary

    path = mod.paths[path_idx]
    shape = (path.n, path.r) if which == "A" else (path.r, path.m)
    packed_key = f"{which}_sign_packed"
    packed = path._buffers.get(packed_key)
    cpu_packed = path._packed_cpu.get(which) if packed is None else None
    src = packed if packed is not None else cpu_packed

    if src is None:
        return

    Wq = unpack_binary(src, shape)
    W_flat = Wq.flatten().float()
    W_flat[flat_indices.to(W_flat.device)] *= -1
    new_packed, _ = pack_binary(W_flat.sign().to(torch.int8).reshape(shape))

    if packed is not None:
        packed.copy_(new_packed.to(packed.device))
    else:
        cpu_packed.copy_(new_packed.cpu())

    opt_attr = f"_opt_{which}_sign"
    if hasattr(path, opt_attr):
        w = getattr(path, opt_attr)
        w_data = w.data.flatten()
        w_data[flat_indices.to(w_data.device)] *= -1


# ---------------------------------------------------------------------------
# Main CD verify step
# ---------------------------------------------------------------------------


@torch.no_grad()
def cd_verify_step(
    model: nn.Module,
    eval_batch: dict,
    method: str,
    dbf_modules: List[Tuple[str, nn.Module]],
    mdbf_modules: List[Tuple[str, nn.Module]],
    device: torch.device,
    cd_k: int = 50,
    cd_fallback: int = 10,
    temperature: float = 1.0,
    teacher_model: Optional[nn.Module] = None,
    teacher_device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    """Run one CD verification step.

    Evaluates STE-suggested flip candidates against actual loss, accepting
    only those that improve (or at least don't degrade) the model.

    Args:
        model: Student model (quantized).
        eval_batch: Single calibration batch {"input_ids": ..., "attention_mask": ...}.
        method: "dbf" or "mdbf".
        dbf_modules: DBF module list.
        mdbf_modules: MDBF module list.
        device: GPU device.
        cd_k: Max candidates per binary matrix.
        cd_fallback: Individual retries after bulk rejection.
        temperature: KL temperature.
        teacher_model: FP16 teacher (for KL-based CD loss).
        teacher_device: Device for teacher.

    Returns:
        (n_accepted, n_tried): Count of accepted and total tried flips.
    """
    from .losses import compute_kl_loss
    from .helpers import get_logits

    model.eval()

    input_ids = eval_batch["input_ids"].to(device)
    attention_mask = eval_batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    t_dev = teacher_device or device

    def loss_of() -> float:
        logits_s = get_logits(model(input_ids))
        if teacher_model is not None:
            t_input = input_ids.to(t_dev) if t_dev != device else input_ids
            logits_t = get_logits(teacher_model(t_input)).to(device)
            return compute_kl_loss(
                logits_t, logits_s, temperature,
                attention_mask=attention_mask,
            ).item()
        else:
            return torch.nn.functional.cross_entropy(
                logits_s[:, :-1].reshape(-1, logits_s.size(-1)),
                input_ids[:, 1:].reshape(-1),
            ).item()

    loss_ref = loss_of()
    n_accepted = 0
    n_tried = 0

    if method == "dbf":
        all_candidates = _get_dbf_candidates(dbf_modules, cd_k)
        for mod_name, mod, bp_attr, opt_attr, indices in all_candidates:
            _flip_dbf_bits(mod, bp_attr, opt_attr, indices)
            loss_new = loss_of()
            n_tried += len(indices)

            if loss_new < loss_ref:
                loss_ref = loss_new
                n_accepted += len(indices)
            else:
                _flip_dbf_bits(mod, bp_attr, opt_attr, indices)
                for idx in indices[:cd_fallback]:
                    _flip_dbf_bits(mod, bp_attr, opt_attr, idx.unsqueeze(0))
                    loss_new = loss_of()
                    n_tried += 1
                    if loss_new < loss_ref:
                        loss_ref = loss_new
                        n_accepted += 1
                    else:
                        _flip_dbf_bits(mod, bp_attr, opt_attr, idx.unsqueeze(0))

    elif method == "mdbf":
        all_candidates = _get_mdbf_candidates(mdbf_modules, cd_k)
        for mod_name, mod, path_idx, which, indices in all_candidates:
            _flip_mdbf_bits(mod, path_idx, which, indices)
            loss_new = loss_of()
            n_tried += len(indices)

            if loss_new < loss_ref:
                loss_ref = loss_new
                n_accepted += len(indices)
            else:
                _flip_mdbf_bits(mod, path_idx, which, indices)
                for idx in indices[:cd_fallback]:
                    _flip_mdbf_bits(mod, path_idx, which, idx.unsqueeze(0))
                    loss_new = loss_of()
                    n_tried += 1
                    if loss_new < loss_ref:
                        loss_ref = loss_new
                        n_accepted += 1
                    else:
                        _flip_mdbf_bits(mod, path_idx, which, idx.unsqueeze(0))

    model.train()
    return n_accepted, n_tried
