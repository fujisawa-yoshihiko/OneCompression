"""Improved Coordinate Descent verification for binary weight updates (v2).

Key improvements over v1:
  1. Multi-batch evaluation: average loss over multiple calibration batches
     to reduce noise in accept/reject decisions.
  2. Adaptive K: scale candidates per matrix proportional to matrix size.
  3. Binary-search group flipping: instead of all-or-nothing + individual
     fallback, use binary search to find the largest accepted subset.
  4. Score-weighted candidate ranking with temperature annealing.
  5. Cumulative tracking of accepted flips across steps.

Copyright 2025-2026 Fujitsu Ltd.
"""

from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = getLogger(__name__)


def _get_mdbf_candidates_v2(
    mdbf_modules: List[Tuple[str, nn.Module]],
    cd_k: int,
    k_ratio: float = 0.005,
    score_temperature: float = 1.0,
) -> List[Tuple[str, nn.Module, int, str, torch.Tensor, torch.Tensor]]:
    """Extract flip candidates from MDBF modules with adaptive K.

    Args:
        cd_k: Base K (minimum candidates per matrix).
        k_ratio: Fraction of total bits to consider as candidates.
        score_temperature: Higher = more exploration (flatter score distribution).

    Returns:
        List of (name, module, path_idx, which, flat_indices, scores).
        Sorted by score descending within each entry.
    """
    from onecomp.quantizer.mdbf.mdbf_layer import unpack_binary

    candidates = []
    for mod_name, mod in mdbf_modules:
        for p, path in enumerate(mod.paths):
            for which in ("A", "B"):
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

                n_positive = int((score > 0).sum().item())
                adaptive_k = max(cd_k, int(score.numel() * k_ratio))
                K_actual = min(adaptive_k, n_positive)
                if K_actual == 0:
                    continue

                flat_score = score.flatten()
                if score_temperature != 1.0:
                    flat_score = flat_score ** (1.0 / max(score_temperature, 0.1))

                top_vals, top_idx = flat_score.topk(K_actual)
                valid = top_vals > 0
                if valid.any():
                    candidates.append(
                        (mod_name, mod, p, which,
                         top_idx[valid], top_vals[valid])
                    )
    return candidates


def _flip_mdbf_bits(
    mod: nn.Module, path_idx: int, which: str, flat_indices: torch.Tensor,
) -> None:
    """Flip specified bits in an MDBF sign matrix."""
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


def _multi_batch_loss(
    model: nn.Module,
    batches: List[dict],
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
    teacher_device: Optional[torch.device] = None,
    temperature: float = 1.0,
) -> float:
    """Compute average loss over multiple calibration batches."""
    from .losses import compute_kl_loss
    from .helpers import get_logits

    t_dev = teacher_device or device
    total_loss = 0.0

    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        logits_s = get_logits(model(input_ids))

        if teacher_model is not None:
            t_input = input_ids.to(t_dev) if t_dev != device else input_ids
            logits_t = get_logits(teacher_model(t_input)).to(device)
            loss = compute_kl_loss(
                logits_t, logits_s, temperature,
                attention_mask=attention_mask,
            ).item()
        else:
            loss = torch.nn.functional.cross_entropy(
                logits_s[:, :-1].reshape(-1, logits_s.size(-1)),
                input_ids[:, 1:].reshape(-1),
            ).item()
        total_loss += loss

    return total_loss / max(len(batches), 1)


class _ForwardBudget:
    """Tracks remaining forward calls to avoid unbounded computation."""
    def __init__(self, max_calls: int):
        self.remaining = max_calls

    def spend(self, n: int = 1) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= n
        return True


def _bisect_accept(
    model: nn.Module,
    mod: nn.Module,
    path_idx: int,
    which: str,
    sorted_indices: torch.Tensor,
    loss_ref: float,
    loss_fn,
    min_group: int = 16,
    budget: _ForwardBudget = None,
) -> Tuple[int, float]:
    """Binary-search for the largest prefix of sorted_indices that reduces loss.

    Tries full set first. On rejection, splits in half and tries each half.
    Recursion stops at min_group size, where top-1 individual fallback is used.
    Forward budget prevents excessive computation.

    Returns (n_accepted, updated_loss_ref).
    """
    n = len(sorted_indices)
    if n == 0:
        return 0, loss_ref
    if budget is not None and not budget.spend():
        return 0, loss_ref

    _flip_mdbf_bits(mod, path_idx, which, sorted_indices)
    loss_new = loss_fn()

    if loss_new < loss_ref:
        return n, loss_new

    _flip_mdbf_bits(mod, path_idx, which, sorted_indices)

    if n <= min_group:
        accepted = 0
        top_n = min(n, 3)
        for i in range(top_n):
            if budget is not None and not budget.spend():
                break
            idx = sorted_indices[i : i + 1]
            _flip_mdbf_bits(mod, path_idx, which, idx)
            loss_new = loss_fn()
            if loss_new < loss_ref:
                loss_ref = loss_new
                accepted += 1
            else:
                _flip_mdbf_bits(mod, path_idx, which, idx)
        return accepted, loss_ref

    mid = n // 2
    acc1, loss_ref = _bisect_accept(
        model, mod, path_idx, which,
        sorted_indices[:mid], loss_ref, loss_fn, min_group, budget,
    )
    acc2, loss_ref = _bisect_accept(
        model, mod, path_idx, which,
        sorted_indices[mid:], loss_ref, loss_fn, min_group, budget,
    )
    return acc1 + acc2, loss_ref


@torch.no_grad()
def cd_verify_step_v2(
    model: nn.Module,
    eval_batches: List[dict],
    mdbf_modules: List[Tuple[str, nn.Module]],
    device: torch.device,
    cd_k: int = 200,
    k_ratio: float = 0.005,
    score_temperature: float = 1.0,
    min_group: int = 16,
    max_forward_calls: int = 60,
    temperature: float = 1.0,
    teacher_model: Optional[nn.Module] = None,
    teacher_device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    """Improved CD verification step (v2).

    Improvements over v1:
      - Multi-batch loss evaluation (reduces noise)
      - Adaptive K (scales with matrix size)
      - Binary-search group acceptance (finds largest accepted subset)
      - Forward budget prevents excessive computation
      - Score temperature for exploration control

    Args:
        eval_batches: List of calibration batches for loss evaluation.
        cd_k: Base K per matrix (adaptive K may increase this).
        k_ratio: Fraction of bits to consider as candidates.
        score_temperature: Temperature for score ranking (>1 = more exploration).
        min_group: Minimum group size for bisect fallback.
        max_forward_calls: Maximum number of forward passes per CD step.

    Returns:
        (n_accepted, n_tried).
    """
    model.eval()

    def loss_fn() -> float:
        return _multi_batch_loss(
            model, eval_batches, device,
            teacher_model, teacher_device, temperature,
        )

    loss_ref = loss_fn()
    n_accepted = 0
    n_tried = 0

    budget = _ForwardBudget(max_forward_calls)

    all_candidates = _get_mdbf_candidates_v2(
        mdbf_modules, cd_k, k_ratio, score_temperature,
    )

    for mod_name, mod, path_idx, which, indices, scores in all_candidates:
        if budget.remaining <= 0:
            break
        n_tried += len(indices)
        acc, loss_ref = _bisect_accept(
            model, mod, path_idx, which,
            indices, loss_ref, loss_fn, min_group, budget,
        )
        n_accepted += acc

    model.train()
    return n_accepted, n_tried
