"""LoRDBA QAT post-process: binary (DBF) LoRA adapters trained via SmoothSign STE.

This module introduces LoRDBA's QAT pathway into OneCompression as a
post-quantization process.  Where :class:`PostProcessLoraSFT` injects and
trains *fp16* LoRA adapters, ``PostProcessLoRDBA`` trains *binary* Double
Binary Factorization (DBF) adapters directly: the ±1 sign matrices are
learned through a straight-through estimator (SmoothSign), while the three
scaling vectors stay in full precision.  After training the signs are frozen
to hard ±1 and each adapter is materialized into the existing
:class:`~onecomp.quantizer.dbf.DoubleBinaryLinear` inference layer, so the
adapter is stored bit-packed (~1-2 bits / weight) and reuses OneCompression's
DBF inference (and GemLite) path.

The whole SFT / teacher-distillation / dataset / training-loop machinery is
inherited from :class:`PostProcessLoraSFT`; only the adapter injection, the
adapter state collection, and the saved-artifact format are overridden.

The QAT binary-DBF structure (5-stage ``Mul -> BitLinear -> Mul -> BitLinear
-> Mul``) and the SmoothSign STE are ported from the LoRDBA reproducibility
package (``double_binary.lora``).  The bit-packing and inference layer are
reused from :mod:`onecomp.quantizer.dbf`.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from logging import getLogger
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..quantizer.dbf.dbf_layer import DoubleBinaryLinear, pack_binary
from ._ste import smooth_sign_ste
from .post_process_lora_sft import (
    PostProcessLoraSFT,
    _DEFAULT_TARGET_MODULES,
    _is_lm_head_module,
    _replace_submodule,
    _resolve_submodule,
)

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# BPW-aware intermediate-dimension sizing (ported from double_binary.factorize)
# ---------------------------------------------------------------------------


def _align8(n: int) -> int:
    return (n + 7) & ~7


def estimate_dbf_bpw(out_features: int, in_features: int, mid_dim: int) -> float:
    """Estimate bits-per-weight of a DBF adapter relative to the full ΔW.

    Stores two bit-packed ±1 matrices (``mid*out + mid*in`` sign bits) and
    three fp16 scaling vectors (``(out + mid + in) * 16`` bits) against an
    ``out * in`` weight count.
    """
    sign_bits = out_features * mid_dim + mid_dim * in_features
    scale_bits = (out_features + mid_dim + in_features) * 16
    return (sign_bits + scale_bits) / (out_features * in_features)


def solve_mid_dim_for_bpw(
    out_features: int,
    in_features: int,
    target_bpw: float,
    lora_rank: int = 0,
) -> int:
    """Largest ``mid_dim`` (multiple of 8) achieving ``<= target_bpw``.

    With ``lora_rank == 0`` the denominator is ``out * in`` (the full ΔW).
    With ``lora_rank > 0`` the denominator is ``lora_rank * (out + in)``
    (the LoRA A+B parameter count), making ``target_bpw`` comparable to
    LoRA-basis methods such as LoRAQuant.
    """
    n, m = out_features, in_features
    denom_params = lora_rank * (n + m) if lora_rank > 0 else n * m
    numerator = target_bpw * denom_params - (n + m) * 16
    denominator = n + m + 16
    mid_dim = int(numerator / denominator)
    # Floor-align to a multiple of 8 (for bit-packing) so the resulting
    # adapter stays within the target budget rather than overshooting it.
    mid_dim = (max(mid_dim, 8) // 8) * 8
    mid_dim = max(mid_dim, 8)

    # The smallest representable adapter (mid_dim=8) can still exceed a very
    # small target_bpw; the floor-clamp above silently overshoots in that case,
    # so surface it rather than returning a budget-violating size quietly.
    sign_bits = n * mid_dim + mid_dim * m
    scale_bits = (n + mid_dim + m) * 16
    achieved_bpw = (sign_bits + scale_bits) / denom_params
    if achieved_bpw > target_bpw:
        logger.warning(
            "target_bpw=%.4f is below the minimum achievable for a %dx%d layer; "
            "using mid_dim=%d (achieved_bpw=%.4f > target).",
            target_bpw,
            out_features,
            in_features,
            mid_dim,
            achieved_bpw,
        )
    return mid_dim


# ---------------------------------------------------------------------------
# Trainable binary DBF adapter (QAT, SmoothSign STE)
# ---------------------------------------------------------------------------


class TrainableDBFAdapter(nn.Module):
    """DBF adapter whose ±1 signs are learned via STE and scales are fp.

    Computes the 5-stage DBF transform::

        y = (((x * scale_in) @ b1^T) * scale_mid @ b2^T) * scale_out

    where ``b1`` (mid, in) and ``b2`` (out, mid) are produced by SmoothSign
    STE during training.  ``scale_out`` is zero-initialized so the adapter
    starts as a no-op (matching the LoRA ``B = 0`` convention).

    Call :meth:`freeze_signs` to harden the signs in place (scales stay
    trainable), or :meth:`export_dbf` to read out the frozen tensors for an
    inference layer.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        in_features: int,
        out_features: int,
        mid_dim: int,
        k_smooth: float = 100.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.mid_dim = mid_dim
        self.k_smooth = k_smooth

        self.b1_latent = nn.Parameter(torch.randn(mid_dim, in_features, **kw) * 0.01)
        self.b2_latent = nn.Parameter(torch.randn(out_features, mid_dim, **kw) * 0.01)
        # scale_out is the *only* zero-initialized factor, so the adapter is a
        # no-op at step 0 (LoRA B=0 convention).  scale_in / scale_mid are
        # initialized to 1.0 (not a small value): every backward path to them
        # passes through `* scale_out`, so a small scale_out already makes
        # their first-step gradients tiny.  Shrinking scale_in / scale_mid too
        # would compound that and freeze them during the cold-start window.
        self.scale_in = nn.Parameter(torch.ones(in_features, **kw))
        self.scale_mid = nn.Parameter(torch.ones(mid_dim, **kw))
        self.scale_out = nn.Parameter(torch.zeros(out_features, **kw))
        self._signs_frozen = False

    def _quantize_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Binarize via the configured STE (SmoothSign or vanilla)."""
        return smooth_sign_ste(x, self.k_smooth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """5-stage binary-DBF forward (signs via STE, scales in fp)."""
        dt = x.dtype
        if self._signs_frozen:
            b1 = self.b1_frozen.to(dt)
            b2 = self.b2_frozen.to(dt)
        else:
            b1 = self._quantize_sign(self.b1_latent).to(dt)
            b2 = self._quantize_sign(self.b2_latent).to(dt)
        h = x * self.scale_in.to(dt)
        h = F.linear(h, b1)  # pylint: disable=not-callable
        h = h * self.scale_mid.to(dt)
        h = F.linear(h, b2)  # pylint: disable=not-callable
        return h * self.scale_out.to(dt)

    @staticmethod
    def _hard_sign(x: torch.Tensor) -> torch.Tensor:
        s = x.sign()
        return torch.where(s == 0, torch.ones_like(s), s)

    @torch.no_grad()
    def init_from_delta(self, delta: torch.Tensor) -> None:  # pylint: disable=too-many-locals
        """SVD-based ``QAT Full`` initialization from a target adapter weight.

        Implements Eq. (7) of the LoRDBA paper: given a target delta
        ``delta`` (shape ``(in_features, out_features)``, the weight the adapter
        output ``x @ W`` should approximate), take its rank-``mid_dim`` thin SVD
        ``delta ~= U_R diag(S_R) V_R^T`` and set

            b1_latent  <- U_R^T          (signs give B1 = sign(U_R))
            b2_latent  <- V_R            (signs give B2 = sign(V_R^T))
            scale_mid  <- S_R            (the R singular values, = beta)

        then recover the input/output channel scales (alpha = scale_in,
        gamma = scale_out) with one closed-form per-axis least-squares sweep
        that best fits ``delta`` given the now-fixed sign carriers and beta.

        A near-zero ``delta`` (e.g. an untrained warm-up) leaves the existing
        random initialization untouched.
        """
        if self._signs_frozen:
            raise RuntimeError("init_from_delta() must be called before freeze_signs().")

        target = delta.detach().to(device=self.b1_latent.device, dtype=torch.float32)
        if target.shape != (self.in_features, self.out_features):
            raise ValueError(
                f"delta shape {tuple(target.shape)} != "
                f"(in={self.in_features}, out={self.out_features})."
            )
        if torch.linalg.norm(target) < 1e-12:  # pylint: disable=not-callable
            logger.debug("init_from_delta: near-zero delta; keeping random init.")
            return

        # Rank-R thin SVD:  target = U diag(S) Vh,  U:(in,k), Vh:(k,out).
        # pylint: disable=not-callable
        u_mat, s_vec, vh_mat = torch.linalg.svd(target, full_matrices=False)
        r = min(self.mid_dim, s_vec.numel())

        kw = {"device": self.b1_latent.device, "dtype": self.b1_latent.dtype}
        b1 = torch.randn(self.mid_dim, self.in_features, **kw) * 0.01
        b2 = torch.randn(self.out_features, self.mid_dim, **kw) * 0.01
        beta = torch.zeros(self.mid_dim, **kw)
        b1[:r, :] = u_mat[:, :r].t().to(b1.dtype)          # U_R^T  -> (R, in)
        b2[:, :r] = vh_mat[:r, :].t().to(b2.dtype)         # V_R    -> (out, R)
        beta[:r] = s_vec[:r].to(beta.dtype)

        self.b1_latent.copy_(b1)
        self.b2_latent.copy_(b2)
        self.scale_mid.copy_(beta)

        # Closed-form least-squares sweep for alpha (scale_in) / gamma (scale_out):
        # minimize || target - diag(alpha) C diag(gamma) ||_F with the fixed
        # signed carriers C = sign(b1)^T diag(beta) sign(b2)^T.
        b1_sign = self._hard_sign(b1).t()                  # (in, R)
        b2_sign = self._hard_sign(b2).t()                  # (R, out)
        core = (b1_sign * beta.unsqueeze(0)) @ b2_sign     # (in, out)
        eps = 1e-8
        gamma = torch.ones(self.out_features, **kw)
        cg = core * gamma.unsqueeze(0)
        alpha = (target * cg).sum(dim=1) / cg.pow(2).sum(dim=1).clamp_min(eps)
        ac = alpha.unsqueeze(1) * core
        gamma = (target * ac).sum(dim=0) / ac.pow(2).sum(dim=0).clamp_min(eps)

        self.scale_in.copy_(alpha.to(self.scale_in.dtype))
        self.scale_out.copy_(gamma.to(self.scale_out.dtype))

    @torch.no_grad()
    def freeze_signs(self) -> None:
        """Freeze ±1 signs in place; scale vectors remain trainable."""
        if self._signs_frozen:
            return
        b1 = self._hard_sign(self.b1_latent.data)
        b2 = self._hard_sign(self.b2_latent.data)
        del self._parameters["b1_latent"]
        del self._parameters["b2_latent"]
        self.register_buffer("b1_frozen", b1)
        self.register_buffer("b2_frozen", b2)
        self._signs_frozen = True

    @torch.no_grad()
    def export_dbf(self) -> dict[str, torch.Tensor]:
        """Return detached DBF tensors for a :class:`DoubleBinaryLinear`.

        Keys map onto the 5-stage inference layer:
        ``scale_in -> dbf_Db``, ``b1 -> dbf_B``, ``scale_mid -> dbf_mid``,
        ``b2 -> dbf_A``, ``scale_out -> dbf_Da``.
        """
        if self._signs_frozen:
            b1, b2 = self.b1_frozen, self.b2_frozen
        else:
            b1 = self._hard_sign(self.b1_latent.data)
            b2 = self._hard_sign(self.b2_latent.data)
        return {
            "scale_in": self.scale_in.data.detach().clone(),
            "b1": b1.detach().clone(),
            "scale_mid": self.scale_mid.data.detach().clone(),
            "b2": b2.detach().clone(),
            "scale_out": self.scale_out.data.detach().clone(),
        }

    def storage_bits(self) -> int:
        """Total adapter storage in bits (1 bit/sign + fp16 scales)."""
        if self._signs_frozen:
            sign_bits = self.b1_frozen.numel() + self.b2_frozen.numel()
        else:
            sign_bits = self.b1_latent.numel() + self.b2_latent.numel()
        scale_bits = (
            self.scale_in.numel() + self.scale_mid.numel() + self.scale_out.numel()
        ) * 16
        return sign_bits + scale_bits


# ---------------------------------------------------------------------------
# Wrapper: frozen quantized base + trainable binary DBF adapter
# ---------------------------------------------------------------------------


class LoRDBALinear(nn.Module):
    """A frozen (quantized) Linear-like layer with a binary DBF LoRA adapter.

    Computes ``y = base(x) + scaling * adapter(x)``.  The base layer (e.g.
    ``GPTQLinear``, ``DoubleBinaryLinear`` or ``nn.Linear``) is frozen; only
    the adapter is trained.  After training, :meth:`freeze_to_inference`
    swaps the trainable adapter for a bit-packed
    :class:`DoubleBinaryLinear`.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        mid_dim: int,
        scaling: float,
        k_smooth: float = 100.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mid_dim = int(mid_dim)
        self.scaling = float(scaling)

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

        self.adapter: nn.Module = TrainableDBFAdapter(
            in_features=in_features,
            out_features=out_features,
            mid_dim=mid_dim,
            k_smooth=k_smooth,
            device=device,
            dtype=dtype,
        )
        self._frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen base output plus the scaled binary-DBF adapter output."""
        base_out = self.base_layer(x)
        delta = self.adapter(x)
        return base_out + self.scaling * delta.to(base_out.dtype)

    @torch.no_grad()
    def freeze_to_inference(self) -> None:
        """Materialize the trained adapter into a bit-packed DoubleBinaryLinear."""
        if self._frozen or not isinstance(self.adapter, TrainableDBFAdapter):
            return
        t = self.adapter.export_dbf()
        device = t["scale_in"].device
        self.adapter = DoubleBinaryLinear(
            dbf_Da=t["scale_out"],
            dbf_A=t["b2"],
            dbf_mid=t["scale_mid"],
            dbf_B=t["b1"],
            dbf_Db=t["scale_in"],
            bias=None,
            device=device,
            use_gemlite=False,
        )
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        """True once the adapter has been materialized for inference."""
        return self._frozen

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"mid={self.mid_dim}, scaling={self.scaling:.4f}, frozen={self._frozen}"
        )


def _infer_in_out_features(base_layer: nn.Module) -> tuple[int, int]:
    """Best-effort (in_features, out_features) for a quantized/plain Linear."""
    in_f = getattr(base_layer, "in_features", None)
    out_f = getattr(base_layer, "out_features", None)
    if in_f is not None and out_f is not None:
        return int(in_f), int(out_f)
    # DoubleBinaryLinear stores shapes only.
    if hasattr(base_layer, "_bp1_shape") and hasattr(base_layer, "_bp3_shape"):
        # pylint: disable=protected-access
        return int(base_layer._bp1_shape[1]), int(base_layer._bp3_shape[0])
    # MultipathMDBFLinear stores m (in) and n (out).
    if hasattr(base_layer, "m") and hasattr(base_layer, "n"):
        return int(base_layer.m), int(base_layer.n)
    raise ValueError(
        f"Cannot infer in/out features of base layer type {type(base_layer).__name__}. "
        "Extend _infer_in_out_features() for this layer type."
    )


# ---------------------------------------------------------------------------
# fp16 LoRA warm-up adapter (source of the SVD-init delta; Eq. 7)
# ---------------------------------------------------------------------------


class _WarmupLoRALinear(nn.Module):
    """A frozen base + standard fp16 low-rank LoRA, used only as the warm-up
    whose trained delta seeds the ``QAT Full`` SVD initialization.

    Computes ``y = base(x) + scaling * (x @ A^T) @ B^T`` with ``A`` Kaiming-
    initialized and ``B`` zero-initialized (the usual LoRA convention, so the
    delta starts at zero).  :meth:`delta_weight` returns the standard
    ``(out, in)`` weight increment ``scaling * B @ A`` after training.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(scaling)
        for param in self.base_layer.parameters():
            param.requires_grad_(False)
        kw = {"device": device, "dtype": dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, **kw))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, **kw))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen base output plus the scaled fp16 low-rank delta."""
        base_out = self.base_layer(x)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b)  # pylint: disable=not-callable
        return base_out + self.scaling * delta.to(base_out.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """Standard ``(out, in)`` weight increment ``scaling * B @ A``."""
        return self.scaling * (self.lora_b @ self.lora_a)


# ---------------------------------------------------------------------------
# Post-process
# ---------------------------------------------------------------------------


@dataclass
class PostProcessLoRDBA(PostProcessLoraSFT):
    """LoRDBA QAT post-process: train binary (DBF) LoRA adapters via STE.

    Inherits the full SFT / teacher-distillation / dataset / training-loop
    implementation from :class:`PostProcessLoraSFT` and only changes *what*
    is injected and trained: a binary DBF adapter (SmoothSign-STE signs +
    fp scaling vectors) instead of an fp16 low-rank pair.

    The DBF intermediate dimension ``mid_dim`` is chosen per layer by, in
    priority order, ``mid_dim`` (explicit), ``target_bpw`` (solved), or
    ``mid_factor`` (``mid = mid_factor * out*in / (out+in)``).

    Adapter output scaling reuses the inherited ``lora_alpha`` / ``lora_r``
    fields: ``scaling = lora_alpha / lora_r``.  ``lora_r`` also serves as the
    reference rank when ``bpw_basis == "lora"``.

    Examples:
        >>> from onecomp import Runner, ModelConfig, GPTQ
        >>> from onecomp.post_process import PostProcessLoRDBA
        >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
        >>> quantizer = GPTQ(wbits=4, groupsize=128)
        >>> runner = Runner(
        ...     model_config=model_config,
        ...     quantizer=quantizer,
        ...     post_processes=[
        ...         PostProcessLoRDBA(
        ...             data_files="train.jsonl",
        ...             target_bpw=1.5,
        ...             k_smooth=100.0,
        ...             output_dir="lordba_adapter",
        ...         )
        ...     ],
        ... )
        >>> runner.run()
    """

    # DBF adapter structure
    mid_dim: int | None = None
    mid_factor: float = 1.5
    target_bpw: float | None = None
    k_smooth: float = 100.0
    bpw_basis: str = "dw"  # "dw" (out*in) | "lora" (lora_r * (out+in))

    # QAT-Full SVD initialization (Eq. 7).  When True, a short fp16 LoRA
    # warm-up is trained first and its rank-mid_dim SVD seeds the binary
    # carriers + scales; this is the paper's main method.  Set False for the
    # "QAT Scratch" ablation (random sign init, no warm-up).
    svd_init: bool = True
    svd_warmup_ratio: float = 0.1  # warm-up updates as a fraction of QAT updates

    # Internal: per-layer warm-up deltas {layer_name: (out, in) tensor}, set by
    # _run_fp16_warmup() and consumed by _inject_lora_layers().
    _warmup_deltas: dict | None = None

    @property
    def scaling(self) -> float:
        """Adapter output scaling ``lora_alpha / lora_r`` (LoRA convention)."""
        return float(self.lora_alpha) / float(self.lora_r)

    def _compute_mid_dim(self, in_features: int, out_features: int) -> int:
        if self.mid_dim is not None:
            return _align8(max(int(self.mid_dim), 8))
        if self.target_bpw is not None:
            lora_rank = self.lora_r if self.bpw_basis == "lora" else 0
            return solve_mid_dim_for_bpw(
                out_features, in_features, self.target_bpw, lora_rank=lora_rank
            )
        mid = int(self.mid_factor * (out_features * in_features) / (out_features + in_features))
        return _align8(max(mid, 8))

    def _load_train_dataset(self):
        """Load the SFT dataset and drop empty / whitespace-only rows.

        Padded fully-empty samples tokenize to all-``-100`` labels, whose
        causal-LM loss is ``nan`` (a mean over zero valid tokens).  A single
        such micro-batch poisons the accumulated gradient, so they are
        filtered out here before tokenization.  (Common datasets such as
        WikiText contain ~30% blank lines.)
        """
        dataset = super()._load_train_dataset()
        col = self.text_column
        n_before = len(dataset)
        dataset = dataset.filter(
            lambda ex: isinstance(ex[col], str) and len(ex[col].strip()) > 0
        )
        n_after = len(dataset)
        if n_after < n_before:
            logger.info(
                "Filtered %d empty/whitespace rows (%d -> %d) to avoid nan loss.",
                n_before - n_after,
                n_before,
                n_after,
            )
        if n_after == 0:
            raise ValueError("All training samples were empty after filtering.")
        return dataset

    def _find_target_layer_names(self, quantized_model: nn.Module) -> list[str]:
        """Find target Linear-like layers by leaf-name (type-flexible).

        Unlike the parent (which is GPTQ-specific), this matches any module
        exposing ``in_features``/``out_features`` or a known quantized-linear
        layout, so DBF / MDBF / fp bases are supported too.
        """
        requested_targets = self.target_modules or _DEFAULT_TARGET_MODULES
        candidate_names: list[str] = []
        fallback_names: list[str] = []

        for name, module in quantized_model.named_modules():
            # Skip our own wrappers and their internal adapters: TrainableDBFAdapter
            # exposes in_features/out_features and would otherwise pass the
            # feature check and get wrapped a second time if injection ever runs
            # on an already-wrapped model.
            if isinstance(module, (LoRDBALinear, TrainableDBFAdapter, nn.ModuleList)):
                continue
            if _is_lm_head_module(name):
                continue
            try:
                _infer_in_out_features(module)
            except ValueError:
                continue
            fallback_names.append(name)
            leaf_name = name.rsplit(".", maxsplit=1)[-1]
            if leaf_name in requested_targets:
                candidate_names.append(name)

        if candidate_names:
            return candidate_names
        if fallback_names:
            logger.warning(
                "No layers matched target_modules=%s. Falling back to all "
                "Linear-like layers except lm_head.",
                requested_targets,
            )
            return fallback_names
        raise ValueError(
            "No Linear-like layers were found in `quantized_model` for PostProcessLoRDBA."
        )

    def _inject_lora_layers(self, quantized_model: nn.Module) -> int:
        target_names = self._find_target_layer_names(quantized_model)
        n_svd_init = 0
        for layer_name in target_names:
            base_layer = _resolve_submodule(quantized_model, layer_name)
            in_f, out_f = _infer_in_out_features(base_layer)
            mid = self._compute_mid_dim(in_f, out_f)
            wrapped = LoRDBALinear(
                base_layer=base_layer,
                in_features=in_f,
                out_features=out_f,
                mid_dim=mid,
                scaling=self.scaling,
                k_smooth=self.k_smooth,
            )
            # QAT Full: seed the carriers/scales from the warm-up delta (Eq. 7).
            # delta_weight is (out, in); the adapter approximates x @ W with W
            # of shape (in, out), and LoRDBALinear scales it by `scaling`, so
            # the SVD target is delta^T / scaling.
            if self._warmup_deltas is not None and layer_name in self._warmup_deltas:
                delta = self._warmup_deltas[layer_name]
                target = delta.t().to(torch.float32) / self.scaling
                wrapped.adapter.init_from_delta(target)
                n_svd_init += 1
            _replace_submodule(quantized_model, layer_name, wrapped)
        if target_names:
            sample = _resolve_submodule(quantized_model, target_names[0])
            logger.info(
                "Injected %d LoRDBA binary-DBF adapters "
                "(scaling=%.4f, k_smooth=%.1f, mid_dim[0]=%d, est_bpw[0]=%.3f, "
                "svd_init=%d/%d).",
                len(target_names),
                self.scaling,
                self.k_smooth,
                sample.mid_dim,
                estimate_dbf_bpw(sample.out_features, sample.in_features, sample.mid_dim),
                n_svd_init,
                len(target_names),
            )
        return len(target_names)

    def _collect_lora_state(self, quantized_model: nn.Module) -> dict:
        """Collect the bit-packed binary-DBF adapter state for export."""
        state: dict = {}
        for name, module in quantized_model.named_modules():
            if not isinstance(module, LoRDBALinear):
                continue
            if isinstance(module.adapter, TrainableDBFAdapter):
                t = module.adapter.export_dbf()
            elif isinstance(module.adapter, DoubleBinaryLinear):
                a = module.adapter
                # pylint: disable=protected-access
                t = {
                    "scale_in": a.scaling0.data,
                    "b1": a._unpack_bp(a.bp1, a._bp1_shape).to(torch.float32),
                    "scale_mid": a.scaling2.data,
                    "b2": a._unpack_bp(a.bp3, a._bp3_shape).to(torch.float32),
                    "scale_out": a.scaling4.data,
                }
                # pylint: enable=protected-access
            else:
                continue
            state[name] = {
                "scale_in": t["scale_in"].detach().to(torch.float16).cpu(),
                "bp1": pack_binary(t["b1"].cpu()),
                "scale_mid": t["scale_mid"].detach().to(torch.float16).cpu(),
                "bp3": pack_binary(t["b2"].cpu()),
                "scale_out": t["scale_out"].detach().to(torch.float16).cpu(),
                "in_features": module.in_features,
                "out_features": module.out_features,
                "mid_dim": module.mid_dim,
                "scaling": module.scaling,
            }
        return state

    def _save_adapter(self, quantized_model: nn.Module) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        state = self._collect_lora_state(quantized_model)
        if not state:
            logger.warning("No LoRDBA adapter state found to save.")
            return

        adapter_path = os.path.join(self.output_dir, "adapter_model.bin")
        config_path = os.path.join(self.output_dir, "adapter_config.json")
        torch.save(state, adapter_path)

        total_bits = 0
        total_weights = 0
        for entry in state.values():
            mid = entry["mid_dim"]
            n, m = entry["out_features"], entry["in_features"]
            total_bits += int(estimate_dbf_bpw(n, m, mid) * n * m)
            total_weights += n * m
        avg_bpw = total_bits / total_weights if total_weights else 0.0

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": "lordba_qat_dbf",
                    "lora_alpha": self.lora_alpha,
                    "lora_r": self.lora_r,
                    "scaling": self.scaling,
                    "mid_dim": self.mid_dim,
                    "mid_factor": self.mid_factor,
                    "target_bpw": self.target_bpw,
                    "bpw_basis": self.bpw_basis,
                    "k_smooth": self.k_smooth,
                    "svd_init": self.svd_init,
                    "svd_warmup_ratio": self.svd_warmup_ratio,
                    "training_mode": "qat_full" if self.svd_init else "qat_scratch",
                    "target_modules": list(self.target_modules or _DEFAULT_TARGET_MODULES),
                    "num_adapters": len(state),
                    "avg_adapter_bpw": avg_bpw,
                    "teacher_loss_weight": self.teacher_loss_weight,
                    "teacher_loss_type": self.teacher_loss_type,
                    "intermediate_block_loss_weight": self.intermediate_block_loss_weight,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        logger.info(
            "Saved LoRDBA adapter to %s (%d adapters, avg %.3f bpw).",
            self.output_dir,
            len(state),
            avg_bpw,
        )

    def _materialize_binary_adapters(self, quantized_model: nn.Module) -> int:
        """Convert trained STE adapters to bit-packed DoubleBinaryLinear in place."""
        n = 0
        for module in quantized_model.modules():
            if isinstance(module, LoRDBALinear) and not module.is_frozen:
                module.freeze_to_inference()
                n += 1
        if n:
            logger.info("Materialized %d LoRDBA adapters to bit-packed DoubleBinaryLinear.", n)
        return n

    @torch.no_grad()
    def _extract_warmup_deltas(self, model: nn.Module) -> dict:
        """Read trained warm-up deltas, then unwrap to restore the base layers."""
        deltas: dict = {}
        warmup_names = [
            name for name, mod in model.named_modules() if isinstance(mod, _WarmupLoRALinear)
        ]
        for name in warmup_names:
            module = _resolve_submodule(model, name)
            deltas[name] = module.delta_weight().detach().to("cpu", torch.float32)
            _replace_submodule(model, name, module.base_layer)  # restore frozen base
        return deltas

    def _run_fp16_warmup(  # pylint: disable=too-many-locals
        self, quantized_model: nn.Module, model_config
    ) -> dict:
        """Train a short fp16 LoRA warm-up and return its per-layer ``(out, in)``
        deltas for QAT-Full SVD initialization (Eq. 7).

        Reuses the inherited dataset/tokenizer/collate machinery but runs a
        compact CE-only loop (no teacher / intermediate distillation): this is
        purely a task-informed seed for the binary carriers.
        """
        tokenizer = model_config.load_tokenizer()
        train_dataset = self._tokenize_dataset(self._load_train_dataset(), tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch,
        )

        # Inject fp16 LoRA warm-up adapters at the same rank as the DBF carriers.
        for param in quantized_model.parameters():
            param.requires_grad_(False)
        target_names = self._find_target_layer_names(quantized_model)
        for layer_name in target_names:
            base_layer = _resolve_submodule(quantized_model, layer_name)
            in_f, out_f = _infer_in_out_features(base_layer)
            mid = self._compute_mid_dim(in_f, out_f)
            _replace_submodule(
                quantized_model,
                layer_name,
                _WarmupLoRALinear(base_layer, in_f, out_f, mid, self.scaling),
            )

        train_device = self._resolve_train_device(model_config)
        use_bf16 = self._resolve_use_bf16(train_device)
        warmup_params = [p for p in quantized_model.parameters() if p.requires_grad]

        total_updates = max(
            1,
            math.ceil((len(train_loader) * self.epochs) / self.gradient_accumulation_steps),
        )
        n_warmup = max(1, int(total_updates * self.svd_warmup_ratio))
        logger.info(
            "QAT-Full warm-up: training fp16 LoRA for %d step(s) (%.0f%% of %d QAT updates).",
            n_warmup,
            self.svd_warmup_ratio * 100,
            total_updates,
        )

        quantized_model.to(train_device)
        original_use_cache = None
        if hasattr(quantized_model, "config") and hasattr(quantized_model.config, "use_cache"):
            original_use_cache = bool(quantized_model.config.use_cache)
            quantized_model.config.use_cache = False
        optimizer = torch.optim.AdamW(warmup_params, lr=self.lr, weight_decay=self.weight_decay)

        try:
            quantized_model.train()
            autocast_enabled = train_device.type == "cuda" and use_bf16
            step = 0
            done = False
            for _epoch in range(self.epochs):
                if done:
                    break
                for batch in train_loader:
                    batch.pop("sample_idx", None)
                    batch = {k: v.to(train_device) for k, v in batch.items()}
                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if autocast_enabled
                        else nullcontext()
                    ):
                        out = quantized_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                    out.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step >= n_warmup:
                        done = True
                        break
        finally:
            if original_use_cache is not None:
                quantized_model.config.use_cache = original_use_cache
            quantized_model.eval()
            deltas = self._extract_warmup_deltas(quantized_model)
            quantized_model.to("cpu")
            torch.cuda.empty_cache()

        logger.info("QAT-Full warm-up complete: extracted %d layer deltas.", len(deltas))
        return deltas

    def run(self, quantized_model: nn.Module, model_config) -> None:
        """Train binary-DBF adapters, then materialize them for inference.

        With ``svd_init`` (QAT Full), a short fp16 LoRA warm-up runs first and
        its rank-mid_dim SVD seeds the binary carriers and scales (Eq. 7).
        """
        # When svd_init is on, seed the SVD from a trained fp16 LoRA delta.
        # If the caller already supplied per-layer deltas (e.g. a fully-trained
        # reference LoRA, matching the paper's "full fp16 seed" recipe), use
        # those and skip the internal short warm-up.
        if self.svd_init and self._warmup_deltas is None:
            self._warmup_deltas = self._run_fp16_warmup(quantized_model, model_config)
        try:
            super().run(quantized_model, model_config)
        finally:
            self._warmup_deltas = None
        # super().run() leaves the model on CPU in eval mode with the trained
        # (STE) adapters; harden them to bit-packed inference layers so the
        # subsequent Runner evaluation uses the truly-compressed adapter.
        self._materialize_binary_adapters(quantized_model)
