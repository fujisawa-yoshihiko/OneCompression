#!/usr/bin/env python3
"""Experiment ④: MDBF 1bpw - scale_bitsのみ修正 (Hessianはバグあり).

  - W_tilde = W @ Q @ diag(sqrt(λ))  [BUG: Q^T missing]
  - scale_bits = 16  [FIXED]
  → 実際BPW ≈ 1.000 (修正後と同じBPW条件でHessianのみ比較)

GPU: cuda:3
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch

# -----------------------------------------------------------------------
# モンキーパッチ: Hessianのみ旧挙動(Q^T欠落)に戻す。scale_bitsは修正済みを使う。
# -----------------------------------------------------------------------
import onecomp.quantizer.mdbf.initialize as _init_mod
import onecomp.quantizer.mdbf.mdbf_layer as _layer_mod


def _buggy_lowrank_osvd(W, H, r, ridge=1e-4):
    """修正前のlowrank_osvd: W_tilde = W @ Q @ diag(sqrt(λ)) (Q^T欠落)"""
    from onecomp.quantizer.mdbf.initialize import (
        _lowrank_svd_standard,
        cleanup_gpu_memory,
        ensure_float32,
    )

    n, m = W.shape
    r = min(r, min(n, m))

    W_fp32 = ensure_float32(W)
    H_fp32 = ensure_float32(H)

    diag_mean = H_fp32.diag().mean().clamp(min=1e-12)
    eps = ridge * diag_mean
    H_reg = H_fp32 + eps * torch.eye(m, device=H_fp32.device, dtype=H_fp32.dtype)

    try:
        eig_vals, eig_vecs = torch.linalg.eigh(H_reg)
    except RuntimeError:
        del H_reg, H_fp32
        return _lowrank_svd_standard(W, r, W.dtype)

    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)

    # ★ バグ: Q^T が欠落
    W_tilde = W_fp32 @ eig_vecs @ torch.diag(sqrt_eig)
    del H_reg

    eps_svd = 1e-6 * W_tilde.abs().max().clamp(min=1e-12)
    W_tilde_reg = W_tilde + eps_svd * torch.randn_like(W_tilde)
    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde_reg, full_matrices=False)
    del W_tilde, W_tilde_reg

    r_eff = min(r, S_w.numel())
    U_r = U_w[:, :r_eff]
    S_r = S_w[:r_eff]
    V_r = Vh_w[:r_eff, :].T
    del U_w, S_w, Vh_w

    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))
    U_prime = U_r * sqrt_S[None, :]

    inv_sqrt_eig = 1.0 / sqrt_eig
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)

    del eig_vals, eig_vecs, sqrt_eig, inv_sqrt_eig, U_r, S_r, V_r, sqrt_S
    del H_fp32, W_fp32
    cleanup_gpu_memory()

    return U_prime.to(W.dtype), V_prime.to(W.dtype)


_init_mod.lowrank_osvd = _buggy_lowrank_osvd
_layer_mod.lowrank_osvd = _buggy_lowrank_osvd
# -----------------------------------------------------------------------

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.mdbf import MDBF

MODEL_PATH = "/data3/yoshida/qep-dev/models/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda:3"
TARGET_BITS = 1.0
L = 8
P = 1
OUTPUT_FILE = Path(__file__).parent / "results_1bpw_scalebits_only.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _build_exclude_keywords(num_layers: int, first_n: int = 4, last_n: int = 4) -> list[str]:
    skip = set(range(first_n)) | set(range(num_layers - last_n, num_layers))
    return [f"model.layers.{i}." for i in sorted(skip)]


def main() -> int:
    print("=" * 80)
    print("MDBF 1bpw - ④ scale_bitsのみ修正 (Hessianはバグあり)")
    print(f"  W_tilde = W @ Q @ diag(sqrt(λ))  [BUG: Q^T missing]")
    print(f"  scale_bits = 16  [FIXED]  → 実際BPW≈1.000")
    print(f"  target_bits = {TARGET_BITS}, l={L}, P={P}")
    print(f"  device: {DEVICE}")
    print("=" * 80)

    model_config = ModelConfig(path=MODEL_PATH, device=DEVICE)
    num_layers = model_config.load_config().num_hidden_layers
    exclude_keywords = _build_exclude_keywords(num_layers)

    quantizer = MDBF(
        target_bits=TARGET_BITS,
        l=L,
        P=P,
        svd_mode="svd",
        use_admm=True,
        admm_iters=1000,
        admm_inner_iters=3,
        admm_reg=0.03,
        use_gradient_refine=True,
        gradient_iters=1500,
        gradient_lr=0.01,
        activation_aware=True,
        act_init="osvd",
        exclude_layer_keywords=exclude_keywords,
    )

    calibration_config = CalibrationConfig(
        calibration_dataset="wikitext2",
        num_calibration_samples=128,
        max_length=2048,
        seed=0,
    )

    qep_config = QEPConfig(percdamp=0.01, perccorr=0.5, device=DEVICE)

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration_config,
        qep=True,
        qep_config=qep_config,
    )

    t0 = time.time()
    runner.run()
    elapsed = time.time() - t0

    _, dequant_ppl, _ = runner.calculate_perplexity(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
    )
    _, dequant_acc, _ = runner.calculate_accuracy(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
        tasks=["arc_easy", "piqa"],
        num_fewshot=0,
    )

    result = {
        "experiment": "scalebits_only_fix",
        "target_bits": TARGET_BITS,
        "l": L,
        "P": P,
        "hessian_fix": False,
        "scale_bits": 16,
        "actual_bpw_approx": 1.000,
        "ppl_wikitext2": dequant_ppl,
        "acc": dequant_acc,
        "elapsed_sec": round(elapsed, 1),
    }

    print("\n" + "=" * 80)
    print("[RESULT] ④ scalebits_only_fix")
    print(f"  PPL (wikitext2): {dequant_ppl}")
    print(f"  ACC (arc_easy, piqa): {dequant_acc}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 80)

    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
