#!/usr/bin/env python3
"""Ablation ④ Llama2-7B: MDBF 1bpw - standard SVD without Hessian weighting, scale_bits=16.

Configuration:
  - W_tilde = W @ Q @ diag(sqrt(λ))  (no Q^T; ignores eigenvector rotation)
  - scale_bits=16  (FP16 envelope parameters counted in BPW budget)
  → actual BPW ≈ 1.000

GPU: cuda:3
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import torch

import onecomp.quantizer.mdbf.initialize as _init_mod
import onecomp.quantizer.mdbf.mdbf_layer as _layer_mod

# Monkey-patch: lowrank_osvd without Q^T (eigenvector back-rotation omitted).
# scale_bits=16 is retained (default).
def _lowrank_osvd_no_qt(W, H, r, ridge=1e-4):
    """lowrank_osvd without Q^T: W_tilde = W @ Q @ diag(sqrt(λ)).

    The eigenvector back-rotation (@ Q^T) is omitted, so the weight-space
    metric is not properly inverted.  Used as an ablation baseline.
    """
    from onecomp.quantizer.mdbf.initialize import _lowrank_svd_standard, cleanup_gpu_memory, ensure_float32
    n, m = W.shape
    r = min(r, min(n, m))
    W_fp32 = ensure_float32(W)
    H_fp32 = ensure_float32(H)
    diag_mean = H_fp32.diag().mean().clamp(min=1e-12)
    H_reg = H_fp32 + ridge * diag_mean * torch.eye(m, device=H_fp32.device, dtype=H_fp32.dtype)
    try:
        eig_vals, eig_vecs = torch.linalg.eigh(H_reg)
    except RuntimeError:
        del H_reg, H_fp32
        return _lowrank_svd_standard(W, r, W.dtype)
    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)
    # W_tilde = W @ Q @ diag(sqrt(λ))  -- eigenvector back-rotation omitted
    W_tilde = W_fp32 @ eig_vecs @ torch.diag(sqrt_eig)
    del H_reg
    eps_svd = 1e-6 * W_tilde.abs().max().clamp(min=1e-12)
    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde + eps_svd * torch.randn_like(W_tilde), full_matrices=False)
    del W_tilde
    r_eff = min(r, S_w.numel())
    sqrt_S = torch.sqrt(S_w[:r_eff].clamp(min=1e-12))
    U_prime = U_w[:, :r_eff] * sqrt_S[None, :]
    inv_sqrt_eig = 1.0 / sqrt_eig
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ Vh_w[:r_eff].T @ torch.diag(sqrt_S)
    del eig_vals, eig_vecs, sqrt_eig, inv_sqrt_eig, S_w, Vh_w, H_fp32, W_fp32
    cleanup_gpu_memory()
    return U_prime.to(W.dtype), V_prime.to(W.dtype)

_init_mod.lowrank_osvd = _lowrank_osvd_no_qt
_layer_mod.lowrank_osvd = _lowrank_osvd_no_qt

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.mdbf import MDBF

MODEL_PATH = "/data3/yoshida/qep-dev/models/Llama-2-7b-hf"
DEVICE = "cuda:3"
TARGET_BITS = 1.0
L, P = 8, 1
OUTPUT_FILE = Path(__file__).parent / "results_llama2_7b_1bpw_scalebits_only.json"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def _exclude_kws(num_layers, first_n=4, last_n=4):
    skip = set(range(first_n)) | set(range(num_layers - last_n, num_layers))
    return [f"model.layers.{i}." for i in sorted(skip)]

def main():
    print("=" * 80)
    print("Llama2-7B MDBF 1bpw - Ablation ④: no Hessian rotation, scale_bits=16")
    print(f"  W_tilde = W @ Q @ diag(sqrt(λ))  (eigenvector back-rotation omitted)")
    print(f"  scale_bits=16  → actual BPW ≈ 1.000")
    print(f"  device: {DEVICE}")
    print("=" * 80)
    model_config = ModelConfig(path=MODEL_PATH, device=DEVICE)
    exclude_kws = _exclude_kws(model_config.load_config().num_hidden_layers)
    quantizer = MDBF(target_bits=TARGET_BITS, l=L, P=P, svd_mode="svd",
        use_admm=True, admm_iters=1000, admm_inner_iters=3, admm_reg=0.03,
        use_gradient_refine=True, gradient_iters=1500, gradient_lr=0.01,
        activation_aware=True, act_init="osvd", exclude_layer_keywords=exclude_kws)
    runner = Runner(
        model_config=model_config, quantizer=quantizer,
        calibration_config=CalibrationConfig(calibration_dataset="wikitext2", num_calibration_samples=128, max_length=2048, seed=0),
        qep=True, qep_config=QEPConfig(percdamp=0.01, perccorr=0.5, device=DEVICE))
    t0 = time.time()
    runner.run()
    elapsed = time.time() - t0
    _, ppl, _ = runner.calculate_perplexity(original_model=False, dequantized_model=True, quantized_model=False,
        dataset_name="wikitext", dataset_config="wikitext-2-raw-v1")
    _, acc, _ = runner.calculate_accuracy(original_model=False, dequantized_model=True, quantized_model=False,
        tasks=["arc_easy", "piqa"], num_fewshot=0)
    result = {"experiment": "ablation_no_hessian_rotation_scale16", "model": "Llama-2-7b-hf", "target_bits": TARGET_BITS,
        "l": L, "P": P, "hessian_weighted_svd": False, "scale_bits": 16, "actual_bpw_approx": 1.000,
        "ppl_wikitext2": ppl, "acc": acc, "elapsed_sec": round(elapsed, 1)}
    print(f"\n{'='*80}\n[RESULT] Ablation ④: no Hessian rotation, scale_bits=16\n  PPL: {ppl}\n  ACC: {acc}\n  Elapsed: {elapsed:.0f}s\n{'='*80}")
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to: {OUTPUT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
