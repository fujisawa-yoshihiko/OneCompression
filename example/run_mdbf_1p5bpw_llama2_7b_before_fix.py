#!/usr/bin/env python3
"""Llama2-7B MDBF 1.5bpw (paper settings) - before fix + text generation.
GPU: cuda:3
"""
from __future__ import annotations
import json, logging, sys, time, math
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import onecomp.quantizer.mdbf.utils as _utils_mod
import onecomp.quantizer.mdbf.initialize as _init_mod
import onecomp.quantizer.mdbf.mdbf_layer as _layer_mod

def _buggy_lowrank_osvd(W, H, r, ridge=1e-4):
    from onecomp.quantizer.mdbf.initialize import _lowrank_svd_standard, cleanup_gpu_memory, ensure_float32
    n, m = W.shape
    r = min(r, min(n, m))
    W_fp32 = ensure_float32(W)
    H_fp32 = ensure_float32(H)
    H_reg = H_fp32 + ridge * H_fp32.diag().mean().clamp(min=1e-12) * torch.eye(m, device=H_fp32.device, dtype=H_fp32.dtype)
    try:
        eig_vals, eig_vecs = torch.linalg.eigh(H_reg)
    except RuntimeError:
        del H_reg, H_fp32
        return _lowrank_svd_standard(W, r, W.dtype)
    sqrt_eig = torch.sqrt(eig_vals.clamp(min=1e-12))
    W_tilde = W_fp32 @ eig_vecs @ torch.diag(sqrt_eig)
    U_w, S_w, Vh_w = torch.linalg.svd(
        W_tilde + 1e-6 * W_tilde.abs().max().clamp(min=1e-12) * torch.randn_like(W_tilde),
        full_matrices=False,
    )
    r_eff = min(r, S_w.numel())
    sqrt_S = torch.sqrt(S_w[:r_eff].clamp(min=1e-12))
    inv_sqrt_eig = 1.0 / sqrt_eig
    U_prime = U_w[:, :r_eff] * sqrt_S[None, :]
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ Vh_w[:r_eff].T @ torch.diag(sqrt_S)
    cleanup_gpu_memory()
    return U_prime.to(W.dtype), V_prime.to(W.dtype)

def _rank_scale0(n, m, b_target, l=1, P=2, min_rank=1, rounding="floor", scale_bits=0):
    scale_bits = 0
    r_real = (b_target * n * m / P) / (n + m)
    r = int(math.floor(r_real)) if rounding == "floor" else int(round(r_real))
    return max(min_rank, min(r, min(n, m)))

_init_mod.lowrank_osvd = _buggy_lowrank_osvd
_layer_mod.lowrank_osvd = _buggy_lowrank_osvd
_utils_mod.rank_from_bpw = _rank_scale0
_layer_mod.rank_from_bpw = _rank_scale0

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.mdbf import MDBF

from llama2_7b_gen_utils import run_text_generation

MODEL_PATH = "/data3/yoshida/qep-dev/models/Llama-2-7b-hf"
DEVICE = "cuda:3"
TARGET_BITS = 1.5
L, P = 8, 1
OUTPUT_FILE = Path(__file__).parent / "results_llama2_7b_1p5bpw_before_fix.json"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def _exclude_kws(num_layers, first_n=4, last_n=4):
    skip = set(range(first_n)) | set(range(num_layers - last_n, num_layers))
    return [f"model.layers.{i}." for i in sorted(skip)]

def main():
    print("=" * 80)
    print("Llama2-7B MDBF 1.5bpw (paper) - BEFORE FIX + generation")
    print(f"  device: {DEVICE}")
    print("=" * 80)
    model_config = ModelConfig(path=MODEL_PATH, device=DEVICE)
    exclude_kws = _exclude_kws(model_config.load_config().num_hidden_layers)
    quantizer = MDBF(
        target_bits=TARGET_BITS, l=L, P=P, svd_mode="svd",
        use_admm=True, admm_iters=1000, admm_inner_iters=3, admm_reg=0.03,
        use_gradient_refine=True, gradient_iters=1500, gradient_lr=0.01,
        activation_aware=True, act_init="osvd", exclude_layer_keywords=exclude_kws,
    )
    runner = Runner(
        model_config=model_config, quantizer=quantizer,
        calibration_config=CalibrationConfig(
            calibration_dataset="wikitext2", num_calibration_samples=128,
            max_length=2048, seed=0,
        ),
        qep=True, qep_config=QEPConfig(percdamp=0.01, perccorr=0.5, device=DEVICE),
    )
    t0 = time.time()
    runner.run()
    quant_elapsed = time.time() - t0

    _, ppl, _ = runner.calculate_perplexity(
        original_model=False, dequantized_model=True, quantized_model=False,
        dataset_name="wikitext", dataset_config="wikitext-2-raw-v1",
    )
    _, acc, _ = runner.calculate_accuracy(
        original_model=False, dequantized_model=True, quantized_model=False,
        tasks=["arc_easy", "piqa"], num_fewshot=0,
    )

    print("\n" + "=" * 80)
    print("Text generation (dequantized weights)")
    print("=" * 80)
    generations = run_text_generation(runner, DEVICE)
    elapsed = time.time() - t0

    result = {
        "experiment": "before_fix", "model": "Llama-2-7b-hf",
        "target_bits": TARGET_BITS, "l": L, "P": P,
        "hessian_fix": False, "scale_bits": 0,
        "ppl_wikitext2": ppl, "acc": acc,
        "generations": generations,
        "quant_elapsed_sec": round(quant_elapsed, 1),
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"\n{'='*80}\n[RESULT] 1.5bpw before_fix\n  PPL: {ppl}\n  ACC: {acc}\n  Elapsed: {elapsed:.0f}s\n{'='*80}")
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to: {OUTPUT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
