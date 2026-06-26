#!/usr/bin/env python3
"""Experiment ② Llama2-7B: MDBF 1bpw - after fix (a9b46df, current HEAD).
  Hessian✅ (Q^T含む), scale_bits=16 → 実際BPW≈1.000
GPU: cuda:3
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.mdbf import MDBF

MODEL_PATH = "/data3/yoshida/qep-dev/models/Llama-2-7b-hf"
DEVICE = "cuda:3"
TARGET_BITS = 1.0
L, P = 8, 1
OUTPUT_FILE = Path(__file__).parent / "results_llama2_7b_1bpw_after_fix.json"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def _exclude_kws(num_layers, first_n=4, last_n=4):
    skip = set(range(first_n)) | set(range(num_layers - last_n, num_layers))
    return [f"model.layers.{i}." for i in sorted(skip)]

def main():
    print("=" * 80)
    print("Llama2-7B MDBF 1bpw - ② AFTER FIX (a9b46df)")
    print(f"  Hessian: Q @ diag(√λ) @ Q^T [FIXED] | scale_bits=16 [FIXED] → 実際BPW≈1.000")
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
    result = {"experiment": "after_fix", "model": "Llama-2-7b-hf", "target_bits": TARGET_BITS,
        "l": L, "P": P, "hessian_fix": True, "scale_bits": 16, "actual_bpw_approx": 1.000,
        "ppl_wikitext2": ppl, "acc": acc, "elapsed_sec": round(elapsed, 1)}
    print(f"\n{'='*80}\n[RESULT] ② after_fix\n  PPL: {ppl}\n  ACC: {acc}\n  Elapsed: {elapsed:.0f}s\n{'='*80}")
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to: {OUTPUT_FILE}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
