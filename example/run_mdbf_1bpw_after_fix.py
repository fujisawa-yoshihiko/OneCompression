#!/usr/bin/env python3
"""Experiment: MDBF 1bpw with Hessian FIX (commit a9b46df).

修正後 (after fix):
  - W_tilde = W @ Q @ diag(sqrt(λ)) @ Q^T  (固有ベクトルの転置を含む)
  - scale_bits=16  (FP16スケールをBPWに含める)

GPU: cuda:0
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.mdbf import MDBF

MODEL_PATH = "/data3/yoshida/qep-dev/models/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda:0"
TARGET_BITS = 1.0
L = 8
P = 1
OUTPUT_FILE = Path(__file__).parent / "results_1bpw_after_fix.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _build_exclude_keywords(num_layers: int, first_n: int = 4, last_n: int = 4) -> list[str]:
    skip = set(range(first_n)) | set(range(num_layers - last_n, num_layers))
    return [f"model.layers.{i}." for i in sorted(skip)]


def main() -> int:
    print("=" * 80)
    print("MDBF 1bpw - AFTER FIX (commit a9b46df)")
    print(f"  W_tilde = W @ Q @ diag(sqrt(λ)) @ Q^T  [FIXED]")
    print(f"  scale_bits = 16  [FIXED]")
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
        "experiment": "after_fix",
        "commit": "a9b46df",
        "target_bits": TARGET_BITS,
        "l": L,
        "P": P,
        "hessian_fix": True,
        "scale_bits": 16,
        "ppl_wikitext2": dequant_ppl,
        "acc": dequant_acc,
        "elapsed_sec": round(elapsed, 1),
    }

    print("\n" + "=" * 80)
    print("[RESULT] after_fix (a9b46df)")
    print(f"  PPL (wikitext2): {dequant_ppl}")
    print(f"  ACC (arc_easy, piqa): {dequant_acc}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 80)

    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
