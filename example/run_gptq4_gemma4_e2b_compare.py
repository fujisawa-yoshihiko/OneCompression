#!/usr/bin/env python3
"""Gemma 4 E2B: GPTQ 4-bit comparison (aligned with MDBF ablation setup).

Settings aligned with MDBF ablation runs:
  - calibration: wikitext2, 128 samples, max_length=2048
  - per_layer_* kept FP16 (same excludes as MDBF ablation)
  - GPTQ: wbits=4, groupsize=128, qep=False
  - eval: dequantized only (quantized_model=False; Gemma4 Triton kernel fails)

GPU: cuda:1
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner
from llama2_7b_gen_utils import build_gemma4_ablation_exclude_keywords, run_text_generation

MODEL_ID = "google/gemma-4-E2B"
DEVICE = "cuda:1"
OUTPUT_FILE = Path(__file__).parent / "results_gemma4_e2b_gptq4_compare.json"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    print("=" * 80)
    print("Gemma 4 E2B: GPTQ 4-bit comparison")
    print(f"  wbits=4, groupsize=128, qep=False")
    print(f"  per_layer_* FP16 (MDBF ablation match)")
    print(f"  device: {DEVICE}")
    print("=" * 80)

    model_config = ModelConfig(model_id=MODEL_ID, device=DEVICE)
    exclude_kws = build_gemma4_ablation_exclude_keywords(model_config)

    quantizer = GPTQ(
        wbits=4,
        groupsize=128,
        exclude_layer_keywords=exclude_kws,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            calibration_dataset="wikitext2",
            num_calibration_samples=128,
            max_length=2048,
            seed=0,
            batch_size=1,  # Gemma4 vocab=262K; batch=128 OOMs at lm_head (~128 GiB)
        ),
        qep=False,
    )

    t0 = time.time()
    runner.run()
    quant_elapsed = time.time() - t0

    orig_ppl, dequant_ppl, quant_ppl = runner.calculate_perplexity(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
    )
    _, dequant_acc, quant_acc = runner.calculate_accuracy(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
        tasks=["arc_easy", "piqa"],
        num_fewshot=0,
    )

    print("\n" + "=" * 80)
    print("Text generation (dequantized weights)")
    print("=" * 80)
    generations = run_text_generation(runner, DEVICE)
    elapsed = time.time() - t0

    result = {
        "experiment": "gptq4_compare",
        "model": MODEL_ID,
        "method": "GPTQ",
        "wbits": 4,
        "groupsize": 128,
        "qep": False,
        "ablation": "per_layer_* kept FP16",
        "eval_mode": "dequantized_only (MDBF-aligned; quantized_model skipped for Gemma4)",
        "ppl_wikitext2": dequant_ppl,
        "ppl_wikitext2_original": orig_ppl,
        "ppl_wikitext2_dequantized": dequant_ppl,
        "ppl_wikitext2_quantized": quant_ppl,
        "acc": dequant_acc,
        "acc_quantized": quant_acc,
        "generations": generations,
        "quant_elapsed_sec": round(quant_elapsed, 1),
        "elapsed_sec": round(elapsed, 1),
    }

    print(f"\n{'='*80}")
    print("[RESULT] GPTQ 4-bit")
    print(f"  PPL (dequant):   {dequant_ppl}")
    print(f"  ACC:             {dequant_acc}")
    print(f"  Elapsed:         {elapsed:.0f}s")
    print("=" * 80)

    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
