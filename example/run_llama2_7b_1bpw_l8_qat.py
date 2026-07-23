#!/usr/bin/env python3
"""Llama2-7B: MDBF 1bpw l=8 + QAT (pure NTP, optimize_binary=True).

QAT settings:
  - w_distill=0.0, w_ntp=1.0  -> pure NTP loss (no teacher model needed)
  - optimize_binary=True      -> also optimize binary signs via STE
  - num_calibration_samples=512, max_length=1024, epochs=10
  - DeepSpeed ZeRO-2 + CPU optimizer offload (runs on a single GPU)

Intended for comparison against GlobalPTQ (KL-only, amp-only, 64 samples,
3 epochs, PPL=14.32).

Usage:
  # Run QAT (load pickle -> QAT -> evaluate)
  CUDA_VISIBLE_DEVICES=1 python run_llama2_7b_1bpw_l8_qat.py

  # Skip QAT and evaluate only (when a saved state already exists)
  CUDA_VISIBLE_DEVICES=1 python run_llama2_7b_1bpw_l8_qat.py --skip-qat

  # Vary sample count / epoch count
  CUDA_VISIBLE_DEVICES=1 python run_llama2_7b_1bpw_l8_qat.py \\
    --num-samples 256 --epochs 5

  # Change the calibration dataset (wikitext2 / HF Hub ID / local path)
  CUDA_VISIBLE_DEVICES=1 python run_llama2_7b_1bpw_l8_qat.py \\
    --calibration-dataset wikitext2
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import gc
import json
import logging
import pickle
import time
from pathlib import Path

import torch

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.mdbf import MDBF
from onecomp.utils.model_inputs import add_model_specific_inputs
from onecomp_globalptq import GlobalPTQDistributed

from llama2_7b_gen_utils import build_exclude_keywords

MODEL_PATH = "/data3/yoshida/qep-dev/models/Llama-2-7b-hf"
DEVICE = "cuda:0"  # physical GPU 1 appears as cuda:0 due to CUDA_VISIBLE_DEVICES=1
TARGET_BITS = 1.0
L, P = 8, 1

MDBF_PICKLE = Path(__file__).parent / "mdbf_l8_llama2_7b_1bpw.pkl"
QAT_STATE_DEFAULT = Path(__file__).parent / "qat_state_llama2_7b_1bpw_l8.pt"
QAT_MODEL_DEFAULT = Path(__file__).parent / "qat_quantized_model_llama2_7b_1bpw_l8.pkl"
OUTPUT_FILE_DEFAULT = Path(__file__).parent / "results_llama2_7b_1bpw_l8_qat.json"
DS_CONFIG = Path(__file__).parent / "ds_zero2_mdbf_offload.json"

REF_GP_RESULTS = Path(__file__).parent / "results_llama2_7b_1bpw_l8_globalptq.json"

DECODE_CONFIGS = {
    "greedy": {"do_sample": False, "max_new_tokens": 128},
    "sampling_rp": {
        "do_sample": True, "temperature": 0.8, "top_p": 0.9, "top_k": 50,
        "repetition_penalty": 1.15, "no_repeat_ngram_size": 3, "max_new_tokens": 128,
    },
}

PROMPTS = [
    "The future of artificial intelligence",
    "Hello, how are you today?",
    "Write a bubble sort algorithm in Python.",
    "Let's schedule a meeting for next week.",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(Path(__file__).parent / "run_llama2_7b_1bpw_l8_qat.log")),
    ],
)
logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _paths_for_dataset(calibration_dataset: str) -> tuple[Path, Path, Path]:
    ds_slug = _slug(calibration_dataset)
    if ds_slug == "c4":
        return QAT_STATE_DEFAULT, QAT_MODEL_DEFAULT, OUTPUT_FILE_DEFAULT
    base = Path(__file__).parent
    return (
        base / f"qat_state_llama2_7b_1bpw_l8_{ds_slug}.pt",
        base / f"qat_quantized_model_llama2_7b_1bpw_l8_{ds_slug}.pkl",
        base / f"results_llama2_7b_1bpw_l8_qat_{ds_slug}.json",
    )


def _save_quantized_model(model, path: Path) -> None:
    model.cpu()
    path.write_bytes(pickle.dumps(model))
    logger.info("Saved quantized model to %s (%.1f GB)", path, path.stat().st_size / 1e9)


def _build_runner(model_config: ModelConfig, mdbf: MDBF) -> Runner:
    return Runner(
        model_config=model_config,
        quantizer=mdbf,
        calibration_config=CalibrationConfig(
            calibration_dataset="wikitext2",
            num_calibration_samples=128, max_length=2048, seed=0,
        ),
        qep=True,
        qep_config=QEPConfig(percdamp=0.01, perccorr=0.5, device=DEVICE),
    )


def _load_qat_quantized_model(
    runner: Runner,
    mdbf: MDBF,
    qat_state: Path,
    device: str = DEVICE,
):
    from onecomp_globalptq.global_ptq._core.helpers import detect_quantization_method
    from onecomp_globalptq.global_ptq._core.mdbf_adapter import load_mdbf_state

    model, _ = runner.create_quantized_model(
        pack_weights=True, quantizer=mdbf, use_gemlite=False,
    )
    model.to(device)
    model.eval()
    _, mods = detect_quantization_method(model)
    state = torch.load(qat_state, map_location="cpu", weights_only=False)
    load_mdbf_state(mods, state)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-qat", action="store_true", help="Load saved QAT state and evaluate only.")
    parser.add_argument(
        "--export-model-only",
        action="store_true",
        help="Load QAT state and save full quantized model pickle (skip training/eval).",
    )
    parser.add_argument("--num-samples", type=int, default=512, help="Calibration samples (default 512).")
    parser.add_argument("--max-length", type=int, default=1024, help="Sequence length (default 1024).")
    parser.add_argument("--epochs", type=int, default=10, help="QAT epochs (default 10).")
    parser.add_argument("--dbf-lr", type=float, default=5e-5, help="Amplitude LR (default 5e-5).")
    parser.add_argument("--no-cpu-offload", action="store_true", help="Disable DeepSpeed CPU optimizer offload.")
    parser.add_argument(
        "--calibration-dataset",
        default="c4",
        help='Calibration dataset: "c4", "wikitext2", HF Hub ID, or local path (default: c4).',
    )
    args = parser.parse_args()

    qat_state, qat_model, output_file = _paths_for_dataset(args.calibration_dataset)

    print("=" * 80)
    print(f"Llama2-7B  |  MDBF {TARGET_BITS}bpw  |  l={L}  |  QAT (pure NTP + optimize_binary)")
    print(f"  epochs={args.epochs}, num_samples={args.num_samples}, max_length={args.max_length}")
    print(f"  calibration_dataset={args.calibration_dataset}")
    print(f"  dbf_lr={args.dbf_lr}, device={DEVICE}")
    print("=" * 80)

    model_config = ModelConfig(model_id=MODEL_PATH, device=DEVICE)
    exclude_kws = build_exclude_keywords(model_config)

    mdbf = MDBF(
        target_bits=TARGET_BITS,
        l=L, P=P,
        svd_mode="svd",
        use_admm=True,
        admm_iters=1000, admm_inner_iters=3, admm_reg=0.03,
        use_gradient_refine=True,
        gradient_iters=1500, gradient_lr=0.01,
        activation_aware=True, act_init="osvd",
        exclude_layer_keywords=exclude_kws,
    )
    runner = _build_runner(model_config, mdbf)

    logger.info("Loading MDBF PTQ from %s ...", MDBF_PICKLE)
    mdbf.results = pickle.loads(MDBF_PICKLE.read_bytes())

    if args.export_model_only:
        if not qat_state.exists():
            raise FileNotFoundError(f"QAT state not found: {qat_state}")
        logger.info("Exporting quantized model from %s ...", qat_state)
        model = _load_qat_quantized_model(runner, mdbf, qat_state, device="cpu")
        _save_quantized_model(model, qat_model)
        print(f"Saved: {qat_model}")
        return 0

    runner.quantized_model, _ = runner.create_quantized_model(
        pack_weights=True, quantizer=mdbf, use_gemlite=False,
    )
    runner.quantized_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    qat_elapsed = 0.0

    if args.skip_qat and qat_state.exists():
        logger.info("Loading QAT state from %s ...", qat_state)
        from onecomp_globalptq.global_ptq._core.helpers import detect_quantization_method
        from onecomp_globalptq.global_ptq._core.mdbf_adapter import load_mdbf_state
        state = torch.load(qat_state, map_location="cpu", weights_only=False)
        _, mods = detect_quantization_method(runner.quantized_model)
        load_mdbf_state(mods, state)
    else:
        ds_config = None if args.no_cpu_offload else str(DS_CONFIG)

        qat = GlobalPTQDistributed(
            # === Core QAT settings ===
            w_distill=0.0,        # no KL loss
            w_ntp=1.0,            # pure NTP loss (against ground-truth tokens)
            optimize_binary=True, # also optimize binary signs via STE
            # =========================
            dbf_lr=args.dbf_lr,
            epochs=args.epochs,
            calibration_dataset=args.calibration_dataset,
            num_calibration_samples=args.num_samples,
            max_length=args.max_length,
            calibration_strategy="drop_rand",
            calibration_seed=42,
            use_gradient_checkpointing=True,
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            deepspeed_config=ds_config,
            output_dir=str(Path(__file__).parent / "qat_output_llama2_7b"),
            logging_steps=5,
            eval_interval=2,
        )

        logger.info("Starting QAT (w_distill=0, w_ntp=1, optimize_binary=True) ...")
        t0 = time.time()
        qat.run(runner.quantized_model, model_config)
        qat_elapsed = time.time() - t0
        logger.info("QAT done in %.0fs (%.1fh)", qat_elapsed, qat_elapsed / 3600)

        # Save the MDBF state after QAT
        from onecomp_globalptq.global_ptq._core.helpers import detect_quantization_method
        from onecomp_globalptq.global_ptq._core.mdbf_adapter import save_mdbf_state
        _, mods = detect_quantization_method(runner.quantized_model)
        torch.save(save_mdbf_state(mods), qat_state)
        logger.info("Saved QAT state to %s", qat_state)

    # Evaluate on a fresh model after DeepSpeed training (same pattern as the
    # binopt/ntp scripts).
    logger.info("Creating fresh quantized model for evaluation ...")
    runner.quantized_model = None
    gc.collect()
    torch.cuda.empty_cache()

    fresh_model = _load_qat_quantized_model(runner, mdbf, qat_state)
    runner.quantized_model = fresh_model

    if not args.skip_qat or not qat_model.exists():
        _save_quantized_model(fresh_model, qat_model)

    logger.info("Evaluating perplexity (wikitext-2) ...")
    _, _, qat_ppl = runner.calculate_perplexity(
        original_model=False, dequantized_model=False, quantized_model=True,
        dataset_name="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1",
    )
    logger.info("QAT PPL = %.4f", qat_ppl)

    logger.info("Evaluating accuracy (arc_easy, piqa) ...")
    _, _, qat_acc = runner.calculate_accuracy(
        original_model=False, dequantized_model=False, quantized_model=True,
        tasks=["arc_easy", "piqa"], num_fewshot=0,
    )

    # Generated text (greedy + sampling_rp)
    runner.quantized_model.to(DEVICE)
    runner.quantized_model.eval()
    tokenizer = model_config.load_tokenizer()

    generations = {}
    for cfg_name, gen_kwargs in DECODE_CONFIGS.items():
        outputs = []
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            inputs = add_model_specific_inputs(inputs, runner.quantized_model)
            with torch.no_grad():
                ids = runner.quantized_model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
            outputs.append({"prompt": prompt, "generated": text})
            print(f"[{cfg_name}] {prompt[:40]}\n  -> {text[:200]}\n")
        generations[cfg_name] = outputs

    # For comparison against the reference (GlobalPTQ KL-only) result
    ref_gp_ppl = None
    ref_gp_arc = None
    if REF_GP_RESULTS.exists():
        ref = json.loads(REF_GP_RESULTS.read_text())
        gp = ref.get("global_ptq") or {}
        ref_gp_ppl = gp.get("ppl_wikitext2")
        ref_gp_arc = (gp.get("acc") or {}).get("arc_easy", {}).get("acc,none")

    result = {
        "model": MODEL_PATH,
        "target_bits": TARGET_BITS,
        "l": L,
        "method": "QAT (w_ntp=1.0, w_distill=0.0, optimize_binary=True)",
        "qat_settings": {
            "w_distill": 0.0,
            "w_ntp": 1.0,
            "optimize_binary": True,
            "epochs": args.epochs,
            "num_calibration_samples": args.num_samples,
            "max_length": args.max_length,
            "dbf_lr": args.dbf_lr,
            "calibration_dataset": args.calibration_dataset,
            "calibration_strategy": "drop_rand",
            "deepspeed_cpu_offload": not args.no_cpu_offload,
        },
        "qat_model_pickle": str(qat_model),
        "qat": {
            "ppl_wikitext2": qat_ppl,
            "acc": qat_acc,
            "generations": generations,
            "elapsed_sec": round(qat_elapsed, 1),
        },
        "ref_globalptq_kl_only": {
            "ppl_wikitext2": ref_gp_ppl,
            "arc_easy": ref_gp_arc,
        },
    }
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("[SUMMARY] Llama2-7B 1bpw l=8 QAT")
    print(f"  QAT PPL           : {qat_ppl:.4f}")
    if ref_gp_ppl:
        print(f"  (ref) GlobalPTQ   : {ref_gp_ppl:.4f}  (improvement: {ref_gp_ppl - qat_ppl:+.4f})")
    arc = (qat_acc.get("arc_easy") or {}).get("acc,none")
    if arc:
        print(f"  arc_easy          : {arc:.4f}")
        if ref_gp_arc:
            print(f"  (ref) arc_easy    : {ref_gp_arc:.4f}")
    print(f"  QAT time          : {qat_elapsed:.0f}s ({qat_elapsed/3600:.1f}h)")
    print(f"  Results           : {output_file}")
    if qat_model.exists():
        print(f"  Model pickle      : {qat_model} ({qat_model.stat().st_size / 1e9:.1f} GB)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
