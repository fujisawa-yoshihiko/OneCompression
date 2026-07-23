#!/usr/bin/env python3
"""Llama2-7B: MDBF paper-aligned protocol (Table 1 PTQ / Table 2 QAT).

Paper settings (mdbf_paper.pdf):
  - Layer exclusion: first/last 4 blocks FP16
  - PTQ calibration: C4, 512 samples, QEP alpha=0.5
  - PPL: average over WikiText-2, C4, PTB
  - ACC: average over 6 tasks (boolq, piqa, hellaswag, winogrande, arc_easy, arc_challenge)
  - Table 1: layer-wise PTQ only, l=8 at 1.00 BPW
  - Table 2: post-QAT with FP16 teacher KL (LittleBit protocol), l=3 at 1.00 BPW

Usage:
  # Table 1: PTQ only (paper Table 1 row)
  CUDA_VISIBLE_DEVICES=1 python run_llama2_7b_paper_protocol.py --table 1

  # Table 2: PTQ + teacher KL QAT (paper Table 2 row)
  CUDA_VISIBLE_DEVICES=2 python run_llama2_7b_paper_protocol.py --table 2

  # Skip PTQ if pickle exists
  python run_llama2_7b_paper_protocol.py --table 1 --skip-ptq
  python run_llama2_7b_paper_protocol.py --table 2 --skip-ptq --skip-qat
"""
from __future__ import annotations

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
from onecomp.utils.accuracy import calculate_accuracy
from onecomp.utils.perplexity import calculate_perplexity
from onecomp_globalptq import GlobalPTQ, GlobalPTQDistributed

from llama2_7b_gen_utils import build_exclude_keywords

MODEL_PATH = "/data3/yoshida/qep-dev/models/Llama-2-7b-hf"
DEVICE = "cuda:0"
TARGET_BITS = 1.0
P = 1
ADMM_ITERS = 1000
GRAD_ITERS = 1500

PAPER_TASKS = [
    "boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge",
]

# (dataset_name, dataset_config, split, max_samples for PPL eval)
# C4: use a single validation shard to avoid downloading the full split (~73GB).
PPL_EVAL_SPECS = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", None),
    ("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", "train", 512),
    ("ptb-text-only/ptb_text_only", "penn_treebank", "test", None),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _artifact_paths(
    table: int, l: int, run_tag: str | None = None,
) -> tuple[Path, Path, Path]:
    base = Path(__file__).parent
    tag = f"paper_t{table}_l{l}"
    # PTQ pickle is shared across QAT sweeps; results/state get a suffix.
    suffix = f"_{run_tag}" if run_tag else ""
    return (
        base / f"mdbf_{tag}_llama2_7b_1bpw.pkl",
        base / f"results_llama2_7b_1bpw_{tag}{suffix}.json",
        base / f"qat_state_{tag}{suffix}.pt",
    )


def _paper_avg_ppl(model, tokenizer) -> tuple[float, dict[str, float]]:
    per_ds: dict[str, float] = {}
    for ds_name, ds_cfg, split, max_samples in PPL_EVAL_SPECS:
        key = ds_name.split("/")[-1]
        logger.info("PPL eval: %s (%s/%s) ...", key, ds_name, ds_cfg)
        ppl = calculate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_name=ds_name,
            dataset_config=ds_cfg,
            split=split,
            max_samples=max_samples,
        )
        per_ds[key] = ppl
        logger.info("  %s PPL = %.4f", key, ppl)
    avg = sum(per_ds.values()) / len(per_ds)
    return avg, per_ds


def _paper_avg_acc(model, tokenizer) -> tuple[float, dict]:
    logger.info("ACC eval: 6 tasks %s ...", PAPER_TASKS)
    results = calculate_accuracy(
        model=model,
        tokenizer=tokenizer,
        tasks=PAPER_TASKS,
        num_fewshot=0,
        display_results=True,
    )
    scores = []
    for task in PAPER_TASKS:
        m = results.get(task) or {}
        v = m.get("acc,none")
        if v is None:
            v = m.get("acc_norm,none")
        if v is not None:
            scores.append(float(v))
    avg = sum(scores) / len(scores) if scores else float("nan")
    return avg, results


def _build_mdbf(l: int, exclude_kws: list[str]) -> MDBF:
    return MDBF(
        target_bits=TARGET_BITS,
        l=l,
        P=P,
        svd_mode="svd",
        use_admm=True,
        admm_iters=ADMM_ITERS,
        admm_inner_iters=3,
        admm_reg=0.03,
        use_gradient_refine=True,
        gradient_iters=GRAD_ITERS,
        gradient_lr=0.01,
        activation_aware=True,
        act_init="osvd",
        exclude_layer_keywords=exclude_kws,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=int, choices=[1, 2], required=True,
                        help="1=PTQ only (Table 1), 2=PTQ+QAT (Table 2)")
    parser.add_argument("--skip-ptq", action="store_true")
    parser.add_argument("--skip-qat", action="store_true", help="Table 2 only")
    parser.add_argument("--skip-ptq-eval", action="store_true",
                        help="Skip PTQ PPL/ACC eval (use with --ptq-eval-cache)")
    parser.add_argument("--ptq-eval-cache", type=Path, default=None,
                        help="JSON with PTQ eval metrics when --skip-ptq-eval")
    parser.add_argument("--teacher-cpu", action="store_true",
                        help="Keep FP16 teacher on CPU (single GPU)")
    parser.add_argument("--teacher-device", default=None,
                        help="Teacher device (e.g. cuda:1 for 2-GPU QAT)")
    parser.add_argument("--distributed-qat", action="store_true",
                        help="Use GlobalPTQDistributed + DeepSpeed (required for optimize_binary on 7B)")
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Local path or HF model id (default: Llama-2-7b-hf on /data3/...)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dbf-lr", type=float, default=5e-5)
    parser.add_argument(
        "--mdbf-ste-k", type=float, default=2.0,
        help="MDBF binary STE sharpness k (default 2.0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="KL distillation temperature (default 1.0)",
    )
    parser.add_argument(
        "--run-tag", default=None,
        help="Suffix for QAT results/state files (keeps PTQ pickle shared)",
    )
    parser.add_argument(
        "--qat-eval-wt2-only", action="store_true",
        help="After QAT, evaluate WikiText-2 PPL only (skip 3ds PPL + ACC)",
    )
    args = parser.parse_args()

    model_path = args.model_path
    l = 8 if args.table == 1 else 3
    run_tag = args.run_tag
    if run_tag is None and args.table == 2 and (
        args.epochs != 5
        or abs(args.dbf_lr - 5e-5) > 0
        or abs(args.mdbf_ste_k - 2.0) > 0
        or abs(args.temperature - 1.0) > 0
    ):
        # Auto-tag sweeps so they do not overwrite the paper baseline artifacts.
        lr_s = f"{args.dbf_lr:.0e}".replace("+", "").replace(".", "p")
        run_tag = (
            f"ep{args.epochs}_lr{lr_s}_k{args.mdbf_ste_k:g}_t{args.temperature:g}"
        )
    mdbf_pickle, output_file, qat_state = _artifact_paths(args.table, l, run_tag)

    print("=" * 80)
    print(f"Llama2-7B paper protocol | Table {args.table} | l={l} | {TARGET_BITS}bpw")
    print(f"  Model: {model_path}")
    print(f"  PTQ calib: C4 512 | QEP alpha=0.5 | device={DEVICE}")
    if args.table == 2:
        print(
            f"  QAT: KL + optimize_binary | epochs={args.epochs} "
            f"lr={args.dbf_lr} mdbf_ste_k={args.mdbf_ste_k} T={args.temperature}"
        )
        if run_tag:
            print(f"  run_tag: {run_tag}")
        if args.teacher_device:
            print(f"  Teacher device: {args.teacher_device}")
    print("=" * 80)

    model_config = ModelConfig(model_id=model_path, device=DEVICE)
    exclude_kws = build_exclude_keywords(model_config)
    mdbf = _build_mdbf(l, exclude_kws)

    runner = Runner(
        model_config=model_config,
        quantizer=mdbf,
        calibration_config=CalibrationConfig(
            calibration_dataset="c4",
            num_calibration_samples=512,
            max_length=2048,
            seed=0,
        ),
        qep=True,
        qep_config=QEPConfig(percdamp=0.01, perccorr=0.5, device=DEVICE),
    )

    ptq_elapsed = 0.0
    if args.skip_ptq and mdbf_pickle.exists():
        logger.info("Loading PTQ pickle %s", mdbf_pickle)
        mdbf.results = pickle.loads(mdbf_pickle.read_bytes())
    else:
        logger.info("Running MDBF PTQ (C4 512) ...")
        t0 = time.time()
        runner.run()
        ptq_elapsed = time.time() - t0
        mdbf_pickle.write_bytes(pickle.dumps(mdbf.results))
        logger.info("PTQ done in %.0fs (%.1fh), saved %s", ptq_elapsed, ptq_elapsed / 3600, mdbf_pickle)

    gc.collect()
    torch.cuda.empty_cache()

    runner.quantized_model, _ = runner.create_quantized_model(
        pack_weights=True, quantizer=mdbf, use_gemlite=False,
    )
    runner.quantized_model.to(DEVICE)
    runner.quantized_model.eval()
    tokenizer = model_config.load_tokenizer()

    ptq_ppl_dequant = None
    if args.skip_ptq_eval:
        cache_path = args.ptq_eval_cache or Path(__file__).parent / "paper_t2_ptq_eval_cache.json"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"--skip-ptq-eval requires cache at {cache_path}"
            )
        cached = json.loads(cache_path.read_text())
        ptq_ppl_dequant = cached["ppl_wikitext2_dequant"]
        ptq_ppl_avg = cached["ppl_avg_3datasets"]
        ptq_ppl_per_ds = cached["ppl_per_dataset"]
        ptq_acc_avg = cached["acc_avg_6tasks"]
        ptq_acc = cached["acc"]
        logger.info("Loaded PTQ eval from cache: %s", cache_path)
    else:
        logger.info("PTQ evaluation (dequantized weights) ...")
        _, ptq_ppl_dequant, _ = runner.calculate_perplexity(
            original_model=False,
            dequantized_model=True,
            quantized_model=False,
            dataset_name="Salesforce/wikitext",
            dataset_config="wikitext-2-raw-v1",
        )
        ptq_ppl_avg, ptq_ppl_per_ds = _paper_avg_ppl(runner.quantized_model, tokenizer)
        ptq_acc_avg, ptq_acc = _paper_avg_acc(runner.quantized_model, tokenizer)

    qat_block = None
    qat_elapsed = 0.0

    if args.table == 2 and not args.skip_qat:
        runner.quantized_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        teacher_dev = (
            "cpu" if args.teacher_cpu
            else (args.teacher_device or DEVICE)
        )
        student_dev = DEVICE

        if args.distributed_qat:
            ds_config = Path(__file__).parent / "ds_zero2_mdbf_offload.json"
            gp = GlobalPTQDistributed(
                w_distill=1.0,
                w_ntp=0.0,
                temperature=args.temperature,
                epochs=args.epochs,
                dbf_lr=args.dbf_lr,
                mdbf_ste_k=args.mdbf_ste_k,
                optimize_binary=True,
                num_calibration_samples=512,
                max_length=2048,
                calibration_dataset="c4",
                calibration_strategy="drop_rand",
                calibration_seed=42,
                use_gradient_checkpointing=True,
                bf16=True,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                teacher_device=teacher_dev,
                deepspeed_config=str(ds_config),
            )
        else:
            gp = GlobalPTQ(
                epochs=args.epochs,
                dbf_lr=args.dbf_lr,
                temperature=args.temperature,
                mdbf_ste_k=args.mdbf_ste_k,
                optimize_binary=True,
                num_calibration_samples=512,
                max_length=2048,
                calibration_dataset="c4",
                calibration_strategy="drop_rand",
                calibration_seed=42,
                use_gradient_checkpointing=True,
                use_mixed_precision=True,
                grad_accum_steps=4,
                student_device=student_dev,
                teacher_device=teacher_dev,
            )
        logger.info(
            "Starting paper QAT (KL, opt_binary, distributed=%s, "
            "epochs=%d, lr=%g, mdbf_ste_k=%g, T=%g) ...",
            args.distributed_qat,
            args.epochs,
            args.dbf_lr,
            args.mdbf_ste_k,
            args.temperature,
        )
        t0 = time.time()
        gp.run(runner.quantized_model, model_config)
        qat_elapsed = time.time() - t0
        logger.info("QAT done in %.0fs (%.1fh)", qat_elapsed, qat_elapsed / 3600)

        from onecomp_globalptq.global_ptq._core.helpers import detect_quantization_method
        from onecomp_globalptq.global_ptq._core.mdbf_adapter import save_mdbf_state

        _, mdbf_mods = detect_quantization_method(runner.quantized_model)
        torch.save(save_mdbf_state(mdbf_mods), qat_state)
        logger.info("Saved QAT state to %s", qat_state)

        runner.quantized_model.to(DEVICE)
        runner.quantized_model.eval()

        _, _, qat_ppl_wt2 = runner.calculate_perplexity(
            original_model=False,
            dequantized_model=False,
            quantized_model=True,
            dataset_name="Salesforce/wikitext",
            dataset_config="wikitext-2-raw-v1",
        )
        if args.qat_eval_wt2_only:
            qat_ppl_avg, qat_ppl_per_ds = float("nan"), {"wikitext": qat_ppl_wt2}
            qat_acc_avg, qat_acc = float("nan"), {}
            logger.info("Skipped 3ds PPL + ACC (--qat-eval-wt2-only)")
        else:
            qat_ppl_avg, qat_ppl_per_ds = _paper_avg_ppl(
                runner.quantized_model, tokenizer,
            )
            qat_acc_avg, qat_acc = _paper_avg_acc(
                runner.quantized_model, tokenizer,
            )

        qat_block = {
            "method": "GlobalPTQ teacher KL + optimize_binary (LittleBit-style)",
            "epochs": args.epochs,
            "dbf_lr": args.dbf_lr,
            "mdbf_ste_k": args.mdbf_ste_k,
            "temperature": args.temperature,
            "run_tag": run_tag,
            "calibration_dataset": "c4",
            "num_calibration_samples": 512,
            "optimize_binary": True,
            "teacher_device": teacher_dev,
            "ppl_wikitext2": qat_ppl_wt2,
            "ppl_avg_3datasets": qat_ppl_avg,
            "ppl_per_dataset": qat_ppl_per_ds,
            "acc_avg_6tasks": qat_acc_avg,
            "acc": qat_acc,
            "elapsed_sec": round(qat_elapsed, 1),
            "paper_table2_ref_ppl_wikitext2_1bpw": 9.36,
        }

    result = {
        "model": model_path,
        "paper_protocol": True,
        "table": args.table,
        "target_bits": TARGET_BITS,
        "l": l,
        "P": P,
        "ptq_calibration": {
            "dataset": "c4",
            "num_samples": 512,
            "max_length": 2048,
            "qep_perccorr": 0.5,
        },
        "ptq": {
            "ppl_wikitext2_dequant": ptq_ppl_dequant,
            "ppl_avg_3datasets": ptq_ppl_avg,
            "ppl_per_dataset": ptq_ppl_per_ds,
            "acc_avg_6tasks": ptq_acc_avg,
            "acc": ptq_acc,
            "elapsed_sec": round(ptq_elapsed, 1),
            "paper_table1_ref_l8_ppl_avg": 99.26,
            "paper_table1_ref_l8_acc": 0.4372,
        },
        "qat": qat_block,
    }
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print(f"[SUMMARY] Paper Table {args.table} | l={l}")
    print(f"  PTQ PPL avg (3 ds) : {ptq_ppl_avg:.4f}")
    print(f"  PTQ ACC avg (6 tk) : {ptq_acc_avg:.4f}")
    if qat_block:
        print(f"  QAT PPL wikitext2  : {qat_block['ppl_wikitext2']:.4f}  (paper ref: 9.36)")
        avg = qat_block["ppl_avg_3datasets"]
        acc = qat_block["acc_avg_6tasks"]
        if avg == avg:  # not NaN
            print(f"  QAT PPL avg (3 ds) : {avg:.4f}")
        if acc == acc:
            print(f"  QAT ACC avg (6 tk) : {acc:.4f}")
    print(f"  Saved: {output_file}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
