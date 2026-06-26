#!/usr/bin/env python3
"""Run MDBF with paper-aligned settings on TinyLlama via OneComp (feature/mdbf).

Settings mirror qep-dev/experiments/lowbit_formats/conf/mdbf_paper_tinyllama_onecomp.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

from onecomp import CalibrationConfig, ModelConfig, QEPConfig, Runner
from onecomp.quantizer.mdbf import MDBF

CONFIG_PATH = Path(__file__).resolve().parents[2] / (
    "qep-dev/experiments/lowbit_formats/conf/mdbf_paper_tinyllama_onecomp.yaml"
)


def _build_exclude_layer_keywords(num_layers: int, first_n: int, last_n: int) -> list[str]:
    skip = set(range(first_n)) | set(range(num_layers - last_n, num_layers))
    return [f"model.layers.{i}." for i in sorted(skip)]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(CONFIG_PATH)

    model_path = cfg["model_path"]
    calib = cfg["calibration"]
    mdbf_cfg = cfg["mdbf"]
    strategy = cfg["strategy"]
    eval_cfg = cfg["evaluation"]

    model_config = ModelConfig(path=model_path, device=cfg.get("device", "cuda:0"))

    num_layers = model_config.load_config().num_hidden_layers
    exclude_keywords = _build_exclude_layer_keywords(
        num_layers=num_layers,
        first_n=strategy["cut_first_layers"],
        last_n=strategy["cut_last_layers"],
    )

    quantizer = MDBF(
        target_bits=mdbf_cfg["target_bits"],
        l=mdbf_cfg["l"],
        P=mdbf_cfg["P"],
        svd_mode=mdbf_cfg["svd_mode"],
        use_admm=mdbf_cfg["use_admm"],
        admm_iters=mdbf_cfg["admm_iters"],
        admm_inner_iters=mdbf_cfg["admm_inner_iters"],
        admm_reg=mdbf_cfg["admm_reg"],
        use_gradient_refine=mdbf_cfg["use_gradient_refine"],
        gradient_iters=mdbf_cfg["gradient_iters"],
        gradient_lr=mdbf_cfg["gradient_lr"],
        activation_aware=mdbf_cfg["activation_aware"],
        act_init=mdbf_cfg["act_init"],
        exclude_layer_keywords=exclude_keywords,
    )

    calibration_config = CalibrationConfig(
        calibration_dataset=calib["dataset"],
        num_calibration_samples=calib["num_samples"],
        max_length=calib["max_length"],
        seed=calib["seed"],
    )

    qep_config = None
    if cfg["qep"]["enabled"]:
        qep_config = QEPConfig(
            percdamp=cfg["qep"]["percdamp"],
            perccorr=cfg["qep"]["perccorr"],
            device=cfg.get("device", "cuda:0"),
        )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration_config,
        qep=cfg["qep"]["enabled"],
        qep_config=qep_config,
    )

    print("=" * 80)
    print("OneComp MDBF (feature/mdbf) - paper settings")
    print(f"Model: {model_path}")
    print(f"MDBF: l={mdbf_cfg['l']}, P={mdbf_cfg['P']}, bits={mdbf_cfg['target_bits']}")
    print(f"ADMM: {mdbf_cfg['admm_iters']} x inner {mdbf_cfg['admm_inner_iters']}")
    print(f"Gradient refine: {mdbf_cfg['gradient_iters']} steps, lr={mdbf_cfg['gradient_lr']}")
    print(f"Skip layers: first {strategy['cut_first_layers']}, last {strategy['cut_last_layers']}")
    print("=" * 80)

    runner.run()

    _, dequant_ppl, _ = runner.calculate_perplexity(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
        dataset_name=eval_cfg["ppl_dataset"],
        dataset_config=eval_cfg["ppl_config"],
    )
    _, dequant_acc, _ = runner.calculate_accuracy(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
        tasks=eval_cfg["tasks"],
        num_fewshot=eval_cfg["num_fewshot"],
    )

    print("\n" + "=" * 80)
    print("[Final Summary]")
    print(f"PPL (wikitext2, dequantized): {dequant_ppl}")
    print(f"ACC ({', '.join(eval_cfg['tasks'])}, dequantized): {dequant_acc}")
    print("=" * 80)

    save_dir = cfg.get("save_dir")
    if save_dir:
        runner.save_quantized_model(save_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
