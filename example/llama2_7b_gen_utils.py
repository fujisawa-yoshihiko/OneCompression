"""Shared helpers for MDBF experiments with text generation."""
from __future__ import annotations

import torch

from onecomp.utils.model_inputs import add_model_specific_inputs

GENERATION_PROMPTS = [
    "The future of artificial intelligence",
    "Hello, how are you today?",
    "Write a bubble sort algorithm in Python.",
    "Let's schedule a meeting for next week.",
]

GENERATION_KWARGS = {
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": 1.0,
}


def get_num_hidden_layers(model_config) -> int:
    """Return layer count (handles Gemma 4 text_config)."""
    cfg = model_config.load_config()
    if hasattr(cfg, "get_text_config"):
        return cfg.get_text_config().num_hidden_layers
    if getattr(cfg, "text_config", None) is not None:
        return cfg.text_config.num_hidden_layers
    return cfg.num_hidden_layers


def get_layer_prefix(model_config) -> str:
    """Return module prefix for exclude_layer_keywords."""
    cfg = model_config.load_config()
    model_type = getattr(cfg, "model_type", "")
    if model_type == "gemma4" or hasattr(cfg, "text_config"):
        return "model.language_model.layers."
    return "model.layers."


def build_exclude_keywords(model_config, first_n: int = 4, last_n: int = 4) -> list[str]:
    num_layers = get_num_hidden_layers(model_config)
    prefix = get_layer_prefix(model_config)
    skip = set(range(first_n)) | set(range(num_layers - last_n, num_layers))
    return [f"{prefix}{i}." for i in sorted(skip)]


def build_gemma4_ablation_exclude_keywords(model_config, first_n: int = 4, last_n: int = 4) -> list[str]:
    """Exclude first/last blocks and all per_layer_* pathways (keep FP16)."""
    kws = build_exclude_keywords(model_config, first_n=first_n, last_n=last_n)
    kws.extend([
        "per_layer_input_gate",
        "per_layer_projection",
        "per_layer_model_projection",
    ])
    return kws


def run_text_generation(runner, device: str) -> list[dict]:
    """Generate text with dequantized weights after quantization."""
    model = runner.model_config.load_model()
    tokenizer = runner.model_config.load_tokenizer()
    runner.update_model_weights(model, quantizer=runner.quantizer)
    model.eval()
    model.to(device)

    outputs: list[dict] = []
    for prompt in GENERATION_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs = add_model_specific_inputs(inputs, model)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **GENERATION_KWARGS)
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append({"prompt": prompt, "generated": generated})
        print(f"\n--- Generation ---\nPrompt: {prompt}\nOutput: {generated}\n")

    del model, tokenizer
    torch.cuda.empty_cache()
    return outputs
