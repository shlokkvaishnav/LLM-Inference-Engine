"""
M5 quantization report: size / quality / speed tradeoff for fp16 baseline
vs INT8 vs INT4 weight-only quantization, on the real production model.

Measures three independent things — don't assume one implies another:
  - size:    bytes used by the quantized target Linear layers (the actual,
             unambiguous win: less to store, less to load from disk/HBM)
  - quality: cosine similarity of next-token logits vs the fp16 baseline,
             averaged over FIXED_PROMPTS (single forward step per prompt —
             isolates quantization error from autoregressive divergence)
  - speed:   tokens/sec generating MAX_NEW_TOKENS via ModelRunner.generate()

Speed is measured, not assumed, because dequantize-on-the-fly weight-only
quantization (see mini_vllm/quantization/quantize.py) does NOT inherently
speed up the matmul — it may even be slightly SLOWER due to the extra
dequant step each forward, despite using less memory to store the weights.
A real decode speedup needs a fused low-precision kernel; this report's
speed column is the evidence for (or against) that claim on this hardware,
not a foregone conclusion.

Usage (Kaggle GPU):
    python benchmarks/quantization_report.py

Env overrides:
    MINI_VLLM_TEST_MODEL   (default TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    MINI_VLLM_TEST_DEVICE  (default cuda)
"""
from __future__ import annotations

import copy
import os
import time

import torch

from mini_vllm.engine.sequence import SamplingParams, Sequence
from mini_vllm.model.loader import ModelConfig, load_model
from mini_vllm.model.runner import ModelRunner
from mini_vllm.quantization.quantized_linear import linear_layer_bytes, quantize_model

MODEL = os.environ.get("MINI_VLLM_TEST_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = os.environ.get("MINI_VLLM_TEST_DEVICE", "cuda")
DTYPE = "float16" if DEVICE == "cuda" else "float32"

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "def fibonacci(n):",
    "The transformer architecture was introduced in",
]
MAX_NEW_TOKENS = 32


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def measure_quality(baseline_model, quant_model, tokenizer) -> float:
    """Average cosine similarity of next-token logits, one forward step per prompt."""
    sims = []
    for prompt in PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits_base = baseline_model(input_ids).logits[0, -1]
            logits_quant = quant_model(input_ids).logits[0, -1]
        sims.append(_cosine_sim(logits_base, logits_quant))
    return sum(sims) / len(sims)


def measure_speed(model, tokenizer, config) -> float:
    """Tokens/sec across all prompts, generated via the dense M1-M3 runner."""
    runner = ModelRunner(model, tokenizer, config)
    seqs = [
        Sequence(tokenizer.encode(p), SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
        for p in PROMPTS
    ]
    _sync()
    start = time.perf_counter()
    runner.generate(seqs)
    _sync()
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(s.output_token_ids) for s in seqs)
    return total_tokens / elapsed


def _row(label: str, size_mb: float, quality: str, tok_s: float) -> str:
    return f"{label:8s} {size_mb:12.1f} MB   {quality:>18s}   {tok_s:8.1f} tok/s"


def main() -> None:
    config = ModelConfig(model_name_or_path=MODEL, dtype=DTYPE, device=DEVICE, max_model_len=512)
    model, tokenizer = load_model(config)
    model.eval()

    fp16_bytes = linear_layer_bytes(model)
    fp16_speed = measure_speed(model, tokenizer, config)

    print(f"model: {MODEL}  device: {DEVICE}")
    print(f"{'':8s} {'size':>15s}   {'quality':>18s}   {'speed':>11s}")
    print(_row("fp16", fp16_bytes / 1e6, "1.0000 (baseline)", fp16_speed))

    for bits in (8, 4):
        quant_model = copy.deepcopy(model)
        quantize_model(quant_model, bits=bits)

        size_bytes = linear_layer_bytes(quant_model)
        quality = measure_quality(model, quant_model, tokenizer)
        speed = measure_speed(quant_model, tokenizer, config)

        print(_row(f"int{bits}", size_bytes / 1e6, f"{quality:.4f}", speed))

        del quant_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print()
    print("size:    bytes used by quantized target layers (q/k/v/o_proj + MLP projections)")
    print("quality: avg cosine similarity of next-token logits vs fp16, single forward step")
    print("speed:   tok/s generating", MAX_NEW_TOKENS, "tokens across", len(PROMPTS), "prompts")
    print("         (weight-only dequant-on-the-fly — NOT expected to be faster than fp16;")
    print("          see mini_vllm/quantization/quantize.py for why)")


if __name__ == "__main__":
    main()
