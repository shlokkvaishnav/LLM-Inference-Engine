"""
Baseline benchmark: naive HF generate() vs static-batched HF generate() vs
mini-vLLM (continuous batching + paged KV-cache) — Milestone 7.

Three systems, same prompts, same model, same hardware:
  naive:     transformers.generate() called once per prompt, sequentially —
             the worst case: no batching at all, GPU idle between requests.
  batched:   transformers.generate() called ONCE with all prompts as one
             left-padded batch — HF's own batching, not our M2 logic.
  mini-vLLM: LLMEngine (+ PagedLlamaRunner on CUDA/Llama, else ModelRunner)
             processes all prompts via continuous batching: admits up to
             max_batch_size at once, retires and admits more as they finish.

Real vLLM comparison is opt-in (--compare-vllm) and best-effort: vllm
typically pins its own torch/transformers versions, which have repeatedly
conflicted with the transformers==4.46.3 pin this project uses on Kaggle
(see notebooks/kaggle_gpu_benchmarks.ipynb's tokenizers/huggingface_hub
version-conflict comments — the same class of problem). If vllm isn't
importable or errors, this script skips it and reports what it could
measure, rather than risking the rest of the pinned environment.

Metrics per system: wall-clock time for the whole prompt set, throughput
(tokens/sec across all prompts), peak GPU memory (if CUDA).

Usage:
    python benchmarks/baseline_hf.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --num-prompts 8 \\
        --output-len 32 \\
        --device cuda

Writes a Markdown table to stdout and benchmarks/results/baseline_hf.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.engine.sequence import SamplingParams, Sequence
from mini_vllm.kv_cache.block_manager import BlockManager
from mini_vllm.model.loader import ModelConfig, load_model
from mini_vllm.model.paged_llama_runner import PagedLlamaRunner
from mini_vllm.model.runner import ModelRunner

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "def fibonacci(n):",
    "The transformer architecture was introduced in",
    "In machine learning, attention mechanisms",
    "The history of the Roman Empire begins",
    "To bake a chocolate cake, you need",
    "Climate change is caused primarily by",
]


def _peak_memory_mb(device: str) -> float | None:
    return torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None


def _reset_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench_naive(model, tokenizer, prompts, output_len, device) -> dict:
    """Sequential: one transformers.generate() call per prompt, no batching."""
    import time

    _reset_memory(device)
    total_tokens = 0
    _sync(device)
    start = time.perf_counter()
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=output_len, max_length=None,
                do_sample=False, temperature=1.0, use_cache=True,
            )
        total_tokens += out.shape[1] - input_ids.shape[1]
    _sync(device)
    elapsed = time.perf_counter() - start
    return {"elapsed_s": elapsed, "throughput_tok_s": total_tokens / elapsed, "peak_mem_mb": _peak_memory_mb(device)}


def bench_hf_batched(model, tokenizer, prompts, output_len, device) -> dict:
    """One transformers.generate() call, all prompts left-padded into a single batch."""
    import time

    _reset_memory(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=output_len, max_length=None,
            do_sample=False, temperature=1.0, use_cache=True,
        )
    _sync(device)
    elapsed = time.perf_counter() - start
    total_tokens = (out.shape[1] - enc["input_ids"].shape[1]) * len(prompts)
    return {"elapsed_s": elapsed, "throughput_tok_s": total_tokens / elapsed, "peak_mem_mb": _peak_memory_mb(device)}


def bench_mini_vllm(model, tokenizer, prompts, output_len, device, config, max_batch_size) -> dict:
    """mini-vLLM: continuous batching (+ paged KV-cache on CUDA/Llama)."""
    import time

    _reset_memory(device)

    if device == "cuda" and getattr(model.config, "model_type", "") == "llama":
        block_manager = BlockManager(num_blocks=512, block_size=16)
        runner = PagedLlamaRunner(model, tokenizer, block_manager, dtype=torch.float16, device=device)
    else:
        block_manager = None
        runner = ModelRunner(model, tokenizer, config)

    engine = LLMEngine(runner, max_batch_size=max_batch_size, block_manager=block_manager)
    seqs = [
        Sequence(tokenizer.encode(p), SamplingParams(temperature=0.0, max_tokens=output_len))
        for p in prompts
    ]
    for seq in seqs:
        engine.add_request(seq)

    _sync(device)
    start = time.perf_counter()
    engine.run_until_done()
    _sync(device)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(s.output_token_ids) for s in seqs)
    return {"elapsed_s": elapsed, "throughput_tok_s": total_tokens / elapsed, "peak_mem_mb": _peak_memory_mb(device)}


def bench_vllm(model_name, prompts, output_len, device) -> dict | None:
    """
    Best-effort real-vLLM comparison — see module docstring for why this is
    opt-in and allowed to fail non-fatally.
    """
    import time

    try:
        from vllm import LLM
        from vllm import SamplingParams as VLLMSamplingParams
    except ImportError:
        print(
            "vllm not installed — skipping real-vLLM comparison (it typically "
            "pins conflicting torch/transformers versions in this environment; "
            "see module docstring)."
        )
        return None

    try:
        llm = LLM(model=model_name, dtype="float16" if device == "cuda" else "float32")
        sp = VLLMSamplingParams(temperature=0.0, max_tokens=output_len)
        _sync(device)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        _sync(device)
        elapsed = time.perf_counter() - start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return {"elapsed_s": elapsed, "throughput_tok_s": total_tokens / elapsed, "peak_mem_mb": _peak_memory_mb(device)}
    except Exception as e:
        print(f"vllm comparison failed at runtime, skipping: {e}")
        return None


def _fmt_row(name: str, result: dict | None) -> str:
    if result is None:
        return f"| {name} | — | — | — | not available |"
    mem = f"{result['peak_mem_mb']:.0f} MB" if result["peak_mem_mb"] is not None else "n/a"
    return f"| {name} | {result['throughput_tok_s']:.1f} | {result['elapsed_s']:.2f} | {mem} | |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--compare-vllm", action="store_true")
    parser.add_argument("--output", default="benchmarks/results/baseline_hf.md")
    args = parser.parse_args()

    prompts = (DEFAULT_PROMPTS * ((args.num_prompts // len(DEFAULT_PROMPTS)) + 1))[: args.num_prompts]
    dtype = "float16" if args.device == "cuda" else "float32"

    config = ModelConfig(model_name_or_path=args.model, dtype=dtype, device=args.device, max_model_len=512)
    model, tokenizer = load_model(config)
    model.eval()

    print(f"model={args.model} device={args.device} num_prompts={len(prompts)} output_len={args.output_len} "
          f"max_batch_size={args.max_batch_size}")

    naive = bench_naive(model, tokenizer, prompts, args.output_len, args.device)
    batched = bench_hf_batched(model, tokenizer, prompts, args.output_len, args.device)
    mini = bench_mini_vllm(model, tokenizer, prompts, args.output_len, args.device, config, args.max_batch_size)
    vllm_result = bench_vllm(args.model, prompts, args.output_len, args.device) if args.compare_vllm else None

    lines = [
        "| System | Throughput (tok/s) | Wall-clock (s) | Peak GPU mem | Notes |",
        "|--------|--------------------|----------------|--------------|-------|",
        _fmt_row("HF `generate()` — naive", naive),
        _fmt_row("HF `generate()` — batched", batched),
        _fmt_row("**mini-vLLM**", mini),
        _fmt_row("vLLM (ceiling)", vllm_result),
    ]
    table = "\n".join(lines)
    print()
    print(table)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Baseline benchmark — {args.model} on {args.device}\n\n"
        f"num_prompts={len(prompts)}  output_len={args.output_len}  max_batch_size={args.max_batch_size}\n\n"
        + table + "\n"
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
