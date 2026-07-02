"""
Correctness tests: our engine must match HuggingFace token-for-token.
All tests run on CPU — no GPU required.

These are the ground-truth checks before every optimization.
A fast wrong answer is worse than a slow correct one.
"""
import os
import pytest
import torch

from mini_vllm.engine.sequence import Sequence, SamplingParams, SequenceStatus
from mini_vllm.model.loader import ModelConfig, load_model
from mini_vllm.model.runner import ModelRunner

# Override via env vars on Kaggle:
#   MINI_VLLM_TEST_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   MINI_VLLM_TEST_DEVICE=cuda
MODEL  = os.environ.get("MINI_VLLM_TEST_MODEL",  "gpt2")
DEVICE = os.environ.get("MINI_VLLM_TEST_DEVICE", "cpu")
DTYPE  = "float16" if DEVICE == "cuda" else "float32"

# Fixed prompts used consistently across all correctness checks.
FIXED_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "def fibonacci(n):",
    "The transformer architecture was introduced in",
]
MAX_NEW_TOKENS = 5   # 5 tokens is enough to catch any mismatch; keeps CPU tests fast


@pytest.fixture(scope="module")
def runner():
    """Load model once. GPT-2 on CPU locally; TinyLlama on GPU via env vars."""
    config = ModelConfig(
        model_name_or_path=MODEL,
        dtype=DTYPE,
        device=DEVICE,
        max_model_len=512,
    )
    model, tokenizer = load_model(config)
    return ModelRunner(model, tokenizer, config)


@pytest.fixture(scope="module")
def hf_model(runner):
    """Reuse the already-loaded model for HuggingFace comparison."""
    return runner.model, runner.tokenizer


def _hf_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    """Reference: what HuggingFace produces with greedy decoding."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            max_length=None,          # suppress conflict with model's default max_length
            do_sample=False,          # greedy
            temperature=1.0,
            use_cache=True,
        )
    # strip prompt tokens; return only the generated part
    return out[0, input_ids.shape[1]:].tolist()


def _our_generate(runner: ModelRunner, prompt: str, max_new_tokens: int) -> list[int]:
    """Our engine: greedy decoding (temperature=0)."""
    token_ids = runner.tokenizer.encode(prompt)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    seq = Sequence(token_ids, params)
    runner.generate([seq])
    return seq.output_token_ids


def test_single_sequence_matches_hf(runner, hf_model):
    """
    Our decode loop produces identical token IDs to transformers.generate()
    for each prompt in FIXED_PROMPTS with greedy (temperature=0) decoding.
    """
    model, tokenizer = hf_model

    for prompt in FIXED_PROMPTS:
        expected = _hf_generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        actual = _our_generate(runner, prompt, MAX_NEW_TOKENS)

        assert actual == expected, (
            f"Mismatch on prompt: {prompt!r}\n"
            f"  HF:  {expected}\n"
            f"  Ours:{actual}\n"
            f"  HF decoded:  {tokenizer.decode(expected)!r}\n"
            f"  Ours decoded:{tokenizer.decode(actual)!r}"
        )


def test_batched_sequences_match_single(runner, hf_model):
    """
    Each sequence in a static batch produces the same tokens as its
    single-sequence run. Proves left-padding and attention masking are correct:
    batching must never change what a sequence generates.
    """
    _, tokenizer = hf_model

    # Run all prompts together as one batch.
    batch_seqs = [
        Sequence(tokenizer.encode(p), SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
        for p in FIXED_PROMPTS
    ]
    runner.generate(batch_seqs)
    batch_outputs = [seq.output_token_ids[:] for seq in batch_seqs]

    # Run each prompt alone and compare.
    for i, prompt in enumerate(FIXED_PROMPTS):
        single_seq = Sequence(
            tokenizer.encode(prompt),
            SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS),
        )
        runner.generate([single_seq])

        assert batch_outputs[i] == single_seq.output_token_ids, (
            f"Batch/single mismatch on prompt {i!r}: {prompt!r}\n"
            f"  batched: {batch_outputs[i]}\n"
            f"  single:  {single_seq.output_token_ids}\n"
            f"  batched decoded: {tokenizer.decode(batch_outputs[i])!r}\n"
            f"  single decoded:  {tokenizer.decode(single_seq.output_token_ids)!r}"
        )


def test_continuous_batch_matches_single(runner, hf_model):
    """
    Every sequence driven through the continuous batching engine produces
    identical tokens to the single-sequence HuggingFace baseline.

    Why this matters: M3 stores and restores per-sequence KV caches and
    admits sequences at different times. A bug in _to_tuple_kv / _slice_kv
    or in the decode_one position encoding would cause divergence here.

    max_batch_size=2 with 4 prompts means the engine will admit the first 2,
    then as each finishes, admit the next — exercising the continuous admit
    path (not just static batching with a scheduler veneer).
    """
    from mini_vllm.engine.llm_engine import LLMEngine

    model, tokenizer = hf_model

    # Build the sequences to run through our engine.
    engine_seqs = [
        Sequence(
            tokenizer.encode(p),
            SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS),
        )
        for p in FIXED_PROMPTS
    ]

    engine = LLMEngine(runner, max_batch_size=2)
    for seq in engine_seqs:
        engine.add_request(seq)
    engine.run_until_done()

    # Compare every engine output against HuggingFace greedy baseline.
    for seq, prompt in zip(engine_seqs, FIXED_PROMPTS):
        expected = _hf_generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        assert seq.output_token_ids == expected, (
            f"M3 engine mismatch on {prompt!r}\n"
            f"  HF:     {expected}\n"
            f"  engine: {seq.output_token_ids}\n"
            f"  HF decoded:     {tokenizer.decode(expected)!r}\n"
            f"  engine decoded: {tokenizer.decode(seq.output_token_ids)!r}"
        )
