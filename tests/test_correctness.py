"""
Correctness tests: our engine must match HuggingFace token-for-token.
All tests run on CPU — no GPU required.

These are the ground-truth checks before every optimization.
A fast wrong answer is worse than a slow correct one.
"""
import pytest
import torch

from mini_vllm.engine.sequence import Sequence, SamplingParams, SequenceStatus
from mini_vllm.model.loader import ModelConfig, load_model
from mini_vllm.model.runner import ModelRunner

# GPT-2 for local CPU correctness tests: 124M params, loads in seconds.
# TinyLlama is the production model on Kaggle — same code, different config.
MODEL = "gpt2"

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
    """Load model once for all tests in this module."""
    config = ModelConfig(
        model_name_or_path=MODEL,
        dtype="float32",   # float32 on CPU for exact reproducibility
        device="cpu",
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
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
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


@pytest.mark.skip(reason="Milestone 2 — not yet implemented")
def test_batched_sequences_match_single():
    """
    Each sequence in a static batch produces the same tokens as its
    equivalent single-sequence run. Batching must not change outputs
    (padding / masking correctness check).
    """
    ...


@pytest.mark.skip(reason="Milestone 3 — not yet implemented")
def test_continuous_batch_matches_single():
    """
    Sequences processed via the continuous batching scheduler produce
    identical tokens to single-sequence baseline — regardless of when
    they were admitted or preempted.
    """
    ...
