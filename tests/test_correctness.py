"""
Correctness tests: our engine must match HuggingFace token-for-token.
All tests run on CPU — no GPU required.

These are the ground-truth checks we use before every optimization.
A fast wrong answer is worse than a slow correct one.
"""
import pytest

# Fixed prompts used consistently across all correctness checks.
FIXED_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "def fibonacci(n):",
    "The transformer architecture was introduced in",
]
MAX_NEW_TOKENS = 20


@pytest.mark.skip(reason="Milestone 1 — not yet implemented")
def test_single_sequence_matches_hf():
    """
    Our decode loop produces identical token IDs to transformers.generate()
    for each prompt in FIXED_PROMPTS, with greedy (temperature=0) decoding.
    """
    ...


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
