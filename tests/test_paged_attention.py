"""
Paged attention correctness tests — Milestone 4c.

paged_attention_reference tests run everywhere (pure PyTorch, CPU is fine —
this is what proves the scatter-gather math is correct).

paged_attention_triton is GPU-only and skipped when CUDA/triton aren't
available; verified on Kaggle against the reference implementation.
"""
import pytest
import torch

from mini_vllm.kv_cache.paged_attention import (
    paged_attention_reference,
    _HAS_TRITON,
)


def _dense_attention_reference(query, keys, values, scale):
    """
    Ground truth: standard scaled dot-product attention over an already
    contiguous (seq_len, num_heads, head_dim) key/value tensor. No blocks,
    no gathering — this is what paged attention must reproduce exactly.
    """
    scores = torch.einsum("hd,thd->ht", query, keys) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", weights, values)


def test_paged_attention_matches_dense_reference_with_scattered_blocks():
    """
    Correctness of the core idea: attention output must be identical whether
    a sequence's KV data lives contiguously or scattered across arbitrary,
    non-adjacent physical block IDs. This is the whole point of paging — if
    this test passes, block placement genuinely doesn't matter.
    """
    torch.manual_seed(0)
    num_heads, head_dim, block_size = 2, 8, 4
    scale = head_dim ** -0.5

    seq_lens = [6, 9]              # different lengths, different block counts
    num_blocks = 10
    key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
    value_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)

    # Deliberately non-contiguous, non-sorted block assignments.
    block_tables_list = [
        [7, 2],       # seq0 needs ceil(6/4) = 2 blocks
        [9, 0, 4],    # seq1 needs ceil(9/4) = 3 blocks
    ]
    max_blocks = max(len(bt) for bt in block_tables_list)
    block_tables = torch.zeros(2, max_blocks, dtype=torch.long)
    for i, bt in enumerate(block_tables_list):
        block_tables[i, : len(bt)] = torch.tensor(bt)

    context_lens = torch.tensor(seq_lens)
    query = torch.randn(2, num_heads, head_dim)

    output = paged_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, scale
    )

    for s, seq_len in enumerate(seq_lens):
        block_ids = block_tables_list[s]
        keys = torch.cat([key_cache[b] for b in block_ids], dim=0)[:seq_len]
        values = torch.cat([value_cache[b] for b in block_ids], dim=0)[:seq_len]
        expected = _dense_attention_reference(query[s], keys, values, scale)
        assert torch.allclose(output[s], expected, atol=1e-6), (
            f"seq {s}: paged output diverges from dense ground truth"
        )


def test_paged_attention_ignores_padding_beyond_context_len():
    """
    Tokens beyond context_len inside a partially-filled last block must never
    influence the output — even if that memory holds garbage. Poison the
    unused slots with NaN and confirm the result stays finite.
    """
    torch.manual_seed(1)
    num_heads, head_dim, block_size = 1, 4, 4
    scale = head_dim ** -0.5

    key_cache = torch.randn(2, block_size, num_heads, head_dim)
    value_cache = torch.randn(2, block_size, num_heads, head_dim)

    # Only 2 of 4 slots in block 0 are real; poison the rest.
    key_cache[0, 2:] = float("nan")
    value_cache[0, 2:] = float("nan")

    block_tables = torch.tensor([[0]])
    context_lens = torch.tensor([2])
    query = torch.randn(1, num_heads, head_dim)

    output = paged_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, scale
    )
    assert torch.isfinite(output).all()


def test_paged_attention_single_token_context():
    """context_len=1 is the very first decode step: attention degenerates to
    returning that one cached value exactly (softmax over 1 element = 1.0)."""
    num_heads, head_dim, block_size = 1, 4, 4
    scale = head_dim ** -0.5

    key_cache = torch.randn(1, block_size, num_heads, head_dim)
    value_cache = torch.randn(1, block_size, num_heads, head_dim)
    block_tables = torch.tensor([[0]])
    context_lens = torch.tensor([1])
    query = torch.randn(1, num_heads, head_dim)

    output = paged_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, scale
    )
    expected = value_cache[0, 0]   # only real slot: softmax weight = 1.0 on it
    assert torch.allclose(output[0], expected, atol=1e-6)


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton paged attention requires a CUDA GPU (verified on Kaggle)",
)
def test_paged_attention_triton_matches_reference():
    """
    GPU-only: the fused Triton kernel must match the pure-PyTorch reference
    bit-for-bit (within float tolerance) on the same scattered block layout.
    """
    from mini_vllm.kv_cache.paged_attention import paged_attention_triton

    torch.manual_seed(2)
    num_heads, head_dim, block_size = 4, 64, 16     # powers of 2 (tl.arange requirement)
    scale = head_dim ** -0.5
    device = "cuda"

    num_seqs, num_blocks = 8, 40
    key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device)
    value_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device)

    context_lens = torch.randint(1, block_size * 3, (num_seqs,), device=device)
    max_blocks = int(((context_lens.max() + block_size - 1) // block_size).item())
    block_tables = torch.stack([
        torch.randperm(num_blocks, device=device)[:max_blocks]
        for _ in range(num_seqs)
    ])
    query = torch.randn(num_seqs, num_heads, head_dim, device=device)

    ref = paged_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, scale
    )
    triton_out = paged_attention_triton(
        query, key_cache, value_cache, block_tables, context_lens, scale
    )
    assert torch.allclose(ref, triton_out, atol=1e-2, rtol=1e-2), (
        f"max diff: {(ref - triton_out).abs().max().item()}"
    )
