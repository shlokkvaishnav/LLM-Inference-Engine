"""
Paged attention — Milestone 4c.

This is the kernel that makes paged KV-cache actually usable: given a query
for the token being decoded, and a KV-cache pool where each sequence's data
lives in *scattered, non-contiguous* physical blocks, compute standard
attention output without ever materialising a contiguous per-sequence tensor.

Scope: decode-step attention only (one query token per sequence, attending
to everything cached so far). This matches vLLM's own `paged_attention_v1`
kernel semantics — prefill uses a different flash-attention-style kernel
over contiguous new tokens, which is out of scope here since our prefill
already runs as one batched dense forward pass through the HF model.

Two implementations, same contract:
  paged_attention_reference — pure PyTorch, CPU or GPU, O(num_seqs) Python
    loop. This is the ground truth: simple enough to trust by inspection,
    used both directly (CPU path) and as the correctness oracle for the
    Triton kernel below.
  paged_attention_triton    — fused GPU kernel, one program per
    (sequence, head), online-softmax accumulation across blocks so it never
    materialises the gathered (context_len, head_dim) tensor either. This is
    the O(1)-passes-per-step replacement for ModelRunner.decode_one's current
    O(N) per-sequence forward passes.

Shapes (fixed contract for both implementations):
  query:        (num_seqs, num_heads, head_dim)      — one query per sequence
  key_cache:    (num_blocks, block_size, num_heads, head_dim)
  value_cache:  (num_blocks, block_size, num_heads, head_dim)
  block_tables: (num_seqs, max_blocks_per_seq) int64  — physical block IDs,
                logical order; entries past a sequence's real block count
                are unused (never read, since context_lens bounds the loop)
  context_lens: (num_seqs,) int64                     — real cached token
                count per sequence (may be less than max_blocks*block_size —
                the tail of the last block is padding, never attended to)
  scale:        1/sqrt(head_dim), applied to raw dot-product scores

  returns:      (num_seqs, num_heads, head_dim)
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Reference implementation — pure PyTorch, always available
# ---------------------------------------------------------------------------

def paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Gather each sequence's blocks into a contiguous (context_len, H, D) view,
    run standard scaled dot-product attention, per sequence.

    O(num_seqs) Python-level loop — fine for correctness testing and as a
    CPU fallback; not the fast path (see paged_attention_triton for that).
    """
    num_seqs, num_heads, head_dim = query.shape
    block_size = key_cache.shape[1]
    output = torch.empty_like(query)

    for s in range(num_seqs):
        ctx_len = int(context_lens[s].item())
        num_blocks_needed = (ctx_len + block_size - 1) // block_size

        block_ids = block_tables[s, :num_blocks_needed]            # (nb,)
        k_blocks = key_cache[block_ids]                            # (nb, block_size, H, D)
        v_blocks = value_cache[block_ids]

        # Flatten blocks into one sequence dimension, then trim padding
        # slots in the (partially filled) last block.
        keys = k_blocks.reshape(-1, num_heads, head_dim)[:ctx_len]     # (ctx_len, H, D)
        values = v_blocks.reshape(-1, num_heads, head_dim)[:ctx_len]

        q = query[s]                                                # (H, D)
        scores = torch.einsum("hd,thd->ht", q, keys) * scale        # (H, ctx_len)
        weights = torch.softmax(scores, dim=-1)
        output[s] = torch.einsum("ht,thd->hd", weights, values)     # (H, D)

    return output


# ---------------------------------------------------------------------------
# Triton kernel — fused GPU implementation, correctness-verified on Kaggle
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _paged_attention_kernel(
        q_ptr, k_cache_ptr, v_cache_ptr, block_tables_ptr, context_lens_ptr, out_ptr,
        scale,
        stride_q_seq, stride_q_head, stride_q_dim,
        stride_kv_block, stride_kv_slot, stride_kv_head, stride_kv_dim,
        stride_bt_seq, stride_bt_block,
        stride_out_seq, stride_out_head, stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_NUM_BLOCKS: tl.constexpr,
    ):
        """
        One program instance per (sequence, head). Walks that sequence's
        block table, accumulating attention output with online (flash-attention
        style) softmax — never materialises the full (context_len, HEAD_DIM)
        gathered tensor, so memory use is O(BLOCK_SIZE) regardless of context
        length.
        """
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        context_len = tl.load(context_lens_ptr + seq_idx)

        dim_offsets = tl.arange(0, HEAD_DIM)
        q_offset = seq_idx * stride_q_seq + head_idx * stride_q_head
        q = tl.load(q_ptr + q_offset + dim_offsets * stride_q_dim)   # (HEAD_DIM,)

        m_i = -float("inf")          # running max score (for numerical stability)
        l_i = 0.0                    # running softmax denominator
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)   # running weighted-value sum

        num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        for b in range(MAX_NUM_BLOCKS):
            if b < num_blocks:
                physical_block = tl.load(
                    block_tables_ptr + seq_idx * stride_bt_seq + b * stride_bt_block
                )
                slot_offsets = tl.arange(0, BLOCK_SIZE)
                token_idx = b * BLOCK_SIZE + slot_offsets
                valid = token_idx < context_len

                k_base = k_cache_ptr + physical_block * stride_kv_block
                v_base = v_cache_ptr + physical_block * stride_kv_block
                kv_offsets = (
                    slot_offsets[:, None] * stride_kv_slot
                    + head_idx * stride_kv_head
                    + dim_offsets[None, :] * stride_kv_dim
                )

                k = tl.load(k_base + kv_offsets, mask=valid[:, None], other=0.0)  # (BLOCK_SIZE, HEAD_DIM)
                v = tl.load(v_base + kv_offsets, mask=valid[:, None], other=0.0)

                scores = tl.sum(q[None, :] * k, axis=1) * scale       # (BLOCK_SIZE,)
                scores = tl.where(valid, scores, -float("inf"))

                m_ij = tl.max(scores, axis=0)
                m_new = tl.maximum(m_i, m_ij)

                alpha = tl.exp(m_i - m_new)          # rescale factor for old accumulators
                p = tl.exp(scores - m_new)           # (BLOCK_SIZE,)

                l_i = l_i * alpha + tl.sum(p, axis=0)
                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                m_i = m_new

        out = acc / l_i
        out_offset = seq_idx * stride_out_seq + head_idx * stride_out_head
        tl.store(out_ptr + out_offset + dim_offsets * stride_out_dim, out)

    def paged_attention_triton(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """
        GPU-only fused paged attention. HEAD_DIM and BLOCK_SIZE must be
        powers of 2 (tl.arange requirement). Falls back is the caller's
        responsibility — call paged_attention_reference on CPU.
        """
        assert query.is_cuda, "paged_attention_triton requires CUDA tensors"
        num_seqs, num_heads, head_dim = query.shape
        block_size = key_cache.shape[1]
        max_num_blocks = block_tables.shape[1]

        output = torch.empty_like(query)
        grid = (num_seqs, num_heads)

        _paged_attention_kernel[grid](
            query, key_cache, value_cache, block_tables, context_lens, output,
            scale,
            query.stride(0), query.stride(1), query.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            block_tables.stride(0), block_tables.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            MAX_NUM_BLOCKS=max_num_blocks,
        )
        return output

else:
    def paged_attention_triton(*args, **kwargs):
        raise ImportError("triton is not installed — install mini-vllm[gpu] on a CUDA machine")
