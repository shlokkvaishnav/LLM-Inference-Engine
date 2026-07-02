"""
Paged decode path for Llama-family models — Milestone 4 engine integration.

Why a separate class instead of extending ModelRunner:
  ModelRunner.decode_one (M3) is proven correct against HF token-for-token
  and stays untouched here — nothing in this file can regress M1-M3. This
  class targets Llama specifically (TinyLlama is the production model) and
  replaces ONLY the attention core with the paged kernel; every other
  sub-layer (RMSNorm, RoPE, SwiGLU MLP, Linear projections) calls the SAME
  nn.Module instances the loaded HF model already has, so their correctness
  is inherited directly from HF's own tested implementation — we are not
  reimplementing Llama, only its attention memory layout.

Interface parity with ModelRunner so LLMEngine can drive either:
  prefill_and_store(sequences) -> list[int]
  decode_batch(sequences)      -> list[int]   (batched — the M4 payoff: ONE
                                  forward pass for the WHOLE running batch,
                                  vs ModelRunner.decode_one's one-per-sequence)
  free_seq(seq_id) -> None

Prefill still runs through the model's own batched forward (reusing M2's
already-correct left-padded logic) and only copies the resulting K/V into
the block pool — only the DECODE step is reimplemented manually. This
mirrors real vLLM, which also uses a different kernel for prefill
(contiguous) than for decode (paged).

Position bookkeeping (the part most likely to have an off-by-one — verify
against test_paged_llama_matches_dense_generate before trusting this file):
  At the moment decode_batch(sequences) is called for a sequence with
  seq.length == L, output_token_ids[-1] is the token at 0-indexed position
  L-1 (it was sampled and appended by a PREVIOUS step; its own K/V has never
  been computed since it was never fed through the model until now).
  So: RoPE position_ids = L-1. Write this token's K/V at pool position L-1.
  Attention context length = L (positions 0..L-1 inclusive — L tokens total).
  The newly predicted token (from this call's logits) occupies position L,
  and its K/V won't be written until the NEXT decode_batch call.
"""
from __future__ import annotations

from typing import Any

import torch

from mini_vllm.engine.sequence import Sequence
from mini_vllm.kv_cache.block_manager import BlockManager
from mini_vllm.kv_cache.paged_attention import (
    paged_attention_reference,
    paged_attention_triton,
    _HAS_TRITON,
)
from mini_vllm.sampling.sampler import Sampler

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _hf_apply_rope
except ImportError:
    _hf_apply_rope = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim: int = 1):
    """
    Standard HF rotary formula (used as a fallback when the internal
    transformers function isn't importable). Prefer the real import when
    available — it guarantees exact numerical match with whatever HF
    version is installed, including any model-specific RoPE variants.
    """
    if _hf_apply_rope is not None:
        return _hf_apply_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _get_rotary_emb(model: Any):
    """
    Modern transformers (>=~4.36) compute RoPE cos/sin once per forward pass
    in LlamaModel.rotary_emb and share it across layers. Older versions
    attached a rotary_emb to each attention layer instead — fall back to
    that so this still works if the shared module isn't present.
    """
    if hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    if hasattr(model.model.layers[0].self_attn, "rotary_emb"):
        return model.model.layers[0].self_attn.rotary_emb
    raise AttributeError(
        "Could not locate a rotary_emb module on this Llama model — "
        "transformers version may be too old/new for this integration."
    )


class PagedLlamaRunner:
    """
    Drives a loaded LlamaForCausalLM through prefill (dense, batched) and
    decode (paged, batched) using a shared physical KV-cache block pool —
    one (key_pool, value_pool) tensor pair per transformer layer.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        block_manager: BlockManager,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.block_manager = block_manager
        self.device = torch.device(device)
        self.rotary_emb = _get_rotary_emb(model)

        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_q_heads)
        self.head_dim = cfg.hidden_size // self.num_q_heads
        self.hidden_size = cfg.hidden_size
        self.scale = self.head_dim ** -0.5

        nb, bs = block_manager.num_blocks, block_manager.block_size
        self.key_pool = [
            torch.zeros(nb, bs, self.num_kv_heads, self.head_dim, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.value_pool = [
            torch.zeros(nb, bs, self.num_kv_heads, self.head_dim, dtype=dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

    # -----------------------------------------------------------------------
    # Prefill — dense batched forward (reuses HF's own attention internally),
    # then copies the resulting per-token K/V into the block pool.
    # -----------------------------------------------------------------------

    def prefill_and_store(self, sequences: list[Sequence]) -> list[int]:
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        batch = len(sequences)
        max_len = max(len(s.prompt_token_ids) for s in sequences)

        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(batch, max_len, dtype=torch.long, device=self.device)
        offsets = []
        for i, seq in enumerate(sequences):
            toks = seq.prompt_token_ids
            offset = max_len - len(toks)
            offsets.append(offset)
            input_ids[i, offset:] = torch.tensor(toks, dtype=torch.long, device=self.device)
            attention_mask[i, offset:] = 1

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp(min=0)

        with torch.no_grad():
            out = self.model(
                input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=True,
            )

        # Reserve blocks for the full prompt. Idempotent if the caller (e.g.
        # Scheduler, when block_manager is shared) already allocated for
        # this sequence — allocate() computes `needed` relative to
        # len(seq.block_table), so a second call is a no-op.
        for seq in sequences:
            self.block_manager.allocate(seq)

        past_kv = out.past_key_values
        for layer_idx in range(self.num_layers):
            k_layer, v_layer = self._layer_kv(past_kv, layer_idx)   # (batch, num_kv_heads, max_len, head_dim)
            for i, seq in enumerate(sequences):
                offset = offsets[i]
                # Real tokens only — skip the left-padding prefix.
                k_real = k_layer[i, :, offset:, :].transpose(0, 1)   # (prompt_len, num_kv_heads, head_dim)
                v_real = v_layer[i, :, offset:, :].transpose(0, 1)
                self._write_tokens(layer_idx, seq, start=0, k=k_real, v=v_real)

        logits = out.logits[:, -1, :]
        return [
            int(self._sample(logits[i : i + 1], seq).item())
            for i, seq in enumerate(sequences)
        ]

    @staticmethod
    def _layer_kv(past_kv: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Same version-straddling logic as ModelRunner._slice_kv (M3 fix)."""
        if hasattr(past_kv, "layers"):
            layer = past_kv.layers[layer_idx]
            return layer.keys, layer.values
        if hasattr(past_kv, "key_cache"):
            return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
        return past_kv[layer_idx][0], past_kv[layer_idx][1]

    def _write_tokens(
        self, layer_idx: int, seq: Sequence, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write k/v (num_tokens, num_kv_heads, head_dim) into the pool, one
        token at a time, starting at 0-indexed logical position `start`."""
        block_size = self.block_manager.block_size
        for t in range(k.shape[0]):
            pos = start + t
            physical_block = seq.block_table[pos // block_size]
            slot = pos % block_size
            self.key_pool[layer_idx][physical_block, slot] = k[t]
            self.value_pool[layer_idx][physical_block, slot] = v[t]

    # -----------------------------------------------------------------------
    # Decode — batched paged attention. ONE forward pass (per layer) for the
    # entire running batch, regardless of how many sequences are in flight —
    # this is the O(1)-passes-per-step replacement for decode_one's O(N).
    # -----------------------------------------------------------------------

    def decode_batch(self, sequences: list[Sequence]) -> list[int]:
        for seq in sequences:
            if not self.block_manager.has_capacity_for_next_token(seq):
                raise RuntimeError(
                    f"seq {seq.seq_id} needs a block but the pool is exhausted; "
                    "Scheduler should have preempted before calling decode_batch."
                )

        input_ids = torch.tensor(
            [[s.output_token_ids[-1]] for s in sequences], dtype=torch.long, device=self.device
        )
        # Position of the token being fed in THIS call (see module docstring).
        position_ids = torch.tensor(
            [[s.length - 1] for s in sequences], dtype=torch.long, device=self.device
        )

        x = self.model.model.embed_tokens(input_ids)          # (batch, 1, hidden)
        cos, sin = self.rotary_emb(x, position_ids)

        block_tables, context_lens = self._build_block_tables(sequences)

        for layer_idx, layer in enumerate(self.model.model.layers):
            x = self._paged_decode_layer(
                layer, layer_idx, x, cos, sin, block_tables, context_lens, sequences
            )

        x = self.model.model.norm(x)
        logits = self.model.lm_head(x)[:, -1, :]               # (batch, vocab)

        return [
            int(self._sample(logits[i : i + 1], seq).item())
            for i, seq in enumerate(sequences)
        ]

    def _build_block_tables(self, sequences: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        context_lens[i] = seq.length = number of real tokens occupying
        positions 0..seq.length-1 — this is exactly what paged_attention
        needs (it already includes the token being fed in this call).
        """
        max_blocks = max(len(s.block_table) for s in sequences)
        bt = torch.zeros(len(sequences), max_blocks, dtype=torch.long, device=self.device)
        ctx = torch.zeros(len(sequences), dtype=torch.long, device=self.device)
        for i, seq in enumerate(sequences):
            n = len(seq.block_table)
            bt[i, :n] = torch.tensor(seq.block_table, dtype=torch.long, device=self.device)
            ctx[i] = seq.length
        return bt, ctx

    def _paged_decode_layer(
        self, layer, layer_idx, x, cos, sin, block_tables, context_lens, sequences
    ) -> torch.Tensor:
        residual = x
        h = layer.input_layernorm(x)

        q = layer.self_attn.q_proj(h).view(x.shape[0], 1, self.num_q_heads, self.head_dim)
        k = layer.self_attn.k_proj(h).view(x.shape[0], 1, self.num_kv_heads, self.head_dim)
        v = layer.self_attn.v_proj(h).view(x.shape[0], 1, self.num_kv_heads, self.head_dim)

        # apply_rotary_pos_emb expects (batch, num_heads, seq_len, head_dim).
        q_r, k_r = _apply_rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        q = q_r.transpose(1, 2).squeeze(1)   # (batch, num_q_heads, head_dim)
        k = k_r.transpose(1, 2).squeeze(1)   # (batch, num_kv_heads, head_dim)
        v = v.squeeze(1)                     # (batch, num_kv_heads, head_dim)

        block_size = self.block_manager.block_size
        for i, seq in enumerate(sequences):
            pos = seq.length - 1   # position of the token just fed in — see module docstring
            physical_block = seq.block_table[pos // block_size]
            slot = pos % block_size
            self.key_pool[layer_idx][physical_block, slot] = k[i]
            self.value_pool[layer_idx][physical_block, slot] = v[i]

        if self.num_kv_heads != self.num_q_heads:
            # GQA: expand cache heads to match query heads for the kernel call.
            # Memory-inefficient (recomputed every layer/step) — a true
            # GQA-aware kernel (grouping queries per kv-head internally) is
            # the natural next optimization, not implemented here.
            group = self.num_q_heads // self.num_kv_heads
            key_cache = self.key_pool[layer_idx].repeat_interleave(group, dim=2)
            value_cache = self.value_pool[layer_idx].repeat_interleave(group, dim=2)
        else:
            key_cache = self.key_pool[layer_idx]
            value_cache = self.value_pool[layer_idx]

        attn_fn = paged_attention_triton if (self.device.type == "cuda" and _HAS_TRITON) else paged_attention_reference
        attn_out = attn_fn(q, key_cache, value_cache, block_tables, context_lens, self.scale)
        attn_out = attn_out.reshape(x.shape[0], 1, self.hidden_size)

        attn_out = layer.self_attn.o_proj(attn_out)
        x = residual + attn_out

        residual = x
        h = layer.post_attention_layernorm(x)
        h = layer.mlp(h)
        return residual + h

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> torch.Tensor:
        return Sampler.sample(
            logits,
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        )

    def free_seq(self, seq_id: int) -> None:
        # No runner-local state to release: the pool is shared, static
        # storage indexed by block ID. Block release is Scheduler.free()'s
        # job (via BlockManager) — nothing to do here.
        pass
