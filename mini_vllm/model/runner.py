"""
ModelRunner: orchestrates forward passes over sequences.

Milestone 1/2: generate() — batched prefill→decode for a fixed group of
  sequences. The whole batch flows together until the last one finishes.

Milestone 3: prefill_and_store() + decode_one() + free_seq() — per-sequence
  KV cache storage that the continuous-batching engine loop drives.

Position IDs note (why we pass them explicitly):
  GPT-2 and other models with absolute position embeddings do not compute
  position_ids from attention_mask on their own. For a left-padded batch,
  token at input index 6 would get position_id=6 even if it is really the
  first real token (preceded by 5 padding tokens). That causes wrong outputs.
  We always pass explicit position_ids so every model type behaves correctly.

  Prefill: position_ids = cumsum(attention_mask) - 1, clamped ≥ 0
    → padding positions get 0 (masked anyway), real tokens get 0,1,2,...
  Decode:  position_ids = attention_mask.sum(dim=1) before appending new col
    → equals the count of real tokens so far = correct next position

Cache format note:
  Transformers 4.38+ returns a DynamicCache object. We slice it using its
  .key_cache / .value_cache lists (always per-layer tensors). For the rare
  legacy tuple-of-tuples format we use index access [0][1] not tuple
  unpacking, so 3-element tuples from some transformers versions don't crash.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.sampling.sampler import Sampler

if TYPE_CHECKING:
    from mini_vllm.model.loader import ModelConfig


# ---------------------------------------------------------------------------
# KV-cache helpers (M3)
# ---------------------------------------------------------------------------

def _slice_kv(past_kv: Any, i: int) -> Any:
    """
    Extract sequence i from a batched cache. Preserves the cache type so the
    sliced result can be passed straight back to the model without conversion.

    DynamicCache has had two incompatible internal layouts across transformers
    versions, so we handle both — pinning one exact transformers version isn't
    possible here since Kaggle and CI install different versions:
      - transformers <5.x (e.g. 4.46.3, pinned on Kaggle): per-layer tensors
        live in flat `.key_cache` / `.value_cache` lists on the DynamicCache.
      - transformers >=5.x (unpinned CI): those lists are gone. Each layer is
        now a `CacheLayerMixin` object in `.layers`, exposing `.keys`/`.values`.
        Rebuild via `DynamicCache(ddp_cache_data=[(k, v), ...])` — the
        documented constructor for building a cache from raw K/V tensors.
    Legacy tuple-of-tuples (very old / non-Cache path): index access [0][1]
      avoids the 3-element unpack error some DynamicCache iterators raise.
    """
    if hasattr(past_kv, "layers"):
        from transformers.cache_utils import DynamicCache
        return DynamicCache(ddp_cache_data=[
            (layer.keys[i : i + 1].contiguous(), layer.values[i : i + 1].contiguous())
            for layer in past_kv.layers
        ])
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            from transformers import DynamicCache   # older path
        sliced = DynamicCache()
        sliced.key_cache   = [k[i : i + 1].contiguous() for k in past_kv.key_cache]
        sliced.value_cache = [v[i : i + 1].contiguous() for v in past_kv.value_cache]
        if hasattr(past_kv, "_seen_tokens"):
            sliced._seen_tokens = past_kv._seen_tokens
        return sliced
    # Legacy: each layer is (key, value) or (key, value, extra…) — index-safe
    return tuple(
        (layer[0][i : i + 1].contiguous(), layer[1][i : i + 1].contiguous())
        for layer in past_kv
    )


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------

class ModelRunner:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: "ModelConfig",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)

        # Per-sequence KV cache storage for M3 (continuous batching).
        # Key: seq_id  Value: DynamicCache or tuple, batch dim = 1
        self._kv_cache: dict[int, Any] = {}
        # Key: seq_id  Value: attention mask of shape (1, seq_len)
        self._attn_masks: dict[int, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Shared forward-pass primitives
    # -----------------------------------------------------------------------

    def prefill(
        self, sequences: list[Sequence]
    ) -> tuple[list[int], Any, torch.Tensor]:
        """
        Batched prefill: left-pad all prompts, one forward pass, return first tokens.

        Left-padding ensures logits[:, -1, :] is always the last *real* token
        for every sequence. Explicit position_ids (from cumsum of the mask)
        fix GPT-2 and other absolute-position models which otherwise assign
        wrong positions to padded sequences.

        Returns (first_token_ids, past_key_values, attention_mask).
        """
        pad_id: int = getattr(self.tokenizer, "pad_token_id", None) or 0
        batch = len(sequences)
        max_len = max(len(s.prompt_token_ids) for s in sequences)

        input_ids = torch.full(
            (batch, max_len), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            batch, max_len, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(sequences):
            toks = seq.prompt_token_ids
            offset = max_len - len(toks)
            input_ids[i, offset:] = torch.tensor(
                toks, dtype=torch.long, device=self.device
            )
            attention_mask[i, offset:] = 1

        # position_ids: real token at mask offset k gets position k-1 (0-indexed).
        # Padding positions get 0 — they are masked out anyway.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp(min=0)

        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        logits = out.logits[:, -1, :]   # last real token per row (left-pad guarantee)

        tokens = [
            int(Sampler.sample(
                logits[i : i + 1],
                temperature=seq.sampling_params.temperature,
                top_p=seq.sampling_params.top_p,
                top_k=seq.sampling_params.top_k,
            ).item())
            for i, seq in enumerate(sequences)
        ]
        return tokens, out.past_key_values, attention_mask

    def decode(
        self,
        sequences: list[Sequence],
        past_kv: Any,
        attention_mask: torch.Tensor,
    ) -> tuple[list[int], Any, torch.Tensor]:
        """
        One decode step over the full batch.

        position_ids for the new token = count of real tokens so far
          = attention_mask.sum(dim=1) before we append the new column.
        This is the correct next position regardless of how many padding
        tokens exist to the left of the prompt.

        Returns (next_token_ids, updated_past_kv, updated_attention_mask).
        """
        batch = len(sequences)
        input_ids = torch.tensor(
            [[s.output_token_ids[-1]] for s in sequences],
            dtype=torch.long,
            device=self.device,
        )   # (batch, 1)

        # Position of the token we are about to generate.
        position_ids = attention_mask.sum(dim=1, keepdim=True)   # (batch, 1)

        new_col = torch.ones(batch, 1, dtype=torch.long, device=self.device)
        full_mask = torch.cat([attention_mask, new_col], dim=1)

        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=past_kv,
                attention_mask=full_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        logits = out.logits[:, -1, :]
        tokens = [
            int(Sampler.sample(
                logits[i : i + 1],
                temperature=seq.sampling_params.temperature,
                top_p=seq.sampling_params.top_p,
                top_k=seq.sampling_params.top_k,
            ).item())
            for i, seq in enumerate(sequences)
        ]
        return tokens, out.past_key_values, full_mask

    # -----------------------------------------------------------------------
    # M2 — static batching
    # -----------------------------------------------------------------------

    def generate(self, sequences: list[Sequence]) -> list[Sequence]:
        """
        Prefill → decode until every sequence finishes (static batching).

        All sequences run together as one fixed batch. Sequences that hit EOS
        early stay in the tensor but their tokens are discarded — the wasted
        compute is what M3 fixes with continuous batching.
        """
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        finished = [False] * len(sequences)

        first_tokens, past_kv, attn_mask = self.prefill(sequences)

        for i, (seq, tok) in enumerate(zip(sequences, first_tokens)):
            seq.append_token(tok)
            seq.status = SequenceStatus.RUNNING
            if seq.check_stop() or tok == eos_id:
                seq.status = SequenceStatus.FINISHED
                finished[i] = True

        while not all(finished):
            next_tokens, past_kv, attn_mask = self.decode(
                sequences, past_kv, attn_mask
            )
            for i, (seq, tok) in enumerate(zip(sequences, next_tokens)):
                if not finished[i]:
                    seq.append_token(tok)
                    if seq.check_stop() or tok == eos_id:
                        seq.status = SequenceStatus.FINISHED
                        finished[i] = True

        return sequences

    # -----------------------------------------------------------------------
    # M3 — per-sequence KV cache
    # -----------------------------------------------------------------------

    def prefill_and_store(self, sequences: list[Sequence]) -> list[int]:
        """
        Batched prefill + store per-sequence KV caches.

        Slices the batched cache into per-sequence entries (batch dim = 1)
        using _slice_kv, which preserves the DynamicCache type so the result
        can be passed straight to the model in decode_one without conversion.
        """
        first_tokens, batch_kv, attn_mask = self.prefill(sequences)
        for i, seq in enumerate(sequences):
            self._kv_cache[seq.seq_id]  = _slice_kv(batch_kv, i)
            self._attn_masks[seq.seq_id] = attn_mask[i : i + 1]
        return first_tokens

    def decode_one(self, seq: Sequence) -> int:
        """
        Decode one step for a single sequence using its stored KV cache.

        Runs one forward pass (batch=1). O(N) passes per decode step — correct
        but not optimal. M4 replaces this with a single batched paged-attention
        pass regardless of how many sequences are in flight.

        Stores out.past_key_values directly (no slicing needed since batch=1).
        """
        past_kv = self._kv_cache[seq.seq_id]
        mask     = self._attn_masks[seq.seq_id]

        input_id = torch.tensor(
            [[seq.output_token_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position of the token we are generating = count of real tokens so far.
        position_ids = mask.sum(dim=1, keepdim=True)   # (1, 1)
        new_mask = torch.cat(
            [mask, torch.ones(1, 1, dtype=torch.long, device=self.device)], dim=1
        )

        with torch.no_grad():
            out = self.model(
                input_id,
                past_key_values=past_kv,
                attention_mask=new_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        logits = out.logits[:, -1, :]
        tok = int(Sampler.sample(
            logits,
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        ).item())

        # batch=1 → store directly, no slicing required.
        self._kv_cache[seq.seq_id]  = out.past_key_values
        self._attn_masks[seq.seq_id] = new_mask

        return tok

    def free_seq(self, seq_id: int) -> None:
        """Release KV cache for a finished or preempted sequence."""
        self._kv_cache.pop(seq_id, None)
        self._attn_masks.pop(seq_id, None)
