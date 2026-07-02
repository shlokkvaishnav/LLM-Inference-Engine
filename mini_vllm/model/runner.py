"""
ModelRunner: orchestrates forward passes over sequences.

Milestone 1/2: generate() — batched prefill→decode for a fixed group of
  sequences. The whole batch flows together until the last one finishes.

Milestone 3: prefill_and_store() + decode_one() + free_seq() — per-sequence
  KV cache storage that the continuous-batching engine loop drives.
  decode_one() runs one forward pass per sequence (O(N) passes per step).
  That is intentionally simple: M4 replaces it with a single batched pass
  via paged attention, making it O(1) passes regardless of batch size.

Milestone 4+: the Scheduler replaces generate(); prefill/decode stay.

KV cache format note:
  Transformers 4.38+ returns past_key_values as a DynamicCache object.
  _to_tuple_kv() normalises it to a tuple of (key, value) tensor pairs so
  the rest of the code can use plain indexing without knowing which version
  of transformers is installed.  We pass tuple-of-tuples back into the model
  — transformers 4.38+ accepts both formats on input.
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
# KV cache helpers
# ---------------------------------------------------------------------------

def _to_tuple_kv(past_kv: Any) -> tuple:
    """
    Normalise any past_key_values format to a tuple of (key, value) pairs.

    Transformers 4.38+ may return a DynamicCache object with .key_cache and
    .value_cache lists. zip() over those always produces 2-tuples, so the
    rest of the code can use `for k, v in kv` without version guards.
    """
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return tuple(zip(past_kv.key_cache, past_kv.value_cache))
    # Legacy tuple-of-tuples; each element is already (key, value).
    return tuple(past_kv)


def _slice_kv(kv: tuple, i: int) -> tuple:
    """
    Extract one sequence (row i) from a batched KV cache tuple.

    kv: tuple of (key, value) pairs, each tensor of shape (B, H, S, D).
    Returns a tuple of (key, value) pairs with shape (1, H, S, D).
    """
    return tuple(
        (k[i : i + 1].contiguous(), v[i : i + 1].contiguous())
        for k, v in kv
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
        # Key: seq_id  Value: tuple of (key, value) tensor pairs (batch dim = 1)
        self._kv_cache: dict[int, tuple] = {}
        # Key: seq_id  Value: attention mask of shape (1, seq_len)
        self._attn_masks: dict[int, torch.Tensor] = {}

    # -----------------------------------------------------------------------
    # Shared forward-pass primitives  (used by both M2 generate and M3 engine)
    # -----------------------------------------------------------------------

    def prefill(
        self, sequences: list[Sequence]
    ) -> tuple[list[int], Any, torch.Tensor]:
        """
        Batched prefill: left-pad all prompts, one forward pass, return first tokens.

        Left-padding ensures logits[:, -1, :] is always the last *real* token
        for every sequence — no per-sequence index arithmetic needed.
        The attention mask (0=padding, 1=real) keeps padded positions masked.

        Returns (first_token_ids, past_key_values, attention_mask).
        The caller threads past_key_values and attention_mask into decode().
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

        with torch.no_grad():
            out = self.model(
                input_ids, attention_mask=attention_mask, use_cache=True
            )

        logits = out.logits[:, -1, :]   # (batch, vocab_size) — last real token

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

        Extends the attention mask by one real column (new token attends to
        itself + all real past positions) and feeds one token per sequence.
        past_kv flows straight through without inspection.

        Returns (next_token_ids, updated_past_kv, updated_attention_mask).
        """
        batch = len(sequences)
        input_ids = torch.tensor(
            [[s.output_token_ids[-1]] for s in sequences],
            dtype=torch.long,
            device=self.device,
        )
        new_col = torch.ones(batch, 1, dtype=torch.long, device=self.device)
        full_mask = torch.cat([attention_mask, new_col], dim=1)

        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=past_kv,
                attention_mask=full_mask,
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
    # M2 — static batching  (used by generate(); M1 tests go through here too)
    # -----------------------------------------------------------------------

    def generate(self, sequences: list[Sequence]) -> list[Sequence]:
        """
        Prefill → decode until every sequence in the batch finishes.

        Static batching: the full batch runs together until the LAST sequence
        finishes. Sequences that hit EOS earlier stay in the tensor (we stop
        recording their tokens); the wasted compute is what M3 fixes.
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
    # M3 — per-sequence KV cache  (used by LLMEngine)
    # -----------------------------------------------------------------------

    def prefill_and_store(self, sequences: list[Sequence]) -> list[int]:
        """
        Batched prefill + store per-sequence KV caches.

        After the joint forward pass, slices the batch KV cache into
        individual tensors stored in self._kv_cache[seq_id].

        Returns the first generated token for each sequence.
        """
        first_tokens, batch_kv, attn_mask = self.prefill(sequences)

        kv_tuple = _to_tuple_kv(batch_kv)
        for i, seq in enumerate(sequences):
            self._kv_cache[seq.seq_id] = _slice_kv(kv_tuple, i)
            self._attn_masks[seq.seq_id] = attn_mask[i : i + 1]

        return first_tokens

    def decode_one(self, seq: Sequence) -> int:
        """
        Decode a single sequence step using its stored KV cache.

        Runs one forward pass for this sequence alone (batch=1). This is
        O(N) passes per decode step — simple and correct but not optimal.
        M4 replaces this with a single batched pass via paged attention.

        Updates _kv_cache[seq_id] and _attn_masks[seq_id] in place.
        Returns the next token id.
        """
        past_kv = self._kv_cache[seq.seq_id]
        mask = self._attn_masks[seq.seq_id]

        input_id = torch.tensor(
            [[seq.output_token_ids[-1]]], dtype=torch.long, device=self.device
        )
        new_mask = torch.cat(
            [mask, torch.ones(1, 1, dtype=torch.long, device=self.device)], dim=1
        )

        with torch.no_grad():
            out = self.model(
                input_id,
                past_key_values=past_kv,
                attention_mask=new_mask,
                use_cache=True,
            )

        logits = out.logits[:, -1, :]
        tok = int(Sampler.sample(
            logits,
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        ).item())

        # Store updated cache (now one position longer).
        new_kv = _to_tuple_kv(out.past_key_values)
        self._kv_cache[seq.seq_id] = _slice_kv(new_kv, 0)
        self._attn_masks[seq.seq_id] = new_mask

        return tok

    def free_seq(self, seq_id: int) -> None:
        """Release KV cache for a finished or preempted sequence."""
        self._kv_cache.pop(seq_id, None)
        self._attn_masks.pop(seq_id, None)
