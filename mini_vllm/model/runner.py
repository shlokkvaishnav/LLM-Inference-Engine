"""
ModelRunner: orchestrates forward passes over sequences.

Milestone 1: single-sequence prefill + decode.
Milestone 2: static batching — N sequences as one padded tensor. The key
  insight is that past_key_values flows as a local variable through the decode
  loop, so we never need to slice or merge the cache. Static batching runs
  the full batch until the LAST sequence finishes (earlier finishers stay in
  the batch but their tokens are discarded). That waste is exactly what M3
  (continuous batching) fixes.
Milestones 3-4: the Scheduler replaces generate(); prefill/decode stay.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.sampling.sampler import Sampler

if TYPE_CHECKING:
    from mini_vllm.model.loader import ModelConfig


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

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self, sequences: list[Sequence]
    ) -> tuple[list[int], Any, torch.Tensor]:
        """
        Pack all prompts into one forward pass.

        Variable-length prompts are LEFT-PADDED to max_prompt_len. With left-
        padding, logits[:, -1, :] is always the last *real* token for every
        sequence — no per-sequence index arithmetic. The attention mask (0 =
        padding, 1 = real) prevents padded positions from affecting attention.

        Returns: (first_token_ids, past_key_values, attention_mask)
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
            offset = max_len - len(toks)   # left-pad offset
            input_ids[i, offset:] = torch.tensor(
                toks, dtype=torch.long, device=self.device
            )
            attention_mask[i, offset:] = 1

        with torch.no_grad():
            out = self.model(
                input_ids, attention_mask=attention_mask, use_cache=True
            )

        # left-padding → position -1 is the last real token for every row
        logits = out.logits[:, -1, :]   # (batch, vocab_size)

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

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        sequences: list[Sequence],
        past_kv: Any,
        attention_mask: torch.Tensor,
    ) -> tuple[list[int], Any, torch.Tensor]:
        """
        One decode step over the full batch.

        Feeds one new token per sequence, reuses the batch's KV cache, and
        extends the attention mask by one real column (the new token always
        attends to itself and all real past positions, never to padding).

        past_kv is whatever transformers returned from the previous call —
        we pass it straight back without inspecting its internal format.

        Returns: (next_token_ids, updated_past_kv, updated_attention_mask)
        """
        batch = len(sequences)

        input_ids = torch.tensor(
            [[s.output_token_ids[-1]] for s in sequences],
            dtype=torch.long,
            device=self.device,
        )   # (batch, 1)

        # Grow the attention mask by one real column for the current token.
        new_col = torch.ones(batch, 1, dtype=torch.long, device=self.device)
        full_mask = torch.cat([attention_mask, new_col], dim=1)

        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=past_kv,
                attention_mask=full_mask,
                use_cache=True,
            )

        logits = out.logits[:, -1, :]   # (batch, vocab_size)

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

    # ------------------------------------------------------------------
    # Generate loop  (replaced by Scheduler in M3)
    # ------------------------------------------------------------------

    def generate(self, sequences: list[Sequence]) -> list[Sequence]:
        """
        Prefill → decode until every sequence in the batch finishes.

        Static batching: all sequences run together as a fixed batch. Sequences
        that hit EOS early stay in the tensor (we just stop recording their
        tokens); compute is wasted on them until the slowest sequence finishes.
        That is the exact inefficiency M3 fixes with continuous batching.

        In M3 this method is retired; the Scheduler's step() loop takes over.
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
