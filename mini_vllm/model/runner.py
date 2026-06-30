"""
ModelRunner: orchestrates forward passes over sequences.

Milestone 1: single-sequence prefill + decode, KV cache in self._kv_cache.
Milestone 2: static batching (N sequences, padded input tensor).
Milestones 3-4: continuous batching; _kv_cache swapped for paged block tables.

Design choice (Option B): the KV cache lives here, not on the Sequence.
Sequence stays a pure data object. At M4, this dict becomes a BlockManager
lookup — nothing in the Sequence abstraction changes.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from mini_vllm.engine.sequence import Sequence, SequenceStatus, SamplingParams
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
        # seq_id → past_key_values tuple returned by the model.
        # Each entry is ((k0, v0), (k1, v1), ...) — one pair per layer.
        # Freed explicitly via self.free() when a sequence finishes.
        self._kv_cache: dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Core forward-pass methods
    # ------------------------------------------------------------------

    def prefill(self, sequences: list[Sequence]) -> list[int]:
        """
        Process full prompts for each sequence.

        For each sequence:
          - Runs a single forward pass over all prompt tokens.
          - Stores the resulting KV cache in self._kv_cache[seq_id].
          - Returns the first generated token (sampled from logits[-1]).

        M1 processes sequences one at a time (no batching yet).
        M2 will batch them together into a single padded tensor.
        """
        next_tokens: list[int] = []

        for seq in sequences:
            input_ids = torch.tensor(
                [seq.prompt_token_ids], dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                out = self.model(input_ids, use_cache=True)

            # out.logits: (1, prompt_len, vocab_size)
            # We only want the last position — the first new token.
            logits = out.logits[:, -1, :]  # (1, vocab_size)

            token_id = Sampler.sample(
                logits,
                temperature=seq.sampling_params.temperature,
                top_p=seq.sampling_params.top_p,
                top_k=seq.sampling_params.top_k,
            ).item()

            self._kv_cache[seq.seq_id] = out.past_key_values
            next_tokens.append(int(token_id))

        return next_tokens

    def decode(self, sequences: list[Sequence]) -> list[int]:
        """
        Single decode step: feed one new token per sequence, get the next.

        For each sequence we:
          - Take the last token it generated as input (shape: (1, 1)).
          - Pass the stored KV cache so the model doesn't reprocess the prompt.
          - Update the cache (it grows by one token-slot per step).
          - Sample and return the next token.

        M1: one sequence at a time.
        M2: batch them (same idea, padded to equal length — trivially 1 here).
        """
        next_tokens: list[int] = []

        for seq in sequences:
            last_token = seq.output_token_ids[-1]
            input_ids = torch.tensor(
                [[last_token]], dtype=torch.long, device=self.device
            )
            past_kv = self._kv_cache[seq.seq_id]

            with torch.no_grad():
                out = self.model(
                    input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            logits = out.logits[:, -1, :]  # (1, vocab_size)

            token_id = Sampler.sample(
                logits,
                temperature=seq.sampling_params.temperature,
                top_p=seq.sampling_params.top_p,
                top_k=seq.sampling_params.top_k,
            ).item()

            self._kv_cache[seq.seq_id] = out.past_key_values
            next_tokens.append(int(token_id))

        return next_tokens

    def free(self, seq_id: int) -> None:
        """Release the KV cache for a finished sequence."""
        self._kv_cache.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Top-level generation loop (used directly until M3 adds a Scheduler)
    # ------------------------------------------------------------------

    def generate(self, sequences: list[Sequence]) -> list[Sequence]:
        """
        Run the full prefill → decode loop for a list of sequences.

        In M3 this loop is replaced by the Scheduler's step() loop.
        Keeping it here for M1/M2 makes the engine testable without
        the scheduler stub raising NotImplementedError.
        """
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        # Prefill: process all prompts, get first tokens.
        first_tokens = self.prefill(sequences)
        for seq, tok in zip(sequences, first_tokens):
            seq.append_token(tok)
            seq.status = SequenceStatus.RUNNING
            if seq.check_stop() or tok == eos_id:
                seq.status = SequenceStatus.FINISHED

        # Decode: step until every sequence is done.
        running = [s for s in sequences if not s.is_finished()]
        while running:
            next_tokens = self.decode(running)
            still_running: list[Sequence] = []
            for seq, tok in zip(running, next_tokens):
                seq.append_token(tok)
                if seq.check_stop() or tok == eos_id:
                    seq.status = SequenceStatus.FINISHED
                    self.free(seq.seq_id)
                else:
                    still_running.append(seq)
            running = still_running

        return sequences
