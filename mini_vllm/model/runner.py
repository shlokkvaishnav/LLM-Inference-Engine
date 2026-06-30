"""
ModelRunner: orchestrates the forward pass over a batch of sequences.

Milestone 1: single-sequence forward pass with manual KV-cache management
Milestone 2: static batching (N sequences, fixed batch size)
Milestones 3-4: continuous batching with paged KV-cache integration

The runner is the bridge between the scheduler's sequence list and the
actual model weights. It knows about tensors; the scheduler knows about
sequences. Neither knows about the other's internals.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mini_vllm.model.loader import ModelConfig
    from mini_vllm.engine.sequence import Sequence


class ModelRunner:
    """
    Two public methods — that's the entire interface the engine uses:

      prefill(sequences) -> list[int]
          Process full prompts, populate KV-cache entries, return first token.

      decode(sequences) -> list[int]
          Single decode step over the running batch; returns next token per seq.
    """

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

    def prefill(self, sequences: list["Sequence"]) -> list[int]:
        """Implemented in Milestone 1."""
        raise NotImplementedError("Milestone 1")

    def decode(self, sequences: list["Sequence"]) -> list[int]:
        """Implemented in Milestone 1."""
        raise NotImplementedError("Milestone 1")
