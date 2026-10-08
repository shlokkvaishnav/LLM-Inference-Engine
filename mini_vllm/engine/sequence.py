"""Request state machine: everything the engine needs to track for one inference request."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()    # queued, not yet scheduled
    RUNNING = auto()    # in the active batch this step
    FINISHED = auto()   # hit EOS or max_tokens
    PREEMPTED = auto()  # evicted from GPU memory; blocks returned to pool


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1          # -1 = disabled
    max_tokens: int = 256
    stop_token_ids: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)


class Sequence:
    """
    The full mutable state of one request.

    Holds prompt tokens, generated tokens, sampling config, and — from
    Milestone 4 onward — the logical block table the BlockManager assigns.
    """

    _id_counter = 0

    def __init__(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        seq_id: int | None = None,
    ) -> None:
        if seq_id is None:
            seq_id = Sequence._id_counter
            Sequence._id_counter += 1
        self.seq_id = seq_id
        self.prompt_token_ids: list[int] = list(prompt_token_ids)
        self.original_prompt_length = len(self.prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING
        self.finish_reason: str | None = None   # "stop" | "length", set when finished
        # Populated by BlockManager in Milestone 4.
        # Each entry is a physical block ID in the KV-cache pool.
        self.block_table: list[int] = []

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def length(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def generated_token_ids(self) -> list[int]:
        """Every token generated so far, including any folded into the prompt by preemption."""
        return self.all_token_ids[self.original_prompt_length:]

    @property
    def num_total_generated(self) -> int:
        return self.length - self.original_prompt_length

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def reset_for_recompute(self) -> None:
        """
        Preemption recompute: the KV cache is gone, so fold the generated
        tokens into the prompt. The sequence then looks like a fresh request
        with a longer prompt, and the next prefill rebuilds its KV and
        samples the next token. `length` is unchanged.
        """
        self.prompt_token_ids = self.prompt_token_ids + self.output_token_ids
        self.output_token_ids = []

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def check_stop(self) -> bool:
        """Return True if the sequence should stop after the last appended token."""
        if not self.output_token_ids:
            return False
        last = self.output_token_ids[-1]
        if last in self.sampling_params.stop_token_ids:
            self.finish_reason = "stop"
            return True
        if self.num_total_generated >= self.sampling_params.max_tokens:
            self.finish_reason = "length"
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.name}, "
            f"len={self.length})"
        )
