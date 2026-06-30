"""
Continuous batching scheduler — Milestone 3 (built interactively).

The scheduler answers one question every decode step:
  "Given the current WAITING queue, RUNNING batch, and KV-cache budget,
   which sequences run this step, which get preempted, and which are done?"

That decision is the heart of continuous batching: instead of waiting for a
full batch to assemble (static batching) or running one sequence at a time,
the scheduler admits new requests mid-flight and retires finished ones without
stalling the rest. We implement this together in Milestone 3.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from mini_vllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """The scheduler's decision for one decode step."""
    scheduled: list[Sequence]   # sequences that run this step
    preempted: list[Sequence]   # sequences kicked out due to memory pressure
    finished: list[Sequence]    # sequences that completed after the last step


class Scheduler:
    """
    Placeholder — implemented in Milestone 3.

    Public contract (nothing above this touches the lists directly):
      add_request(seq)   — enqueue a new sequence
      step() -> SchedulerOutput — select the batch for this decode step
      free(seq)          — called after a sequence fully retires
      has_work() -> bool — False when both queues are empty
    """

    def __init__(self, max_batch_size: int, max_waiting: int = 1024) -> None:
        self.max_batch_size = max_batch_size
        self.max_waiting = max_waiting
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add_request(self, seq: Sequence) -> None:
        raise NotImplementedError("Milestone 3")

    def step(self) -> SchedulerOutput:
        raise NotImplementedError("Milestone 3")

    def free(self, seq: Sequence) -> None:
        raise NotImplementedError("Milestone 3")

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
