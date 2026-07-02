"""
Continuous batching scheduler — Milestone 3.

The scheduler answers one question every decode step:
  "Given the current WAITING queue and RUNNING batch,
   which sequences run this step, which are done, and which get freed?"

This is the heart of continuous batching. Unlike static batching (which
assembles a full batch, runs it to completion, then starts the next),
the scheduler runs every step and can:
  - Admit a WAITING sequence the moment a running slot opens up
  - Retire a FINISHED sequence immediately without stalling others

M3 policy: First-Come-First-Served (FCFS), no memory-pressure preemption.
  Preemption is added in M4 when we have block-level memory accounting.

Interview note: the key insight is that step() runs EVERY decode step — it
is called in a tight loop by the engine, not once per batch. That's what
makes "continuous" batching continuous.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from mini_vllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """The scheduler's decision for one decode step."""
    scheduled: list[Sequence]   # sequences that run this step (prefill or decode)
    preempted: list[Sequence]   # evicted due to memory pressure (M4+)
    finished: list[Sequence]    # completed after the previous step (informational)


class Scheduler:
    """
    FCFS continuous batching scheduler.

    State machines:
      add_request → WAITING
      step()      → WAITING sequences promoted to RUNNING (up to max_batch_size)
      free(seq)   → removes from RUNNING (called by engine after seq finishes)

    Invariant: len(self.running) <= max_batch_size at all times.
    """

    def __init__(self, max_batch_size: int, max_waiting: int = 1024) -> None:
        self.max_batch_size = max_batch_size
        self.max_waiting = max_waiting
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add_request(self, seq: Sequence) -> None:
        """Enqueue a new sequence. Called before the engine loop starts."""
        if len(self.waiting) >= self.max_waiting:
            raise RuntimeError(
                f"Waiting queue full ({self.max_waiting}). "
                "Raise max_waiting or apply back-pressure upstream."
            )
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def step(self) -> SchedulerOutput:
        """
        Decide what runs this step.

        1. Admit WAITING sequences into RUNNING (FCFS) until the batch is full.
        2. Return the full running batch as `scheduled`.

        The engine is responsible for:
          - Distinguishing prefill (num_generated_tokens == 0) from decode.
          - Appending tokens and calling free() when a sequence finishes.
          - Calling free() on preempted sequences (M4+).

        M4 will add a memory-budget check here and preempt the youngest
        running sequences when the block pool is exhausted.
        """
        # Admit as many waiting sequences as the batch has room for.
        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting.pop(0)      # FCFS: take the oldest waiter
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

        return SchedulerOutput(
            scheduled=list(self.running),  # snapshot — engine must not mutate this
            preempted=[],                  # M4 will populate this
            finished=[],                   # engine tracks this; included for completeness
        )

    def free(self, seq: Sequence) -> None:
        """
        Remove a sequence from the running batch.

        Called by the engine when a sequence finishes (FINISHED) or is
        preempted (PREEMPTED). After free(), the slot is available and the
        next step() will admit a waiting sequence into it.
        """
        try:
            self.running.remove(seq)
        except ValueError:
            pass   # already removed (idempotent — safe to call twice)

    def has_work(self) -> bool:
        """False only when both queues are empty — engine loop terminates."""
        return bool(self.waiting or self.running)
