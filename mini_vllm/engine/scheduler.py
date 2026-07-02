"""
Continuous batching scheduler — Milestone 3 (admission) + Milestone 4 (preemption).

The scheduler answers one question every decode step:
  "Given the current WAITING queue and RUNNING batch,
   which sequences run this step, which get preempted, and which are done?"

M3 policy: FCFS admission, no memory accounting — a Scheduler(block_manager=None)
  behaves exactly like the original M3 scheduler (batch-size gating only).

M4 policy: pass a BlockManager and step() also does memory-pressure preemption.
  Two independent phases, each bounded (no thrashing):

  1. Preemption — for every RUNNING sequence (checked LIFO: most-recently
     admitted first), ask "will it have room for the token it's about to
     generate?" If not and no free block exists, evict it: mark PREEMPTED,
     free its blocks, and requeue it at the FRONT of the waiting queue (so
     it resumes before any newer request — same recompute priority vLLM uses).

  2. Admission — try to admit WAITING sequences (FCFS) up to max_batch_size.
     Admission NEVER preempts a running sequence to make room. Without this
     rule, admitting seq B by evicting seq A, then evicting seq B on the next
     iteration to re-admit seq A, could oscillate forever. If the head of the
     waiting queue doesn't fit, later waiters are left blocked too (simple
     head-of-line policy — no queue-jumping in M4).
"""
from __future__ import annotations
from dataclasses import dataclass

from mini_vllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """The scheduler's decision for one decode step."""
    scheduled: list[Sequence]   # sequences that run this step (prefill or decode)
    preempted: list[Sequence]   # sequences evicted this step due to memory pressure
    finished: list[Sequence]    # completed after the previous step (informational)


class Scheduler:
    """
    FCFS continuous batching scheduler, with optional M4 memory-pressure preemption.

    Invariant: len(self.running) <= max_batch_size at all times.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_waiting: int = 1024,
        block_manager: object | None = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_waiting = max_waiting
        self.block_manager = block_manager   # None = M3 behavior, no memory accounting
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
        Decide what runs this step: preempt if necessary, then admit.

        The engine is responsible for:
          - Distinguishing prefill (num_generated_tokens == 0) from decode.
          - Appending tokens and calling free() when a sequence finishes.
        """
        preempted = self._preempt_if_needed()
        self._admit_waiting()

        return SchedulerOutput(
            scheduled=list(self.running),  # snapshot — engine must not mutate this
            preempted=preempted,
            finished=[],                   # engine tracks this; included for completeness
        )

    def _preempt_if_needed(self) -> list[Sequence]:
        """
        M4: evict RUNNING sequences (LIFO — most recently admitted first)
        that won't have room for their next generated token and can't get
        a new block. No-op when block_manager is None (M3 behavior).
        """
        if self.block_manager is None:
            return []

        preempted: list[Sequence] = []
        i = len(self.running) - 1
        while i >= 0:
            seq = self.running[i]
            if not self.block_manager.has_capacity_for_next_token(seq):
                victim = self.running.pop(i)
                victim.status = SequenceStatus.PREEMPTED
                self.block_manager.free(victim)
                self.waiting.insert(0, victim)   # resumes before any newer request
                preempted.append(victim)
            i -= 1
        return preempted

    def _admit_waiting(self) -> None:
        """
        Promote WAITING → RUNNING (FCFS) up to max_batch_size. With a
        block_manager set, a sequence that doesn't currently fit blocks
        admission entirely (head-of-line blocking) rather than preempting
        a running sequence to make room.
        """
        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting[0]

            if self.block_manager is not None:
                if not self.block_manager.can_allocate(seq):
                    break   # doesn't fit yet; leave it (and the rest) waiting
                self.block_manager.allocate(seq)

            self.waiting.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

    def free(self, seq: Sequence) -> None:
        """
        Remove a sequence from the running batch and release its blocks.

        Called by the engine when a sequence finishes (FINISHED). Idempotent
        — safe to call even if the sequence was already removed (e.g. by
        preemption).
        """
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        if self.block_manager is not None:
            self.block_manager.free(seq)

    def has_work(self) -> bool:
        """False only when both queues are empty — engine loop terminates."""
        return bool(self.waiting or self.running)
