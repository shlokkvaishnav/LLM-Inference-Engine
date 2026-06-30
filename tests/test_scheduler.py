"""Scheduler unit tests — Milestone 3."""
import pytest

from mini_vllm.engine.sequence import Sequence, SamplingParams, SequenceStatus
from mini_vllm.engine.scheduler import Scheduler


@pytest.mark.skip(reason="Milestone 3 — not yet implemented")
def test_admit_when_space_available():
    """Sequences move from WAITING to RUNNING when batch has headroom."""
    ...


@pytest.mark.skip(reason="Milestone 3 — not yet implemented")
def test_no_admit_when_batch_full():
    """Sequences stay WAITING when max_batch_size is saturated."""
    ...


@pytest.mark.skip(reason="Milestone 3 — not yet implemented")
def test_preempt_under_memory_pressure():
    """Scheduler preempts a RUNNING sequence when KV-cache is exhausted."""
    ...


@pytest.mark.skip(reason="Milestone 3 — not yet implemented")
def test_finished_sequences_are_retired():
    """FINISHED sequences are not re-scheduled in the next step."""
    ...
