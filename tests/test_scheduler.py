"""Scheduler unit tests — Milestone 3 (admission) + Milestone 4 (preemption).

All pure Python: Scheduler + Sequence + BlockManager, no model/GPU needed.
"""
from mini_vllm.engine.sequence import Sequence, SamplingParams, SequenceStatus
from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.kv_cache.block_manager import BlockManager


def _make_seq(num_prompt_tokens: int) -> Sequence:
    return Sequence(list(range(num_prompt_tokens)), SamplingParams())


def test_admit_when_space_available():
    """Sequences move from WAITING to RUNNING when batch has headroom."""
    sched = Scheduler(max_batch_size=2)
    seq1, seq2 = _make_seq(3), _make_seq(3)
    sched.add_request(seq1)
    sched.add_request(seq2)

    output = sched.step()

    assert output.scheduled == [seq1, seq2]
    assert sched.waiting == []
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING


def test_no_admit_when_batch_full():
    """Sequences stay WAITING when max_batch_size is saturated."""
    sched = Scheduler(max_batch_size=1)
    seq1, seq2 = _make_seq(3), _make_seq(3)
    sched.add_request(seq1)
    sched.add_request(seq2)

    output = sched.step()

    assert output.scheduled == [seq1]
    assert sched.waiting == [seq2]
    assert seq2.status == SequenceStatus.WAITING


def test_finished_sequences_are_retired():
    """FINISHED sequences are not re-scheduled in the next step."""
    sched = Scheduler(max_batch_size=2)
    seq1, seq2 = _make_seq(3), _make_seq(3)
    sched.add_request(seq1)
    sched.add_request(seq2)
    sched.step()   # both admitted

    seq1.status = SequenceStatus.FINISHED
    sched.free(seq1)

    output = sched.step()
    assert output.scheduled == [seq2]


def test_preempt_under_memory_pressure():
    """
    Scheduler preempts a RUNNING sequence when its next token would need a
    new block and the pool is exhausted.

    Setup: 2 blocks total, block_size=4.
      seq1: 4 prompt tokens -> 1 block, exactly at capacity.
      seq2: 1 prompt token  -> 1 block, plenty of headroom.
    Both fit and get admitted (uses all 2 blocks).

    After a simulated decode round, seq1 has grown to 5 tokens (needs a 2nd
    block) but seq2 has only grown to 2 tokens (still fits in its 1 block).
    No blocks are free -> seq1 (checked LIFO, so seq2 first, then seq1) must
    be preempted since it's the one that actually needs a new block.
    """
    bm = BlockManager(num_blocks=2, block_size=4)
    sched = Scheduler(max_batch_size=2, block_manager=bm)

    seq1 = _make_seq(4)   # will need a 2nd block after 1 more token
    seq2 = _make_seq(1)   # has headroom for several more tokens

    sched.add_request(seq1)
    sched.add_request(seq2)
    output = sched.step()
    assert output.scheduled == [seq1, seq2]
    assert bm.num_free_blocks == 0   # pool fully consumed by the two 1-block allocations

    # Simulate one decode round: both sequences generate a token.
    seq1.append_token(999)   # length 4 -> 5, exceeds its 1-block (4 token) capacity
    seq2.append_token(999)   # length 1 -> 2, still within its 1-block capacity

    output = sched.step()

    assert seq1 in output.preempted
    assert seq1.status == SequenceStatus.PREEMPTED
    assert seq1.block_table == []          # blocks fully released
    assert seq2 in output.scheduled        # seq2 had room, stays running
    assert seq1 not in sched.running
    # seq1 needs 2 blocks now (ceil(5/4)) but only 1 is free (seq2 holds the
    # other) -> it doesn't fit yet, stays at the front of the waiting queue.
    assert sched.waiting[0] is seq1
