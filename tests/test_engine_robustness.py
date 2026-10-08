"""
Engine robustness tests: preemption recompute, abort, loop failure, stop strings.

Uses a tiny deterministic fake runner (no model, no GPU). The next token is a
pure function of all previous tokens, so a preempted-and-recomputed sequence
must produce exactly the same output as one that was never preempted.
"""
import asyncio

import pytest

from mini_vllm.engine.async_engine import AsyncLLMEngine
from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.engine.scheduler import QueueFullError, Scheduler
from mini_vllm.engine.sequence import SamplingParams, Sequence
from mini_vllm.kv_cache.block_manager import BlockManager


class FakeTokenizer:
    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


class FakeRunner:
    """next token = (sum of all tokens so far + 1) % 50 — recompute-safe."""

    tokenizer = FakeTokenizer()

    def __init__(self, fail_on_step=None):
        self.prefills = 0
        self.steps = 0
        self.fail_on_step = fail_on_step

    def _next(self, seq):
        return (sum(seq.all_token_ids) + 1) % 50

    def prefill_and_store(self, seqs):
        self._maybe_fail()
        self.prefills += len(seqs)
        return [self._next(s) for s in seqs]

    def decode_batch(self, seqs):
        self._maybe_fail()
        return [self._next(s) for s in seqs]

    def _maybe_fail(self):
        self.steps += 1
        if self.fail_on_step is not None and self.steps == self.fail_on_step:
            raise RuntimeError("boom")

    def free_seq(self, seq_id):
        pass


def _seqs(n=2, prompt_len=2, max_tokens=8):
    return [
        Sequence([i + 1] * prompt_len, SamplingParams(temperature=0.0, max_tokens=max_tokens))
        for i in range(n)
    ]


def test_preemption_recompute_matches_unconstrained_run():
    # Unconstrained reference.
    ref = _seqs()
    eng = LLMEngine(FakeRunner(), max_batch_size=4)
    for s in ref:
        eng.add_request(s)
    eng.run_until_done()

    # Tight pool: one 10-token sequence fits (5 blocks) but two cannot, so one is preempted.
    seqs = _seqs()
    runner = FakeRunner()
    bm = BlockManager(num_blocks=5, block_size=2)
    eng = LLMEngine(runner, max_batch_size=4, block_manager=bm)
    for s in seqs:
        eng.add_request(s)
    eng.run_until_done()

    assert runner.prefills > len(seqs), "expected at least one re-prefill after preemption"
    for got, want in zip(seqs, ref):
        assert got.generated_token_ids == want.generated_token_ids
        assert got.num_total_generated == 8
        assert got.finish_reason == "length"
    assert bm.num_free_blocks == 5


def test_scheduler_free_removes_waiting_sequence():
    sched = Scheduler(max_batch_size=1)
    a, b = _seqs()
    sched.add_request(a)
    sched.add_request(b)
    sched.step()
    assert sched.waiting == [b]
    sched.free(b)
    assert sched.waiting == []


def test_queue_full_raises_specific_error():
    sched = Scheduler(max_batch_size=1, max_waiting=1)
    a, b = _seqs()
    sched.add_request(a)
    with pytest.raises(QueueFullError):
        sched.add_request(b)


def test_stop_string_spanning_tokens_finishes_with_stop():
    # Tokens are chr(97 + id % 26); a 2-char stop string cannot be matched by a single token id.
    seq = Sequence([1, 1], SamplingParams(temperature=0.0, max_tokens=50))
    runner = FakeRunner()
    probe = Sequence([1, 1], SamplingParams(temperature=0.0, max_tokens=4))
    eng = LLMEngine(runner, max_batch_size=1)
    eng.add_request(probe)
    eng.run_until_done()
    text = runner.tokenizer.decode(probe.generated_token_ids)

    seq.sampling_params.stop_strings = [text[2:4]]
    eng = LLMEngine(FakeRunner(), max_batch_size=1)
    eng.add_request(seq)
    eng.run_until_done()
    assert seq.finish_reason == "stop"
    assert seq.num_total_generated < 50


def test_async_engine_loop_failure_propagates_and_recovers():
    async def run():
        eng = AsyncLLMEngine(FakeRunner(fail_on_step=1), max_batch_size=2)
        bad = _seqs(1)[0]
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in eng.generate(bad):
                pass
        assert not eng.engine.scheduler.has_work()

        good = _seqs(1)[0]   # the loop must still serve later requests
        toks = [t async for t in eng.generate(good)]
        assert len(toks) == 8
        eng._loop_task.cancel()

    asyncio.run(run())


def test_async_abort_while_waiting_removes_from_scheduler():
    async def run():
        eng = AsyncLLMEngine(FakeRunner(), max_batch_size=1)
        a, b = _seqs(2, max_tokens=50)
        eng.submit(a)
        eng.submit(b)
        gen = eng.generate(b)
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task   # generator's finally block runs _abort(b)
        assert b not in eng.engine.scheduler.waiting
        assert b not in eng.engine.scheduler.running
        eng._abort(a)
        eng._loop_task.cancel()

    asyncio.run(run())
