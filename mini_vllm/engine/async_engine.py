"""
AsyncLLMEngine — async wrapper around LLMEngine for the API server (M6).

LLMEngine.run_until_done() is a blocking, synchronous loop: call step()
until every sequence finishes. That's fine for offline batch generation
(what M1-M5's tests do) but wrong for a server, where requests arrive at
arbitrary times and each needs its own token-by-token stream.

AsyncLLMEngine runs ONE background asyncio task that repeatedly calls
engine.step() — the same continuous-batching step from M3/M4 — and after
each step, pushes the token each scheduled sequence just generated into
that sequence's own asyncio.Queue. A request's async generator just reads
from its queue; it doesn't know or care that other requests are sharing
the same batch. This is what makes "continuous batching" visible at the
API layer: a request submitted while others are mid-generation gets
admitted into the very next step, not queued behind them.

Single-threaded by design: everything here runs on one asyncio event loop.
Coroutines only yield at `await` points, so calling engine.add_request()
or engine.step() (both plain sync methods) from a coroutine is safe without
locks — nothing else can run in between.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.engine.sequence import Sequence, SequenceStatus


class AsyncLLMEngine:
    def __init__(
        self,
        runner: Any,
        max_batch_size: int = 8,
        block_manager: Any | None = None,
    ) -> None:
        self.engine = LLMEngine(runner, max_batch_size=max_batch_size, block_manager=block_manager)
        self._queues: dict[int, asyncio.Queue] = {}
        self._new_work = asyncio.Event()
        self._loop_task: asyncio.Task | None = None

    def start(self) -> None:
        """Launch the background step loop. Idempotent — safe to call per-request."""
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        while True:
            if not self.engine.scheduler.has_work():
                self._new_work.clear()
                await self._new_work.wait()
                continue

            output = self.engine.step()

            # Every sequence in output.scheduled generated exactly one new
            # token this step, whether via prefill or decode (see
            # LLMEngine.step()) — push it to that sequence's stream.
            for seq in output.scheduled:
                queue = self._queues.get(seq.seq_id)
                if queue is None:
                    continue
                if seq.output_token_ids:
                    queue.put_nowait(seq.output_token_ids[-1])
                if seq.status == SequenceStatus.FINISHED:
                    queue.put_nowait(None)   # sentinel: stream is done

            # Yield control so other coroutines (new requests arriving,
            # queue consumers) get a turn between steps.
            await asyncio.sleep(0)

    async def generate(self, seq: Sequence) -> AsyncIterator[int]:
        """
        Submit a sequence and yield its token ids as they're generated,
        one per decode step, until it finishes.

        On early exit (caller stops iterating, e.g. an HTTP client
        disconnects mid-stream) the finally block aborts the sequence —
        frees its blocks and removes it from the scheduler — so a dropped
        client doesn't leave a zombie sequence permanently occupying a
        batch slot.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[seq.seq_id] = queue
        self.engine.add_request(seq)
        self._new_work.set()
        self.start()

        try:
            while True:
                tok = await queue.get()
                if tok is None:
                    break
                yield tok
        finally:
            self._abort(seq)

    def _abort(self, seq: Sequence) -> None:
        """Idempotent — safe even if the sequence already finished normally."""
        self.engine.scheduler.free(seq)
        self.engine.runner.free_seq(seq.seq_id)
        self._queues.pop(seq.seq_id, None)
