"""
LLMEngine — the continuous batching loop (Milestone 3, batched decode in M4).

Wires the Scheduler (who runs?) to a runner (how?).

The loop:

  while scheduler.has_work():
      output = scheduler.step()          # decide who runs (+ preempt, M4)

      prefill  newly-admitted seqs       # batched — one GPU pass
      decode   each already-running seq  # batched if the runner supports it

      for each finished seq:
          scheduler.free(seq)
          runner.free_seq(seq_id)

Two runner protocols, both supported:
  ModelRunner (M1-M3): decode_one(seq) -> token — one GPU pass PER sequence,
    O(N) passes/step. Detected by the absence of decode_batch.
  PagedLlamaRunner (M4): decode_batch(sequences) -> list[token] — ONE GPU
    pass (per layer) for the WHOLE running batch, O(1) passes/step. This is
    the actual vLLM-style payoff of paged attention: batch size stops
    mattering to how many forward passes a decode step costs.

Pass block_manager to enable M4 preemption in the Scheduler (see M4b);
omit it for plain M3 FCFS-only behavior.
"""
from __future__ import annotations

from typing import Any

from mini_vllm.engine.scheduler import Scheduler, SchedulerOutput
from mini_vllm.engine.sequence import Sequence, SequenceStatus


class LLMEngine:
    """
    Continuous batching engine.

    Usage:
        engine = LLMEngine(runner, max_batch_size=8)
        for seq in sequences:
            engine.add_request(seq)
        engine.run_until_done()
        # seq.output_token_ids now populated for every seq
    """

    def __init__(
        self,
        runner: Any,
        max_batch_size: int = 8,
        block_manager: Any | None = None,
    ) -> None:
        self.runner = runner
        self.scheduler = Scheduler(max_batch_size=max_batch_size, block_manager=block_manager)

    def add_request(self, seq: Sequence) -> None:
        self.scheduler.add_request(seq)

    def step(self) -> SchedulerOutput:
        """
        One engine step: scheduler decides, runner executes.

        Called by run_until_done() in a tight loop, but exposed so callers
        can drive the loop themselves (useful for streaming or testing).
        """
        eos_id = getattr(self.runner.tokenizer, "eos_token_id", None)
        output = self.scheduler.step()

        # Newly admitted sequences (no generated tokens yet) need prefill.
        to_prefill = [s for s in output.scheduled if s.num_generated_tokens == 0]
        # Sequences already in flight need one more decode step.
        to_decode  = [s for s in output.scheduled if s.num_generated_tokens > 0]

        # --- Prefill (batched) ---
        if to_prefill:
            first_tokens = self.runner.prefill_and_store(to_prefill)
            for seq, tok in zip(to_prefill, first_tokens):
                seq.append_token(tok)
                seq.status = SequenceStatus.RUNNING

        # --- Decode: batched (M4, PagedLlamaRunner) or per-sequence (M3) ---
        if to_decode:
            if hasattr(self.runner, "decode_batch"):
                tokens = self.runner.decode_batch(to_decode)
                for seq, tok in zip(to_decode, tokens):
                    seq.append_token(tok)
            else:
                for seq in to_decode:
                    tok = self.runner.decode_one(seq)
                    seq.append_token(tok)

        # --- Retire finished sequences ---
        for seq in list(output.scheduled):
            if seq.check_stop() or (
                eos_id is not None
                and seq.output_token_ids
                and seq.output_token_ids[-1] == eos_id
            ):
                seq.status = SequenceStatus.FINISHED
                self.scheduler.free(seq)
                self.runner.free_seq(seq.seq_id)

        return output

    def run_until_done(self) -> None:
        """Drive step() until every sequence finishes."""
        while self.scheduler.has_work():
            self.step()
