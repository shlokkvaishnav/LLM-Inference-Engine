"""
LLMEngine — the continuous batching loop (Milestone 3).

Wires the Scheduler (who runs?) to the ModelRunner (how?).

The loop:

  while scheduler.has_work():
      output = scheduler.step()          # decide who runs

      prefill  newly-admitted seqs       # batched — one GPU pass
      decode   each already-running seq  # one GPU pass each (M4 batches these)

      for each finished seq:
          scheduler.free(seq)
          runner.free_seq(seq_id)

Design note — why decode_one instead of batched decode:
  Continuous batching admits sequences at different times, so their KV caches
  have different sequence lengths.  Batching them requires padding + a mask
  that spans heterogeneous lengths, which needs either careful bookkeeping
  (possible but fiddly) or paged attention (M4).  decode_one keeps M3 simple
  and correct at the cost of O(N) GPU passes per step.  M4 collapses that to
  O(1) passes regardless of batch size — which is the real vLLM innovation.
"""
from __future__ import annotations

from mini_vllm.engine.scheduler import Scheduler, SchedulerOutput
from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.model.runner import ModelRunner


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

    def __init__(self, runner: ModelRunner, max_batch_size: int = 8) -> None:
        self.runner = runner
        self.scheduler = Scheduler(max_batch_size=max_batch_size)

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

        # --- Decode (one pass per sequence until M4) ---
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
