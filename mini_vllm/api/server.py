"""
FastAPI server with SSE streaming.

Endpoint: POST /v1/completions
  - Non-streaming: returns CompletionResponse JSON
  - Streaming (stream=true): returns text/event-stream of CompletionChunk

The server is a thin adapter: it translates HTTP requests into Sequence
objects, hands them to AsyncLLMEngine, and streams tokens back. No
inference logic lives here — that's entirely the engine's job.

Model/device/batching are configured via environment variables so the same
code serves TinyLlama on a Kaggle/cloud GPU or GPT-2 on a laptop CPU for
local testing (the correctness tests and quantization report read the same
settings with a _TEST_ infix, e.g. MINI_VLLM_TEST_MODEL):

  MINI_VLLM_MODEL            default: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  MINI_VLLM_DEVICE            default: cuda if available else cpu
  MINI_VLLM_MAX_BATCH_SIZE    default: 8
  MINI_VLLM_NUM_BLOCKS        default: 512   (paged KV-cache pool, Llama+GPU only)
  MINI_VLLM_BLOCK_SIZE        default: 16

Runner selection: PagedLlamaRunner (paged attention) when running a
Llama-family model on CUDA; ModelRunner (dense) otherwise — e.g. for
GPT-2 or CPU-only deployments, where PagedLlamaRunner's Llama-specific
layer access wouldn't apply anyway.

Read lazily inside the lifespan handler (not at module import time) so
tests can override env vars right up until the TestClient triggers startup.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from mini_vllm.api.protocol import (
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
)
from mini_vllm.engine.async_engine import AsyncLLMEngine
from mini_vllm.engine.scheduler import QueueFullError
from mini_vllm.engine.sequence import SamplingParams, Sequence, SequenceStatus
from mini_vllm.kv_cache.block_manager import BlockManager
from mini_vllm.model.loader import ModelConfig, load_model
from mini_vllm.model.paged_llama_runner import PagedLlamaRunner
from mini_vllm.model.runner import ModelRunner

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.environ.get("MINI_VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device = os.environ.get("MINI_VLLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = "float16" if device == "cuda" else "float32"
    max_batch_size = int(os.environ.get("MINI_VLLM_MAX_BATCH_SIZE", "8"))
    num_blocks = int(os.environ.get("MINI_VLLM_NUM_BLOCKS", "512"))
    block_size = int(os.environ.get("MINI_VLLM_BLOCK_SIZE", "16"))

    config = ModelConfig(model_name_or_path=model_name, dtype=dtype, device=device, max_model_len=2048)
    model, tokenizer = load_model(config)
    model.eval()

    if device == "cuda" and getattr(model.config, "model_type", "") == "llama":
        block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        runner = PagedLlamaRunner(model, tokenizer, block_manager, dtype=torch.float16, device=device)
    else:
        block_manager = None
        runner = ModelRunner(model, tokenizer, config)

    state["engine"] = AsyncLLMEngine(runner, max_batch_size=max_batch_size, block_manager=block_manager)
    state["tokenizer"] = tokenizer
    # Hard cap on prompt + generated tokens: the model context, and (paged
    # runner) the whole KV pool — a request that can never fit would block
    # the waiting queue forever.
    state["max_seq_tokens"] = min(
        config.max_model_len, num_blocks * block_size if block_manager else config.max_model_len
    )
    state["model_name"] = model_name
    yield
    state.clear()


app = FastAPI(title="mini-vLLM", lifespan=lifespan)


def _build_sequence(prompt: str, req: CompletionRequest) -> Sequence:
    tokenizer = state["tokenizer"]
    token_ids = tokenizer.encode(prompt)

    # Stop strings are matched on decoded text by the engine (a stop string
    # can span several tokens, so token-id matching would be wrong).
    stop_strs: list[str] = []
    if req.stop:
        stop_strs = [req.stop] if isinstance(req.stop, str) else list(req.stop)
        stop_strs = [s for s in stop_strs if s]

    params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
        stop_strings=stop_strs,
    )
    return Sequence(token_ids, params)


def _apply_stop(text: str, stops: list[str], final: bool) -> tuple[str, bool]:
    """
    Truncate `text` at the earliest stop string. Returns (text, hit_stop).
    While streaming (final=False) also withholds a trailing partial match of
    a stop string, so "ST" isn't sent before "STOP" completes.
    """
    cut = min((i for i in (text.find(st) for st in stops) if i != -1), default=-1)
    if cut != -1:
        return text[:cut], True
    if not final:
        hold = 0
        for st in stops:
            for k in range(min(len(st) - 1, len(text)), 0, -1):
                if text.endswith(st[:k]):
                    hold = max(hold, k)
                    break
        if hold:
            return text[:-hold], False
    return text, False


def _check_fits(prompt_len: int, max_tokens: int) -> None:
    limit = state["max_seq_tokens"]
    if prompt_len + max_tokens > limit:
        raise HTTPException(
            status_code=400,
            detail=f"prompt ({prompt_len} tokens) + max_tokens ({max_tokens}) "
                   f"exceeds the maximum sequence length of {limit}.",
        )


async def _stream_completion(seq: Sequence, req: CompletionRequest, request_id: str) -> AsyncIterator[str]:
    engine: AsyncLLMEngine = state["engine"]
    tokenizer = state["tokenizer"]
    stops = seq.sampling_params.stop_strings
    prev_text = ""
    count = 0

    async for _ in engine.generate(seq):
        count += 1
        # Re-decode the full token list each step rather than decoding just
        # the new token: multi-byte/multi-token unicode characters can span
        # token boundaries, so decoding incrementally token-by-token can
        # produce garbled partial characters. Diffing the re-decoded string
        # against what we already sent is correct at the cost of a little
        # redundant work — cheap relative to the forward pass.
        # Only the last token's chunk carries finish_reason: the sequence can
        # already be FINISHED while a slow client is still draining earlier tokens.
        is_last = seq.status == SequenceStatus.FINISHED and count >= seq.num_total_generated
        text = tokenizer.decode(seq.generated_token_ids, skip_special_tokens=True)
        hit_stop = False
        if stops:
            text, hit_stop = _apply_stop(text, stops, final=is_last)
        delta = text[len(prev_text):]
        prev_text = text

        finish_reason = (seq.finish_reason or "stop") if is_last else None
        if hit_stop:
            finish_reason = "stop"
        chunk = CompletionChunk(
            id=request_id,
            model=req.model,
            choices=[CompletionChoice(text=delta, index=0, finish_reason=finish_reason)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/completions", response_model=None)
async def completions(req: CompletionRequest):
    if isinstance(req.prompt, list):
        raise HTTPException(
            status_code=400,
            detail="Batch prompts in a single request aren't supported yet — "
                   "send one prompt per request (continuous batching still "
                   "interleaves multiple concurrent requests efficiently).",
        )

    engine: AsyncLLMEngine = state["engine"]
    tokenizer = state["tokenizer"]
    # Check the string, not the token ids: Llama tokenizers prepend BOS, so an
    # empty prompt still encodes to one token.
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty.")
    seq = _build_sequence(req.prompt, req)
    _check_fits(seq.prompt_length, req.max_tokens)
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    try:
        engine.submit(seq)
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if req.stream:
        return StreamingResponse(
            _stream_completion(seq, req, request_id), media_type="text/event-stream"
        )

    async for _ in engine.generate(seq):
        pass

    text = tokenizer.decode(seq.generated_token_ids, skip_special_tokens=True)
    finish_reason = seq.finish_reason or "stop"
    stops = seq.sampling_params.stop_strings
    if stops:
        text, hit_stop = _apply_stop(text, stops, final=True)
        if hit_stop:
            finish_reason = "stop"
    return CompletionResponse(
        id=request_id,
        model=req.model,
        choices=[CompletionChoice(text=text, index=0, finish_reason=finish_reason)],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": state.get("model_name")}
