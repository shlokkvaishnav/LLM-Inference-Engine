"""
FastAPI server with SSE streaming — Milestone 6.

Endpoint: POST /v1/completions
  - Non-streaming: returns CompletionResponse JSON
  - Streaming (stream=true): returns text/event-stream of CompletionChunk

The server is a thin adapter: it translates HTTP requests into Sequence
objects, hands them to AsyncLLMEngine, and streams tokens back. No
inference logic lives here — that's entirely M1-M5's job.

Model/device/batching are configured via environment variables so the same
code serves TinyLlama on a Kaggle/cloud GPU or GPT-2 on a laptop CPU for
local testing (mirrors the MINI_VLLM_TEST_MODEL/DEVICE convention already
used by tests/test_correctness.py and benchmarks/quantization_report.py,
just without the _TEST_ infix since this is the actual runtime, not a test):

  MINI_VLLM_MODEL            default: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  MINI_VLLM_DEVICE            default: cuda if available else cpu
  MINI_VLLM_MAX_BATCH_SIZE    default: 8
  MINI_VLLM_NUM_BLOCKS        default: 512   (paged KV-cache pool, Llama+GPU only)
  MINI_VLLM_BLOCK_SIZE        default: 16

Runner selection: PagedLlamaRunner (M4, paged attention) when running a
Llama-family model on CUDA; ModelRunner (M1-M3, dense) otherwise — e.g. for
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
    state["model_name"] = model_name
    yield
    state.clear()


app = FastAPI(title="mini-vLLM", lifespan=lifespan)


def _build_sequence(prompt: str, req: CompletionRequest) -> Sequence:
    tokenizer = state["tokenizer"]
    token_ids = tokenizer.encode(prompt)

    stop_ids: list[int] = []
    if req.stop:
        stop_strs = [req.stop] if isinstance(req.stop, str) else req.stop
        for s in stop_strs:
            stop_ids.extend(tokenizer.encode(s))

    params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
        stop_token_ids=stop_ids,
    )
    return Sequence(token_ids, params)


async def _stream_completion(seq: Sequence, req: CompletionRequest, request_id: str) -> AsyncIterator[str]:
    engine: AsyncLLMEngine = state["engine"]
    tokenizer = state["tokenizer"]
    prev_text = ""

    async for _ in engine.generate(seq):
        # Re-decode the full token list each step rather than decoding just
        # the new token: multi-byte/multi-token unicode characters can span
        # token boundaries, so decoding incrementally token-by-token can
        # produce garbled partial characters. Diffing the re-decoded string
        # against what we already sent is correct at the cost of a little
        # redundant work — cheap relative to the forward pass.
        text = tokenizer.decode(seq.output_token_ids)
        delta = text[len(prev_text):]
        prev_text = text

        finish_reason = "stop" if seq.status == SequenceStatus.FINISHED else None
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
    seq = _build_sequence(req.prompt, req)
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        return StreamingResponse(
            _stream_completion(seq, req, request_id), media_type="text/event-stream"
        )

    async for _ in engine.generate(seq):
        pass

    text = tokenizer.decode(seq.output_token_ids)
    finish_reason = "length" if len(seq.output_token_ids) >= req.max_tokens else "stop"
    return CompletionResponse(
        id=request_id,
        model=req.model,
        choices=[CompletionChoice(text=text, index=0, finish_reason=finish_reason)],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": state.get("model_name")}
