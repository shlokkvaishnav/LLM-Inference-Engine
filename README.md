# mini-vLLM

[![CI](https://github.com/shlokkvaishnav/LLM-Inference-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokkvaishnav/LLM-Inference-Engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

A from-scratch LLM inference engine implementing the two systems ideas behind
[vLLM](https://github.com/vllm-project/vllm): **continuous (in-flight) batching** and a
**block-based, paged KV-cache**. It serves an OpenAI-compatible `/v1/completions` API and is
verified token-for-token against Hugging Face `transformers`.

## Features

- **Continuous batching** — a decode step every tick; new requests join and finished ones leave
  without stalling the rest of the batch.
- **Paged KV-cache** — a fixed pool of physical blocks with per-sequence block tables, LIFO
  preemption (recompute) under memory pressure, and a Triton paged-attention kernel on CUDA
  (PyTorch reference on CPU).
- **Weight-only quantization** — INT8 and INT4 per-channel quantization with a measured
  size / quality / speed tradeoff.
- **OpenAI-compatible API** — FastAPI server with SSE streaming, stop strings, request
  validation and back-pressure.
- **Benchmarks** — baseline comparison against Hugging Face `generate()` and a concurrent
  load test against the running server.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        API layer                          │
│          POST /v1/completions  (FastAPI + SSE)            │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│  AsyncLLMEngine — one background task steps the engine    │
│  and fans tokens out to a per-request stream              │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│                         LLMEngine                         │
│                                                           │
│   ┌──────────────────┐      ┌───────────────────────┐    │
│   │    Scheduler     │◄────►│     Block Manager      │    │
│   │ admit / preempt  │      │ physical block pool +  │    │
│   │ every step       │      │ per-sequence tables    │    │
│   └────────┬─────────┘      └───────────────────────┘    │
│            │                                              │
│   ┌────────▼───────────────────────────────────────┐    │
│   │ Runner                                          │    │
│   │   ModelRunner       dense KV cache (CPU / any)  │    │
│   │   PagedLlamaRunner  paged KV + Triton kernel    │    │
│   │                     (Llama models on CUDA)      │    │
│   └────────┬───────────────────────────────────────┘    │
│            │                                              │
│   ┌────────▼────────────────────┐                        │
│   │ Model + weights (loader.py) │  TinyLlama-1.1B, GPT-2 │
│   └─────────────────────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

**Attention kernel path**
- CPU (development and tests): standard PyTorch scaled dot-product attention.
- CUDA: custom Triton paged-attention kernel (`PagedLlamaRunner`).

## How it works

### Continuous batching
Standard batching waits for a full batch before running, then releases the whole batch
when the *slowest* sequence finishes — wasting GPU time whenever sequences differ in length.
Continuous batching runs a decode step every tick, admitting new sequences and retiring
finished ones without stalling the rest. The scheduler decides each step which sequences
run, which wait, and which (under memory pressure) get preempted. A preempted sequence has
its blocks freed and is re-prefilled from its prompt plus the tokens generated so far when
it is admitted again.

### Paged KV-cache
The KV-cache (the key and value tensors every attention layer accumulates) grows with
sequence length. Allocating a contiguous buffer per sequence fragments memory badly and caps
batch size. Paged KV-cache treats the cache like OS virtual memory: a fixed pool of physical
blocks, each holding `block_size` token slots, with a per-sequence block table mapping logical
positions to physical storage. Sequences of wildly different lengths share the same pool
without fragmentation.

## Quickstart

```bash
# Install (CPU development — no GPU needed)
pip install -e ".[dev]"      # add ".[bench]" for the load-test chart, ".[gpu]" for Triton

# Run the tests (CPU-safe subset; GPT-2 and a tiny random Llama)
pytest tests/ -v

# Run the API server (defaults to TinyLlama on CUDA if available, else CPU)
MINI_VLLM_MODEL=gpt2 MINI_VLLM_DEVICE=cpu uvicorn mini_vllm.api.server:app
```

### Configuration

The server reads these environment variables at startup:

| Variable | Default | Meaning |
|---|---|---|
| `MINI_VLLM_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Hugging Face model name or local path |
| `MINI_VLLM_DEVICE` | `cuda` if available, else `cpu` | Device to run on |
| `MINI_VLLM_MAX_BATCH_SIZE` | `8` | Maximum sequences decoded together |
| `MINI_VLLM_NUM_BLOCKS` | `512` | KV-cache blocks in the pool (paged runner only) |
| `MINI_VLLM_BLOCK_SIZE` | `16` | Token slots per block (paged runner only) |

The paged runner is used for Llama-family models on CUDA; everything else uses the dense
runner. Prompt length plus `max_tokens` is capped at 2048 tokens and at the KV pool size.

### API

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

Set `"stream": true` for server-sent events. Supported fields: `max_tokens`, `temperature`,
`top_p`, `top_k`, `stop` (string or list), `stream`. Errors: `400` for an empty prompt, a prompt
that cannot fit, or a list of prompts; `422` for out-of-range parameters; `503` when the waiting
queue is full. `GET /health` reports the loaded model.

### Docker

```bash
docker compose up api        # GPU (needs the NVIDIA Container Toolkit), port 8000
docker compose up api-cpu    # CPU only, port 8001
```

## Results

All GPU numbers are from a Kaggle T4 (16 GB VRAM) with TinyLlama-1.1B in fp16.

### Correctness

Decode paths are verified token-for-token (or, for quantization, within a calibrated error
bound) against a Hugging Face `transformers` ground truth. **38/38 tests pass on the GPU**
with none skipped (last run: 8 Oct 2026). On CPU the same suite gives 36 passed and 2 skipped;
the two skipped tests (Triton kernel, real-model paged decode) need CUDA.

| Suite | Tests | What it proves |
|---|---|---|
| Dense runner correctness | 3/3 | single, batched and continuous-batch decode match `model.generate()` exactly |
| Block manager, scheduler, paged attention (incl. Triton) | 12/12 | block allocation, LIFO preemption under memory pressure, paged attention matches dense attention on scattered blocks |
| Paged runner (`PagedLlamaRunner`, real model) | 3/3 | full paged decode path matches HF on GPU with real weights |
| Quantization | 6/6 | INT8/INT4 round-trip and model-level error bounds |
| API server (real model + `PagedLlamaRunner`) | 8/8 | streaming equals non-streaming, concurrent requests match solo runs, bad parameters / oversized / empty prompts rejected, stop strings truncate output |
| Engine robustness | 6/6 | preemption recompute matches an unconstrained run, abort while waiting, engine-loop failure propagates and recovers, stop strings spanning several tokens |

### Quantization tradeoff

Weight-only symmetric per-channel quantization (dequantize on the fly before each matmul —
see [`mini_vllm/quantization/quantize.py`](mini_vllm/quantization/quantize.py) for why this
shrinks memory but does not speed up compute):

| | Size | Quality (cos-sim vs fp16) | Speed |
|---|---|---|---|
| fp16 | 1937.8 MB | 1.0000 (baseline) | 99.7 tok/s |
| INT8 | 970.5 MB (2x smaller) | 1.0000 | 42.3 tok/s (2.3x slower) |
| INT4 | 486.0 MB (4x smaller) | 0.9785 | 22.2 tok/s (4.4x slower) |

Naive dequantize-on-the-fly pays a dequantization step per matmul without saving memory
bandwidth during compute, so the memory saving is real but decode is *slower*. A genuine
speedup needs a fused low-precision GEMM kernel that never materializes the full-precision
weight, which is not implemented.

### End-to-end throughput vs baselines

`benchmarks/baseline_hf.py` — 8 prompts, output length 32:

| System | Throughput (tok/s) | Wall-clock (s) | Peak GPU mem | Notes |
|--------|--------------------|-----------------|--------------|-------|
| HF `generate()` — naive | 39.2 | 6.53 | 2212 MB | sequential, one prompt at a time |
| HF `generate()` — batched | 285.8 | 0.90 | 2223 MB | single call, all 8 prompts batched |
| **mini-vLLM** | 72.6 | 3.52 | 2463 MB | continuous batch + paged KV, `max_batch_size=4` |
| vLLM | — | — | — | opt-in (`--compare-vllm`), not run |

mini-vLLM is slower than batched HF here. These are hypotheses from reading the code, not
profiled results:
1. **The comparison is not apples-to-apples**: this run used `max_batch_size=4` against 8
   prompts, so mini-vLLM only gets half the parallelism of HF's single batched call.
2. **Per-step Python overhead**: `PagedLlamaRunner.decode_batch` loops over Llama's 22 layers
   in Python and, within each layer, loops over sequences to write K/V into the block pool.
   HF's `generate()` runs one fused attention call per layer.
3. **GQA expansion over the whole pool**: TinyLlama has 4 KV heads and 32 query heads, and the
   runner `repeat_interleave`s the *entire* K/V pool (not just the blocks in use) up to 32
   heads for every layer on every step (`paged_llama_runner.py`, `_paged_decode_layer`). This
   is likely the largest single cost.
4. **A GPU sync per sequence**: sampling calls `.item()` on each sequence's token.
5. **Possible Triton recompiles**: the kernel takes `MAX_NUM_BLOCKS` as a `constexpr`, so it
   may recompile each time the longest sequence crosses a block boundary.

### Concurrent load test

`benchmarks/load_test.py` against the running server, ramped concurrency:

| Concurrency | P50 TTFT (ms) | P95 TTFT (ms) | P99 TTFT (ms) | Throughput (tok/s) |
|---|---|---|---|---|
| 1 | 58 | 58 | 58 | 40.1 |
| 2 | 105 | 117 | 118 | 64.8 |
| 4 | 138 | 163 | 167 | 108.4 |
| 8 | 208 | 240 | 240 | 198.7 |
| 16 | 768 | 1420 | 1421 | 211.0 |

Throughput keeps climbing through concurrency 16, while time-to-first-token stays low up to 8
(the server's default `max_batch_size`) and then rises as requests queue — continuous batching
under varying concurrent load, a more realistic measure than one fixed static batch.

Benchmark CSV, PNG and JSON outputs are gitignored; regenerate them with the scripts in
`benchmarks/` or the Kaggle notebook.

## Testing

```bash
pytest tests/ -v                       # CPU subset
```

- GPU-only tests (Triton kernel, real-model paged decode) skip cleanly without CUDA.
- `MINI_VLLM_TEST_MODEL` / `MINI_VLLM_TEST_DEVICE` select the model and device for the
  correctness tests; the API tests read `MINI_VLLM_MODEL` / `MINI_VLLM_DEVICE`, the same
  variables the server uses.
- `transformers` is supported from 4.45; CI runs the latest 5.x on CPU and the Kaggle GPU run
  pins 4.46.3. The notebook [`notebooks/kaggle_gpu_benchmarks.ipynb`](notebooks/kaggle_gpu_benchmarks.ipynb)
  runs the full suite and the benchmarks on a Kaggle T4 end to end.

## Repo layout

```
mini_vllm/
  engine/         scheduler (+ preemption), sequence state machine,
                   LLMEngine (sync) + AsyncLLMEngine (streaming server)
  kv_cache/       block manager, paged attention
                   (PyTorch reference + Triton kernel)
  model/          weight loading (swappable), dense ModelRunner,
                   PagedLlamaRunner (paged decode for Llama models)
  quantization/   INT8/INT4 weight-only quantization primitives +
                   QuantizedLinear + quantize_model()
  api/            FastAPI server (SSE streaming) + OpenAI protocol types
  sampling/       greedy / top-k / top-p token sampling
benchmarks/       quantization report, baseline comparison, load test
tests/            correctness and unit tests (CPU-safe; GPU-only tests skip
                   without CUDA)
notebooks/        Kaggle GPU notebook: tests + benchmarks, self-contained
Dockerfile        GPU image          docker-compose.yml   api / api-cpu services
Dockerfile.cpu    CPU image
```

## Limitations

- Prefix caching is not implemented (block `ref_count` is reserved for it).
- The paged runner supports Llama-family models on CUDA only; other models use the dense runner.
- Quantization saves memory but is slower than fp16 (see above).
- A single request takes one prompt; list prompts are rejected. There is no chat endpoint,
  token-usage reporting, authentication or rate limiting.
- The server is a single process holding one model; running several uvicorn workers loads
  one copy per worker.

## License

MIT — see [LICENSE](LICENSE).
