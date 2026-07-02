# mini-vLLM

A from-scratch LLM inference engine implementing the two systems innovations behind [vLLM](https://github.com/vllm-project/vllm):
**continuous (in-flight) batching** and a **block-based, paged KV-cache** —
benchmarked honestly against Hugging Face `transformers` and real vLLM.

Built to understand every line, not just run it.

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│                    API Layer                        │
│          POST /v1/completions  (FastAPI + SSE)      │
└───────────────────────┬────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────┐
│                   LLM Engine                        │
│                                                     │
│   ┌──────────────────┐   ┌─────────────────────┐   │
│   │    Scheduler     │   │   Block Manager      │   │
│   │                  │   │                      │   │
│   │  Continuous      │   │  Paged KV-cache      │   │
│   │  batching:       │   │  allocation:         │   │
│   │  admit / evict   │   │  physical blocks     │   │
│   │  every step      │   │  + block tables      │   │
│   └────────┬─────────┘   └──────────┬───────────┘   │
│            └──────────┬─────────────┘               │
│                ┌──────▼──────┐                      │
│                │ ModelRunner │                      │
│                │             │                      │
│                │  prefill()  │  ← process prompt    │
│                │  decode()   │  ← one step / batch  │
│                └──────┬──────┘                      │
│                       │                             │
│            ┌──────────▼──────────┐                  │
│            │   Model + Weights   │                  │
│            │  TinyLlama-1.1B     │                  │
│            │  (loader.py —       │                  │
│            │   swappable)        │                  │
│            └─────────────────────┘                  │
└────────────────────────────────────────────────────┘
```

**Attention kernel path:**
- CPU (dev / tests): standard PyTorch scaled dot-product attention
- CUDA (Kaggle/Colab T4): custom Triton paged-attention kernel

---

## How it works

### Continuous batching
Standard batching waits for a full batch before running, then releases the
whole batch when the *slowest* sequence finishes — wasting GPU time whenever
sequences differ in length. Continuous batching runs a decode step every tick,
admitting new sequences and retiring finished ones without stalling the rest.
The scheduler decides each step which sequences run, which wait, and which
(under memory pressure) get preempted.

### Paged KV-cache
The KV-cache (the key and value tensors every attention layer accumulates)
grows with sequence length. Allocating a contiguous block per sequence
fragments memory badly and caps batch size. Paged KV-cache treats the cache
like OS virtual memory: a fixed pool of physical blocks, each holding
`block_size` token slots, with a per-sequence block table mapping logical
positions to physical storage. Sequences of wildly different lengths share the
same pool without fragmentation.

---

## Results

### Correctness (M1–M6, Kaggle T4, TinyLlama-1.1B)

Every milestone is verified token-for-token (or, for quantization, within a
calibrated error bound) against a HuggingFace `transformers` ground truth —
28/28 tests passing:

| Suite | Tests | What it proves |
|---|---|---|
| M1–M3 correctness | 3/3 | single/batched/continuous-batch decode matches `model.generate()` exactly |
| M4 BlockManager + Scheduler + paged attention (incl. Triton) | 12/12 | block allocation, LIFO preemption under memory pressure, paged attention kernel matches dense attention on scattered blocks |
| M4 engine integration (`PagedLlamaRunner`, real model) | 3/3 | full paged decode path matches HF, GPU, real weights |
| M5 quantization | 6/6 | INT8/INT4 round-trip + model-level error bounds |
| M6 API server (real model + `PagedLlamaRunner`, concurrency) | 4/4 | streaming == non-streaming, concurrent requests match solo-run baseline |

### Quantization tradeoff (M5, real TinyLlama-1.1B, Kaggle T4)

Weight-only symmetric per-channel quantization (dequantize-on-the-fly before
each matmul — see [`mini_vllm/quantization/quantize.py`](mini_vllm/quantization/quantize.py)
for why this shrinks memory but isn't assumed to speed up compute):

| | Size | Quality (cos-sim vs fp16) | Speed |
|---|---|---|---|
| fp16 | 1937.8 MB | 1.0000 (baseline) | 99.7 tok/s |
| INT8 | 970.5 MB (2x smaller) | 1.0000 | 42.3 tok/s (2.3x slower) |
| INT4 | 486.0 MB (4x smaller) | 0.9785 | 22.2 tok/s (4.4x slower) |

This is the actual, measured tradeoff — not assumed. Naive dequant-on-the-fly
quantization pays a dequantization step per matmul without saving memory
bandwidth during compute, so it's real memory savings but *slower* decode on
this hardware. A genuine speedup needs a fused low-precision GEMM kernel that
never materializes the full-precision weight (structurally similar to the
paged-attention Triton kernel below) — a natural follow-up, not yet built.

### End-to-end throughput vs baselines

*(Filled in at Milestone 7 — naive HF `generate()`, static-batched HF, and a
real vLLM ceiling, all on the same hardware/model/prompts as above)*

| System | Throughput (tok/s) | P50 latency (ms) | P95 latency (ms) | Notes |
|--------|-------------------|-----------------|-----------------|-------|
| HF `generate()` — naive | — | — | — | single request |
| HF `generate()` — batched | — | — | — | static batch |
| **mini-vLLM** | — | — | — | continuous batch + paged KV |
| vLLM (ceiling) | — | — | — | if available |

Hardware: Kaggle T4 (16 GB VRAM) · TinyLlama-1.1B · fp16

---

## Milestones

- [x] Scaffold + repo structure
- [x] **M1** Correctness baseline — single-sequence decode, CPU, token-for-token match with HF
- [x] **M2** Static batching — left-padded prefill, shared batch KV cache
- [x] **M3** Continuous in-flight batching — FCFS scheduler, per-sequence KV cache, O(N) decode
- [x] **M4** Paged KV-cache — BlockManager, LIFO preemption, Triton paged-attention kernel, full engine integration (O(1) batched decode)
- [x] **M5** Quantization — INT8 + INT4, weight-only, measured quality/speed tradeoff (see Results)
- [x] **M6** OpenAI-compatible API — `/v1/completions` with SSE streaming, async continuous-batching engine
- [ ] **M7** Benchmark suite + concurrent load test vs HF/vLLM

---

## Quickstart

```bash
# Install (CPU dev — no GPU needed)
pip install -e ".[dev]"

# Run tests (CPU-safe subset — GPT-2, no GPU required)
pytest tests/ -v
# GPU-only tests (Triton kernel, real-model paged decode) skip cleanly here
# and are verified separately on Kaggle.

# GPU environment (Kaggle / Colab T4)
# See notebooks/kaggle_gpu_benchmarks.ipynb — fully self-contained

# Run the API server locally (defaults to GPT-2/CPU; set MINI_VLLM_MODEL /
# MINI_VLLM_DEVICE for TinyLlama on GPU)
uvicorn mini_vllm.api.server:app --reload
```

---

## Repo layout

```
mini_vllm/
  engine/         scheduler (+ preemption), sequence state machine,
                   LLMEngine (sync) + AsyncLLMEngine (streaming server)
  kv_cache/       block manager, block tables, paged attention
                   (PyTorch reference + Triton kernel)
  model/          weight loading (swappable), dense ModelRunner (M1-M3),
                   PagedLlamaRunner (M4, paged decode for Llama models)
  quantization/   INT8/INT4 weight-only quantization primitives +
                   QuantizedLinear + quantize_model()
  api/            FastAPI server (SSE streaming) + OpenAI protocol types
  sampling/       greedy / top-k / top-p token sampling
benchmarks/       quantization report + baseline/load-test scripts (M7)
tests/            correctness + unit tests (CPU-safe; GPU-only tests skip
                   cleanly without CUDA and are verified on Kaggle)
notebooks/        Kaggle GPU benchmark notebook — fully self-contained
```
