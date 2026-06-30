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

*(Filled in at Milestone 7)*

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
- [ ] **M1** Correctness baseline — single-sequence decode, CPU, token-for-token match with HF
- [ ] **M2** Static batching
- [ ] **M3** Continuous in-flight batching *(built interactively)*
- [ ] **M4** Paged KV-cache *(built interactively)*
- [ ] **M5** Quantization — INT8, then INT4; explicit quality/speed tradeoff
- [ ] **M6** OpenAI-compatible API — `/v1/completions` with SSE streaming
- [ ] **M7** Benchmark suite + concurrent load test

---

## Quickstart

```bash
# Install (CPU dev — no GPU needed)
pip install -e ".[dev]"

# Run tests (all skipped until their milestone lands)
pytest tests/ -v

# GPU environment (Kaggle / Colab T4)
# See notebooks/kaggle_gpu_benchmarks.ipynb — fully self-contained
```

---

## Repo layout

```
mini_vllm/
  engine/       scheduler, sequence state machine, output types
  kv_cache/     block manager, per-sequence block tables
  model/        weight loading (swappable), forward pass runner
  kernels/      Triton paged-attention kernel (CUDA only)
  api/          FastAPI server + OpenAI protocol types
  sampling/     greedy / top-k / top-p token sampling
benchmarks/     baseline scripts + load test + results/
tests/          correctness + unit tests (CPU)
notebooks/      Kaggle GPU benchmark notebook
```
