"""
Concurrent load test: bursty multi-client traffic against the running API
server — Milestone 7.

Ramps concurrency 1 -> 2 -> 4 -> ... -> max_concurrency (each level is a
fresh burst of exactly `concurrency` simultaneous streaming requests, not a
sustained-duration load). Measures per level:
  - P50 / P95 / P99 time-to-first-token (TTFT) latency
  - P50 / P95 / P99 total request latency
  - Throughput: requests/sec and tokens/sec across the whole burst

Token counts come from counting SSE chunks received per request, not from
guessing at decoded text — server.py's streaming path yields exactly one
chunk per generated token (see mini_vllm/engine/async_engine.py), so this
is an exact count, not an approximation.

Requires the server to already be running (this hits real HTTP, not an
in-process ASGI client — that's the point: it measures the actual
concurrency behavior of AsyncLLMEngine's background step loop under load):

    # Start the server first:
    uvicorn mini_vllm.api.server:app --port 8000

    # Then run the load test:
    python benchmarks/load_test.py \\
        --url http://localhost:8000 \\
        --max-concurrency 16 \\
        --max-tokens 32 \\
        --output benchmarks/results/load_test.csv

Writes a CSV to --output, and a PNG chart alongside it (same path, .png
extension) if matplotlib is available — chart is best-effort, CSV is not.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import httpx

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "def fibonacci(n):",
    "The transformer architecture was introduced in",
    "In machine learning, attention mechanisms",
    "The history of the Roman Empire begins",
    "To bake a chocolate cake, you need",
    "Climate change is caused primarily by",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


async def _one_request(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int, model: str) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    start = time.perf_counter()
    ttft: float | None = None
    num_tokens = 0

    async with client.stream("POST", f"{url}/v1/completions", json=payload, timeout=120.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - start
            json.loads(data)   # validates the chunk; content itself isn't needed
            num_tokens += 1

    total = time.perf_counter() - start
    return {"ttft_s": ttft, "total_s": total, "num_tokens": num_tokens}


async def run_level(url: str, model: str, concurrency: int, max_tokens: int) -> dict:
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[
            _one_request(client, url, DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)], max_tokens, model)
            for i in range(concurrency)
        ])
        elapsed = time.perf_counter() - start

    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    totals = [r["total_s"] for r in results]
    total_tokens = sum(r["num_tokens"] for r in results)

    return {
        "concurrency": concurrency,
        "p50_ttft_ms": _percentile(ttfts, 50) * 1000,
        "p95_ttft_ms": _percentile(ttfts, 95) * 1000,
        "p99_ttft_ms": _percentile(ttfts, 99) * 1000,
        "p50_total_ms": _percentile(totals, 50) * 1000,
        "p95_total_ms": _percentile(totals, 95) * 1000,
        "p99_total_ms": _percentile(totals, 99) * 1000,
        "throughput_req_s": concurrency / elapsed,
        "throughput_tok_s": total_tokens / elapsed,
    }


def _concurrency_levels(max_concurrency: int) -> list[int]:
    levels = []
    c = 1
    while c <= max_concurrency:
        levels.append(c)
        c *= 2
    if levels[-1] != max_concurrency:
        levels.append(max_concurrency)
    return levels


async def main_async(args: argparse.Namespace) -> list[dict]:
    # Warm up before measuring: the very first request anywhere pays a
    # one-time cold-start cost (first forward pass, lazy kernel selection)
    # that has nothing to do with concurrency. Without this, concurrency=1
    # — always measured first — absorbs that cost and looks artificially
    # SLOWER than higher concurrency levels, inverting the real signal.
    print("warming up...")
    async with httpx.AsyncClient() as client:
        await _one_request(client, args.url, DEFAULT_PROMPTS[0], args.max_tokens, args.model)

    rows = []
    for c in _concurrency_levels(args.max_concurrency):
        print(f"concurrency={c} ...")
        row = await run_level(args.url, args.model, c, args.max_tokens)
        rows.append(row)
        print(
            f"  p50 ttft={row['p50_ttft_ms']:.0f}ms  p95 ttft={row['p95_ttft_ms']:.0f}ms  "
            f"p99 ttft={row['p99_ttft_ms']:.0f}ms  throughput={row['throughput_tok_s']:.1f} tok/s"
        )
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_chart(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (CSV was still written).")
        return

    concurrencies = [r["concurrency"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(concurrencies, [r["p50_ttft_ms"] for r in rows], marker="o", label="P50")
    ax1.plot(concurrencies, [r["p95_ttft_ms"] for r in rows], marker="o", label="P95")
    ax1.plot(concurrencies, [r["p99_ttft_ms"] for r in rows], marker="o", label="P99")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Time-to-first-token (ms)")
    ax1.set_title("TTFT latency vs concurrency")
    ax1.legend()

    ax2.plot(concurrencies, [r["throughput_tok_s"] for r in rows], marker="o", color="green")
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_title("Throughput vs concurrency")

    fig.tight_layout()
    fig.savefig(path)
    print(f"Wrote chart to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", default="benchmarks/results/load_test.csv")
    args = parser.parse_args()

    rows = asyncio.run(main_async(args))

    csv_path = Path(args.output)
    _write_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    _write_chart(rows, csv_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
