"""
Concurrent load test: bursty multi-client traffic against the API server.

Ramps concurrency from 1 → N clients, measures:
  - P50 / P95 / P99 time-to-first-token latency
  - P50 / P95 / P99 total request latency
  - Throughput (requests/sec, tokens/sec) vs batch occupancy

Produces a CSV + matplotlib chart saved to benchmarks/results/.

Usage:
    # Start the server first:
    uvicorn mini_vllm.api.server:app --port 8000

    # Then run the load test:
    python benchmarks/load_test.py \\
        --url http://localhost:8000 \\
        --max-concurrency 32 \\
        --duration 60 \\
        --output benchmarks/results/load_test.csv

Implemented in Milestone 7.
"""
# Milestone 7
