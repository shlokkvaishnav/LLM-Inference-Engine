"""
Baseline benchmark: naive HuggingFace model.generate() vs mini-vLLM.

Measures:
  - Single-request latency (time-to-first-token + total)
  - Throughput (tokens/sec) at fixed batch sizes
  - Memory usage peak

Usage:
    python benchmarks/baseline_hf.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --num-prompts 50 \\
        --output-len 128 \\
        --device cuda

Implemented in Milestone 7.
"""
# Milestone 7
