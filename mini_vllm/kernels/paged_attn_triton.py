"""
Triton kernel for paged attention — Milestone 4 (GPU step).

This file is only imported when CUDA is available. On CPU environments
it is never touched; paged_attention.py uses the torch reference path instead.

The kernel walks the block table in SRAM, computing attention over
scattered KV blocks without copying them into a contiguous buffer first.
That's what makes paged KV-cache memory-efficient at high batch sizes.
"""
# Implemented in Milestone 4 (GPU path — run on Kaggle/Colab T4).
