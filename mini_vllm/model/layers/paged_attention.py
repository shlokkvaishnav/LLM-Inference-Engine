"""
Paged attention forward pass — Milestone 4.

CPU path: reference implementation using standard torch operations.
  Q attends over scattered KV blocks via gather → standard scaled dot-product attention.

CUDA path: delegates to the Triton kernel in mini_vllm/kernels/paged_attn_triton.py,
  which walks the block table without materializing a contiguous KV tensor.

The kernel is chosen at runtime based on whether CUDA is available.
"""
# Implemented in Milestone 4.
