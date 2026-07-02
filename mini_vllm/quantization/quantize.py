"""
Weight-only quantization primitives — Milestone 5.

Scheme: symmetric, per-output-channel, round-to-nearest (RTN). For a weight
matrix of shape (out_features, in_features), each ROW gets its own scale —
per-channel is more accurate than a single tensor-wide scale because
different output channels can have very different weight magnitudes; a
shared scale would waste precision on the small-magnitude channels.

INT8: values in [-127, 127] (not the full int8 range [-128,127] — keeping
  it symmetric around zero avoids a special-cased asymmetric zero-point,
  at the cost of one representable level. Standard tradeoff for weight-only
  PTQ where zero-point complexity isn't worth it.)

INT4: values in [-7, 7], packed two per byte (a torch.uint8 tensor is used
  as the storage container since there's no native 4-bit dtype). This is
  what actually delivers the ~8x-vs-fp32 / ~4x-vs-fp16 size reduction —
  INT8 alone only gets to 4x/2x.

IMPORTANT — what this does and doesn't buy you:
  This is WEIGHT-ONLY quantization with dequantize-on-the-fly: at inference
  time we reconstruct a full-precision weight tensor from (qweight, scale)
  and then do a normal matmul. That shrinks the weights' RESIDENT MEMORY
  FOOTPRINT (what's stored, what's loaded from disk/HBM as int8/int4) but
  does NOT by itself speed up the matmul — we still materialize and compute
  in full precision. Real inference speedups (e.g. bitsandbytes, GPTQ
  kernels) come from a FUSED low-precision kernel that never materializes
  the full-precision tensor, so the matmul itself reads less memory
  bandwidth. That's a natural M5+ follow-up (a Triton int8 GEMM, structurally
  similar to the M4 paged-attention kernel), not implemented here — measure,
  don't assume, which is exactly what benchmarks/quantization_report.py does.
"""
from __future__ import annotations

import torch


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    weight: (out_features, in_features), any float dtype.
    Returns (qweight: int8 (out_features, in_features), scale: float32 (out_features,)).
    """
    w = weight.float()
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)   # (out_features, 1)
    scale = absmax / 127.0
    qweight = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return qweight, scale.squeeze(1)


def dequantize_int8(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int8. Returns float32 (out_features, in_features)."""
    return qweight.to(torch.float32) * scale.unsqueeze(1)


def quantize_int4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """
    weight: (out_features, in_features), any float dtype.
    Returns (packed: uint8 (out_features, ceil(in_features/2)), scale: float32
    (out_features,), orig_shape) — orig_shape is needed to trim padding on
    dequantize when in_features is odd.

    Packing: two signed nibbles per byte. Values are quantized to [-7, 7],
    shifted to the unsigned range [1, 15] (bias of 8) so they fit in 4 bits,
    then interleaved: byte = (odd_value << 4) | even_value.
    """
    w = weight.float()
    out_features, in_features = w.shape
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0
    q = torch.round(w / scale).clamp(-7, 7).to(torch.int8)          # [-7, 7]
    q_unsigned = (q + 8).to(torch.uint8)                            # [1, 15] — fits a nibble

    if in_features % 2 != 0:
        pad = torch.full((out_features, 1), 8, dtype=torch.uint8)   # bias(8) == dequantizes to 0
        q_unsigned = torch.cat([q_unsigned, pad], dim=1)

    lo = q_unsigned[:, 0::2]
    hi = q_unsigned[:, 1::2]
    packed = (hi << 4) | lo                                         # (out_features, ceil(in/2))
    return packed, scale.squeeze(1), (out_features, in_features)


def dequantize_int4(
    packed: torch.Tensor, scale: torch.Tensor, orig_shape: tuple[int, int]
) -> torch.Tensor:
    """Inverse of quantize_int4. Returns float32 (out_features, in_features)."""
    out_features, in_features = orig_shape
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8

    interleaved = torch.empty(
        packed.shape[0], packed.shape[1] * 2, dtype=torch.int16, device=packed.device
    )
    interleaved[:, 0::2] = lo
    interleaved[:, 1::2] = hi
    interleaved = interleaved[:, :in_features]                      # drop the odd-width pad column

    return interleaved.to(torch.float32) * scale.unsqueeze(1)
