"""
Drop-in replacement for nn.Linear that stores weights quantized and
dequantizes on the fly at forward time — Milestone 5.

quantize_model() walks a loaded HF model and swaps matching nn.Linear
submodules in place, so ModelRunner/PagedLlamaRunner can drive a quantized
model exactly like a full-precision one — quantization is orthogonal to
M3 (scheduling) and M4 (paging): neither knows or cares that the Linear
layers underneath are int8/int4.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.quantization.quantize import (
    dequantize_int4,
    dequantize_int8,
    quantize_int4,
    quantize_int8,
)

# Attention + MLP projections — the bulk of a transformer's parameters.
# Embedding and lm_head are deliberately excluded: they're a small fraction
# of total size for a model with many layers, and quantizing the output
# projection (lm_head) has an outsized effect on generation quality since
# every token's logits pass through it directly.
DEFAULT_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


class QuantizedLinear(nn.Module):
    """
    Holds a quantized weight (+ per-channel scale) and an unquantized bias.
    forward() dequantizes the full weight matrix and calls F.linear — see
    quantize.py's module docstring for why this doesn't speed up compute.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        bits: int,
        orig_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.bits = bits
        self.orig_shape = orig_shape

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int) -> "QuantizedLinear":
        w = linear.weight.data
        if bits == 8:
            qweight, scale = quantize_int8(w)
            orig_shape = tuple(w.shape)
        elif bits == 4:
            qweight, scale, orig_shape = quantize_int4(w)
        else:
            raise ValueError(f"unsupported bits={bits!r}; expected 8 or 4")

        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(qweight, scale, bias, bits, orig_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 8:
            weight = dequantize_int8(self.qweight, self.scale)
        else:
            weight = dequantize_int4(self.qweight, self.scale, self.orig_shape)
        weight = weight.to(dtype=x.dtype, device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def packed_weight_bytes(self) -> int:
        """Bytes actually used to STORE the weight (what quantization saves)."""
        return self.qweight.numel() * self.qweight.element_size() + self.scale.numel() * self.scale.element_size()


def quantize_model(
    model: nn.Module,
    bits: int,
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
) -> nn.Module:
    """
    In-place: replaces every nn.Linear submodule whose attribute name ends
    a target suffix (e.g. "q_proj") with a QuantizedLinear. Returns the same
    model object for convenience chaining.
    """
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_suffixes:
                setattr(module, child_name, QuantizedLinear.from_linear(child, bits))
    return model


def linear_layer_bytes(model: nn.Module, target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES) -> int:
    """
    Total bytes used by the target Linear/QuantizedLinear layers — the
    portion of the model quantize_model() actually touches. Use this before
    AND after quantize_model() on comparable models to measure size
    reduction (see benchmarks/quantization_report.py).
    """
    total = 0
    for module in model.modules():
        for child_name, child in module.named_children():
            if child_name not in target_suffixes:
                continue
            if isinstance(child, QuantizedLinear):
                total += child.packed_weight_bytes()
            elif isinstance(child, nn.Linear):
                total += child.weight.numel() * child.weight.element_size()
    return total
