"""
Quantization correctness tests — Milestone 5.

Unlike M1-M4 (exact token-for-token match required), quantization is
inherently lossy — the correctness bar here is BOUNDED ERROR, not exact
equality. Thresholds below were calibrated by measuring actual error on
random weights of realistic transformer magnitude (~N(0, 0.02)), not
guessed: INT8 typically lands at ~1e-4 mean abs error / >0.9999 cosine
similarity; INT4 (7x coarser quantization step) at ~2e-3 / >0.99. See the
quantize.py module docstring for why this is weight-only (memory savings)
rather than a compute speedup.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from mini_vllm.quantization.quantize import (
    dequantize_int4,
    dequantize_int8,
    quantize_int4,
    quantize_int8,
)
from mini_vllm.quantization.quantized_linear import (
    QuantizedLinear,
    linear_layer_bytes,
    quantize_model,
)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(16, 33) * 0.02   # odd in_features exercises INT4 padding path


def test_int8_roundtrip_error_bounds(weight):
    qweight, scale = quantize_int8(weight)
    assert qweight.dtype == torch.int8
    assert qweight.shape == weight.shape

    deq = dequantize_int8(qweight, scale)
    assert deq.shape == weight.shape
    # RTN guarantees per-element error <= scale/2; scale = absmax/127 for this
    # weight magnitude is tiny, so this bound is generous but not vacuous.
    max_scale = scale.max().item()
    assert (weight - deq).abs().max().item() <= max_scale / 2 + 1e-8
    assert _cosine_sim(weight, deq) > 0.999


def test_int4_roundtrip_error_bounds(weight):
    packed, scale, orig_shape = quantize_int4(weight)
    assert packed.dtype == torch.uint8
    assert orig_shape == tuple(weight.shape)
    # Packed width is ceil(in_features/2) — odd in_features (33) pads to 34/2=17.
    assert packed.shape == (weight.shape[0], (weight.shape[1] + 1) // 2)

    deq = dequantize_int4(packed, scale, orig_shape)
    assert deq.shape == weight.shape
    max_scale = scale.max().item()
    assert (weight - deq).abs().max().item() <= max_scale / 2 + 1e-8
    # Looser bound than INT8 — 4-bit has a 7x coarser quantization step.
    assert _cosine_sim(weight, deq) > 0.98


def test_quantized_linear_matches_dense_closely():
    torch.manual_seed(0)
    linear = nn.Linear(20, 10, bias=True)
    x = torch.randn(4, 20)
    expected = linear(x)

    q8 = QuantizedLinear.from_linear(linear, bits=8)
    out8 = q8(x)
    assert out8.shape == expected.shape
    assert _cosine_sim(out8, expected) > 0.999

    q4 = QuantizedLinear.from_linear(linear, bits=4)
    out4 = q4(x)
    assert out4.shape == expected.shape
    assert _cosine_sim(out4, expected) > 0.95


def test_quantize_model_replaces_target_linears_only():
    """Only q/k/v/o_proj and MLP projections get swapped — embed_tokens and
    lm_head (deliberately excluded, see quantized_linear.py docstring) stay
    as plain nn.Linear/nn.Embedding."""
    config = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)

    fp32_bytes = linear_layer_bytes(model)
    quantize_model(model, bits=8)
    int8_bytes = linear_layer_bytes(model)

    assert isinstance(model.lm_head, nn.Linear)          # untouched
    assert not isinstance(model.lm_head, QuantizedLinear)
    for layer in model.model.layers:
        assert isinstance(layer.self_attn.q_proj, QuantizedLinear)
        assert isinstance(layer.mlp.gate_proj, QuantizedLinear)

    # int8 storage should be roughly 4x smaller than fp32 (weight bytes only;
    # scale tensor overhead is negligible — one float per output channel).
    assert int8_bytes < fp32_bytes / 3


def test_quantize_model_int4_smaller_than_int8():
    config = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model_int8 = LlamaForCausalLM(config)
    model_int4 = LlamaForCausalLM(config)
    model_int4.load_state_dict(model_int8.state_dict())   # identical weights

    quantize_model(model_int8, bits=8)
    quantize_model(model_int4, bits=4)

    bytes_int8 = linear_layer_bytes(model_int8)
    bytes_int4 = linear_layer_bytes(model_int4)
    assert bytes_int4 < bytes_int8 * 0.6   # packing halves it, minus scale overhead


def test_quantized_model_logits_close_to_fp32_baseline():
    """
    Single-step logits comparison (not multi-token generation) — isolates
    quantization error from autoregressive divergence, which would otherwise
    swamp the signal: once one sampled token differs, every subsequent token
    differs too, regardless of how good the quantization is.
    """
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    baseline = LlamaForCausalLM(config)
    baseline.eval()

    model_int8 = LlamaForCausalLM(config)
    model_int8.load_state_dict(baseline.state_dict())
    quantize_model(model_int8, bits=8)
    model_int8.eval()

    model_int4 = LlamaForCausalLM(config)
    model_int4.load_state_dict(baseline.state_dict())
    quantize_model(model_int4, bits=4)
    model_int4.eval()

    input_ids = torch.tensor([[5, 12, 7, 20, 3]])
    with torch.no_grad():
        logits_fp32 = baseline(input_ids).logits[0, -1]
        logits_int8 = model_int8(input_ids).logits[0, -1]
        logits_int4 = model_int4(input_ids).logits[0, -1]

    sim8 = _cosine_sim(logits_fp32, logits_int8)
    sim4 = _cosine_sim(logits_fp32, logits_int4)
    assert sim8 > 0.99, f"INT8 logits diverged too far from fp32 baseline: cos_sim={sim8}"
    assert sim4 > 0.90, f"INT4 logits diverged too far from fp32 baseline: cos_sim={sim4}"
    # INT8 should stay closer to the baseline than INT4 — sanity check that
    # more bits really does mean less error, not just "both are fine".
    assert sim8 > sim4
