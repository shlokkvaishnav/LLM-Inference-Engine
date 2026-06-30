"""
Model loading — the only layer that knows which model is on disk.

Everything above this (scheduler, block manager, API server) talks to
ModelRunner, not to transformers. Swapping in a different model later is
a ModelConfig change, not an engine rewrite.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "float32"       # "float32" for CPU; "float16"/"bfloat16" for GPU
    device: str = "cpu"
    max_model_len: int = 2048
    trust_remote_code: bool = False


def load_model(config: ModelConfig) -> tuple[nn.Module, Any]:
    """
    Load weights and tokenizer. Returns (model, tokenizer).

    Model is in eval mode with gradients disabled — inference only.
    This function is the single import boundary for transformers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[config.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
    )
    model = model.to(config.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, tokenizer
