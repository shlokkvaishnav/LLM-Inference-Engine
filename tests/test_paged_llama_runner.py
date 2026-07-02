"""
PagedLlamaRunner correctness test — Milestone 4 engine integration.

Strategy: instantiate a tiny, randomly-initialized LlamaForCausalLM (real
Llama architecture class, no download — fast on CPU) and run the SAME
prompts through both:
  - ModelRunner.generate()            (M1-M3, dense, already proven correct
                                        against real HF token-for-token)
  - PagedLlamaRunner (prefill_and_store + decode_batch, M4, paged)

If both produce identical greedy tokens on identical weights, the paged
implementation is a provably faithful reimplementation of the same
architecture's forward pass — this is the strongest test available without
downloading TinyLlama's real weights.

num_key_value_heads < num_attention_heads exercises the GQA head-expansion
path in PagedLlamaRunner (most real Llama models, including TinyLlama, use
GQA — a bug there wouldn't be caught by an MHA-only config).

test_paged_llama_matches_hf_on_real_model (bottom of this file) runs the
same comparison against the real TinyLlama-1.1B weights on GPU — gated on
MINI_VLLM_TEST_DEVICE=cuda since it needs real hardware and downloads the
model; runs on Kaggle, skipped everywhere else.
"""
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM

from mini_vllm.engine.sequence import Sequence, SamplingParams
from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.kv_cache.block_manager import BlockManager
from mini_vllm.model.loader import ModelConfig, load_model
from mini_vllm.model.paged_llama_runner import PagedLlamaRunner
from mini_vllm.model.runner import ModelRunner


class _TinyTokenizer:
    """Minimal tokenizer stub — sequences are built directly from token IDs
    in this test, so only pad_token_id/eos_token_id need to exist."""

    def __init__(self, pad_token_id: int, eos_token_id: int):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,     # GQA: fewer KV heads than query heads
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=99,           # unreachable by random logits in 5 steps — no early stop
    )
    model = LlamaForCausalLM(config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


MAX_NEW_TOKENS = 4
PROMPTS = [
    [5, 12, 7, 20],
    [3, 3, 3],           # different length — exercises left-padding in prefill
    [40, 41, 42, 43, 44, 45],
]


def test_paged_llama_matches_dense_generate(tiny_model):
    tokenizer = _TinyTokenizer(pad_token_id=0, eos_token_id=99)

    class _Config:
        model_name_or_path = "tiny-llama-test"
        dtype = "float32"
        device = "cpu"
        max_model_len = 64
        trust_remote_code = False

    # --- Dense baseline (M1-M3, already proven correct) ---
    dense_runner = ModelRunner(tiny_model, tokenizer, _Config())
    dense_seqs = [
        Sequence(list(p), SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
        for p in PROMPTS
    ]
    dense_runner.generate(dense_seqs)
    dense_outputs = [s.output_token_ids[:] for s in dense_seqs]

    # --- Paged path (M4) ---
    block_manager = BlockManager(num_blocks=64, block_size=4)
    paged_runner = PagedLlamaRunner(
        tiny_model, tokenizer, block_manager, dtype=torch.float32, device="cpu"
    )
    engine = LLMEngine(paged_runner, max_batch_size=len(PROMPTS), block_manager=block_manager)
    paged_seqs = [
        Sequence(list(p), SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
        for p in PROMPTS
    ]
    for seq in paged_seqs:
        engine.add_request(seq)
    engine.run_until_done()
    paged_outputs = [s.output_token_ids[:] for s in paged_seqs]

    for i, prompt in enumerate(PROMPTS):
        assert paged_outputs[i] == dense_outputs[i], (
            f"Paged/dense mismatch on prompt {prompt!r}\n"
            f"  dense: {dense_outputs[i]}\n"
            f"  paged: {paged_outputs[i]}"
        )


def test_paged_llama_releases_blocks_on_finish(tiny_model):
    """After every sequence finishes, all blocks return to the free pool —
    no leaks from the paged decode path."""
    tokenizer = _TinyTokenizer(pad_token_id=0, eos_token_id=99)
    block_manager = BlockManager(num_blocks=64, block_size=4)
    paged_runner = PagedLlamaRunner(
        tiny_model, tokenizer, block_manager, dtype=torch.float32, device="cpu"
    )
    engine = LLMEngine(paged_runner, max_batch_size=2, block_manager=block_manager)

    for p in PROMPTS:
        engine.add_request(
            Sequence(list(p), SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
        )
    engine.run_until_done()

    assert block_manager.num_free_blocks == block_manager.num_blocks


@pytest.mark.skipif(
    os.environ.get("MINI_VLLM_TEST_DEVICE") != "cuda",
    reason="Real-model paged run needs GPU + a TinyLlama download — verified on Kaggle",
)
def test_paged_llama_matches_hf_on_real_model():
    """
    End-to-end proof on the actual production model: PagedLlamaRunner +
    LLMEngine (paged, Triton kernel on GPU) must match transformers.generate()
    token-for-token on real TinyLlama-1.1B weights — not just the tiny
    random-weight architecture check above.
    """
    real_prompts = [
        "The capital of France is",
        "Once upon a time in a land far away,",
        "def fibonacci(n):",
    ]
    max_new_tokens = 5

    config = ModelConfig(
        model_name_or_path=os.environ.get(
            "MINI_VLLM_TEST_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ),
        dtype="float16",
        device="cuda",
        max_model_len=512,
    )
    model, tokenizer = load_model(config)

    # HF ground truth.
    expected = []
    for prompt in real_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new_tokens, max_length=None,
                do_sample=False, temperature=1.0, use_cache=True,
            )
        expected.append(out[0, input_ids.shape[1]:].tolist())

    # Paged path.
    block_manager = BlockManager(num_blocks=256, block_size=16)
    paged_runner = PagedLlamaRunner(
        model, tokenizer, block_manager, dtype=torch.float16, device="cuda"
    )
    engine = LLMEngine(paged_runner, max_batch_size=len(real_prompts), block_manager=block_manager)
    seqs = [
        Sequence(tokenizer.encode(p), SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
        for p in real_prompts
    ]
    for seq in seqs:
        engine.add_request(seq)
    engine.run_until_done()

    for i, prompt in enumerate(real_prompts):
        actual = seqs[i].output_token_ids
        assert actual == expected[i], (
            f"Real-model paged/HF mismatch on {prompt!r}\n"
            f"  HF:    {expected[i]} ({tokenizer.decode(expected[i])!r})\n"
            f"  paged: {actual} ({tokenizer.decode(actual)!r})"
        )
