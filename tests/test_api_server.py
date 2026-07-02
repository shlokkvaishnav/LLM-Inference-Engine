"""
API server tests — Milestone 6.

Runs entirely on CPU with GPT-2 (fast, deterministic greedy decoding) via
env var overrides — mirrors the MINI_VLLM_TEST_MODEL/DEVICE convention used
elsewhere, applied here to server.py's MINI_VLLM_MODEL/DEVICE (no _TEST_
infix — those are the server's actual runtime config knobs).

Since server.py reads its env vars lazily inside the lifespan handler (not
at module import time), setting os.environ before triggering lifespan is
enough — no need to reload the module.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
httpx = pytest.importorskip("httpx")

os.environ.setdefault("MINI_VLLM_MODEL", "gpt2")
os.environ.setdefault("MINI_VLLM_DEVICE", "cpu")

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from mini_vllm.api import server as server_module


@pytest.fixture
def client():
    with TestClient(server_module.app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "gpt2"


def test_completions_non_streaming_matches_hf_baseline(client):
    prompt = "The capital of France is"
    resp = client.post(
        "/v1/completions",
        json={"model": "gpt2", "prompt": prompt, "max_tokens": 5, "temperature": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    server_text = body["choices"][0]["text"]

    # Independent ground truth: fresh GPT-2 load, greedy transformers.generate().
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=5, max_length=None,
            do_sample=False, temperature=1.0, use_cache=True,
        )
    expected_text = tokenizer.decode(out[0, input_ids.shape[1]:])

    assert server_text == expected_text


def test_completions_streaming_matches_non_streaming(client):
    prompt = "def fibonacci(n):"
    payload = {"model": "gpt2", "prompt": prompt, "max_tokens": 5, "temperature": 0.0}

    non_stream_resp = client.post("/v1/completions", json={**payload, "stream": False})
    expected_text = non_stream_resp.json()["choices"][0]["text"]

    collected = ""
    saw_done = False
    with client.stream("POST", "/v1/completions", json={**payload, "stream": True}) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload_str = line[len("data: "):]
            if payload_str == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(payload_str)
            collected += chunk["choices"][0]["text"]

    assert saw_done
    assert collected == expected_text


@pytest.mark.asyncio
async def test_concurrent_requests_interleave_correctly():
    """
    Two requests submitted concurrently must each produce results as if run
    alone — proves AsyncLLMEngine's background step loop interleaves
    multiple streams via continuous batching without corrupting either one.
    """
    prompts = ["The capital of France is", "Once upon a time in a land far away,"]
    payload_base = {"model": "gpt2", "max_tokens": 5, "temperature": 0.0, "stream": False}

    async with server_module.lifespan(server_module.app):
        transport = ASGITransport(app=server_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            import asyncio

            responses = await asyncio.gather(*[
                client.post("/v1/completions", json={**payload_base, "prompt": p})
                for p in prompts
            ])

    texts = [r.json()["choices"][0]["text"] for r in responses]

    # Ground truth: same two prompts run one at a time through a fresh model.
    from mini_vllm.engine.sequence import SamplingParams, Sequence
    from mini_vllm.model.loader import ModelConfig, load_model
    from mini_vllm.model.runner import ModelRunner

    config = ModelConfig(model_name_or_path="gpt2", dtype="float32", device="cpu", max_model_len=512)
    model, tokenizer = load_model(config)
    runner = ModelRunner(model, tokenizer, config)

    for i, prompt in enumerate(prompts):
        seq = Sequence(tokenizer.encode(prompt), SamplingParams(temperature=0.0, max_tokens=5))
        runner.generate([seq])
        expected = tokenizer.decode(seq.output_token_ids)
        assert texts[i] == expected, f"prompt {i}: concurrent result diverged from solo baseline"
