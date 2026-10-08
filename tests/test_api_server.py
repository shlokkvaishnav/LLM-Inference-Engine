"""
API server tests.

Runs on CPU with GPT-2 by default (fast, deterministic greedy decoding).
Override via the SAME env vars server.py itself reads (no _TEST_ infix —
these ARE the server's real runtime config knobs) to exercise the real
production model + PagedLlamaRunner path on Kaggle:

    MINI_VLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
    MINI_VLLM_DEVICE=cuda

Since server.py reads its env vars lazily inside the lifespan handler (not
at module import time), setting os.environ before triggering lifespan is
enough — no need to reload the module. This file reads the SAME env vars
for its own ground-truth comparisons, so switching MODEL/DEVICE covers both
the server under test and what it's being checked against — a test that
silently compared against the wrong baseline would be worse than no test.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
httpx = pytest.importorskip("httpx")

MODEL = os.environ.get("MINI_VLLM_MODEL", "gpt2")
DEVICE = os.environ.get("MINI_VLLM_DEVICE", "cpu")
DTYPE = "float16" if DEVICE == "cuda" else "float32"

os.environ.setdefault("MINI_VLLM_MODEL", MODEL)
os.environ.setdefault("MINI_VLLM_DEVICE", DEVICE)

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
    assert body["model"] == MODEL


def test_completions_non_streaming_matches_hf_baseline(client):
    prompt = "The capital of France is"
    resp = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": 5, "temperature": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    server_text = body["choices"][0]["text"]

    # Independent ground truth: fresh model load, greedy transformers.generate().
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # torch_dtype= (not dtype=) — matches mini_vllm/model/loader.py's own
    # handling of the transformers 4.46.3 (Kaggle-pinned) vs 5.x API split.
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=getattr(torch, DTYPE))
    model = model.to(DEVICE)
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=5, max_length=None,
            do_sample=False, temperature=1.0, use_cache=True,
        )
    expected_text = tokenizer.decode(out[0, input_ids.shape[1]:])

    assert server_text == expected_text


def test_completions_streaming_matches_non_streaming(client):
    prompt = "def fibonacci(n):"
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 5, "temperature": 0.0}

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
            if chunk["choices"]:
                collected += chunk["choices"][0]["text"]
            else:
                usage = chunk["usage"]

    assert saw_done
    assert collected == expected_text
    assert usage == non_stream_resp.json()["usage"]


@pytest.mark.asyncio
async def test_concurrent_requests_interleave_correctly():
    """
    Two requests submitted concurrently must each produce results as if run
    alone — proves AsyncLLMEngine's background step loop interleaves
    multiple streams via continuous batching without corrupting either one.

    Ground truth uses ModelRunner (dense) directly even when the
    server itself is using PagedLlamaRunner (for Llama-on-CUDA) —
    test_paged_llama_runner.py already proves those two produce identical
    output, so ModelRunner remains a valid independent baseline either way.
    """
    prompts = ["The capital of France is", "Once upon a time in a land far away,"]
    payload_base = {"model": MODEL, "max_tokens": 5, "temperature": 0.0, "stream": False}

    async with server_module.lifespan(server_module.app):
        transport = ASGITransport(app=server_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            import asyncio

            responses = await asyncio.gather(*[
                client.post("/v1/completions", json={**payload_base, "prompt": p})
                for p in prompts
            ])

    texts = [r.json()["choices"][0]["text"] for r in responses]

    from mini_vllm.engine.sequence import SamplingParams, Sequence
    from mini_vllm.model.loader import ModelConfig, load_model
    from mini_vllm.model.runner import ModelRunner

    config = ModelConfig(model_name_or_path=MODEL, dtype=DTYPE, device=DEVICE, max_model_len=512)
    model, tokenizer = load_model(config)
    runner = ModelRunner(model, tokenizer, config)

    for i, prompt in enumerate(prompts):
        seq = Sequence(tokenizer.encode(prompt), SamplingParams(temperature=0.0, max_tokens=5))
        runner.generate([seq])
        expected = tokenizer.decode(seq.output_token_ids)
        assert texts[i] == expected, f"prompt {i}: concurrent result diverged from solo baseline"


def test_rejects_invalid_sampling_params(client):
    for bad in ({"max_tokens": 0}, {"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": -5}):
        resp = client.post("/v1/completions", json={"model": MODEL, "prompt": "Hi", **bad})
        assert resp.status_code == 422, bad


def test_rejects_prompt_that_cannot_fit(client):
    resp = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "Hi", "max_tokens": 10_000_000},
    )
    assert resp.status_code == 400


def test_rejects_empty_prompt(client):
    resp = client.post("/v1/completions", json={"model": MODEL, "prompt": "", "max_tokens": 2})
    assert resp.status_code == 400


def test_stop_string_truncates_output(client):
    base = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "The capital of France is", "max_tokens": 12, "temperature": 0.0},
    ).json()["choices"][0]["text"]
    stop = base[5:8]
    resp = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "The capital of France is", "max_tokens": 12,
              "temperature": 0.0, "stop": stop},
    ).json()["choices"][0]
    assert stop not in resp["text"]
    assert resp["finish_reason"] == "stop"
    assert base.startswith(resp["text"])


def test_models_endpoint(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == [MODEL]
    assert body["data"][0]["object"] == "model"


def test_usage_counts_match_tokenizer(client):
    from transformers import AutoTokenizer

    prompt = "The capital of France is"
    resp = client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": 5, "temperature": 0.0},
    ).json()
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_tokens = len(tok.encode(prompt))
    assert resp["usage"]["prompt_tokens"] == prompt_tokens
    assert resp["usage"]["completion_tokens"] == 5
    assert resp["usage"]["total_tokens"] == prompt_tokens + 5


def test_batched_prompts_match_solo_runs(client):
    prompts = ["The capital of France is", "Once upon a time", "def fibonacci(n):"]
    base = {"model": MODEL, "max_tokens": 6, "temperature": 0.0}
    solo = [
        client.post("/v1/completions", json={**base, "prompt": p}).json() for p in prompts
    ]
    batched = client.post("/v1/completions", json={**base, "prompt": prompts}).json()

    assert [c["index"] for c in batched["choices"]] == [0, 1, 2]
    assert [c["text"] for c in batched["choices"]] == [r["choices"][0]["text"] for r in solo]
    assert batched["usage"]["prompt_tokens"] == sum(r["usage"]["prompt_tokens"] for r in solo)
    assert batched["usage"]["completion_tokens"] == sum(r["usage"]["completion_tokens"] for r in solo)


def test_batched_prompts_streaming_indices(client):
    prompts = ["The capital of France is", "Once upon a time"]
    base = {"model": MODEL, "max_tokens": 6, "temperature": 0.0}
    expected = client.post("/v1/completions", json={**base, "prompt": prompts}).json()

    texts = {0: "", 1: ""}
    finished = set()
    usage = None
    with client.stream("POST", "/v1/completions", json={**base, "prompt": prompts, "stream": True}) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            if not chunk["choices"]:
                usage = chunk["usage"]
                continue
            c = chunk["choices"][0]
            texts[c["index"]] += c["text"]
            if c["finish_reason"]:
                finished.add(c["index"])

    assert finished == {0, 1}
    assert [texts[0], texts[1]] == [c["text"] for c in expected["choices"]]
    assert usage == expected["usage"]


def test_batched_prompts_reject_whole_request_if_one_is_invalid(client):
    base = {"model": MODEL, "max_tokens": 4}
    assert client.post("/v1/completions", json={**base, "prompt": ["ok", ""]}).status_code == 400
    assert client.post("/v1/completions", json={**base, "prompt": []}).status_code == 400
    too_long = {**base, "max_tokens": 10_000_000, "prompt": ["ok", "also ok"]}
    assert client.post("/v1/completions", json=too_long).status_code == 400
    # nothing was left running
    engine = server_module.state["engine"]
    assert not engine.engine.scheduler.has_work()
