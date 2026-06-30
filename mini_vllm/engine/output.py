"""Output types returned by the engine to callers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompletionOutput:
    """The generated text for one sequence within a request."""
    text: str
    token_ids: list[int]
    finish_reason: Optional[str]  # "stop" | "length" | None (still running)


@dataclass
class RequestOutput:
    """Top-level output object returned per request after each engine step."""
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool
