"""OpenAI-compatible request/response types — Milestone 6."""
from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, list[str]]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str]  # "stop" | "length" | None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    model: str
    choices: list[CompletionChoice]


class CompletionChunk(BaseModel):
    """SSE streaming chunk."""
    id: str
    object: Literal["text_completion"] = "text_completion"
    choices: list[CompletionChoice]
