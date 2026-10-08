"""OpenAI-compatible request/response types."""
from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, list[str]]
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(-1, ge=-1)
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
    model: str
    choices: list[CompletionChoice]
