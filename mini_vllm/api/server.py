"""
FastAPI server with SSE streaming — Milestone 6.

Endpoint: POST /v1/completions
  - Non-streaming: returns CompletionResponse JSON
  - Streaming (stream=true): returns text/event-stream of CompletionChunk

The server is a thin adapter: it translates HTTP requests into Sequence
objects, hands them to the engine, and streams tokens back. No inference
logic lives here.
"""
# Implemented in Milestone 6.
