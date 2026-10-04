"""Ollama via its OpenAI-compatible endpoint.

Logprob support depends on the Ollama version; if the server returns none, scoring
raises and the stage falls back to the incoming spans. Not verified against a live
Ollama server yet.
"""
from __future__ import annotations

from .openai_compat import OpenAICompatProvider


class OllamaProvider(OpenAICompatProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", **kw) -> None:
        super().__init__(base_url, model, **kw)
