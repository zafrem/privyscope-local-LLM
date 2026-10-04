"""OpenAI-compatible ``/completions`` adapter (vLLM, LM Studio, llama.cpp server, Ollama /v1).

One request per prompt with ``max_tokens=1`` and ``logprobs=top_logprobs``: no
generation, so there is nothing to parse. Stdlib only.
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence


class ProviderError(RuntimeError):
    pass


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        top_logprobs: int = 20,
        timeout: float = 30.0,
        concurrency: int = 4,
    ) -> None:
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.api_key = api_key
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.concurrency = max(1, concurrency)

    def _one(self, prompt: str) -> Dict[str, float]:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": self.top_logprobs,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.load(resp)
            return dict(data["choices"][0]["logprobs"]["top_logprobs"][0])
        except Exception as exc:  # network, HTTP, or a server that returns no logprobs
            raise ProviderError(f"{self.url}: {exc}") from exc

    def score(self, prompts: Sequence[str]) -> List[Dict[str, float]]:
        if len(prompts) <= 1 or self.concurrency == 1:
            return [self._one(p) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, prompts))
