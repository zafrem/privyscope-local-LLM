from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Tuple

DEFAULT_LABELS: Dict[str, str] = {
    "PER": "person name",
    "PHONE": "phone number",
    "ID_NUM": "national or government ID number",
    "EMAIL": "email address",
    "LOC": "detailed street address",
    "BANK": "bank account or card number",
    "DATE": "private date (e.g. birth date)",
    "SECRET": "password, API key or credential",
}


@dataclass
class LocalLLMConfig:
    base_url: Optional[str] = None      # unset → stage is a no-op
    model: str = ""
    api_key: Optional[str] = None
    provider: str = "openai"            # "openai" (vLLM, LM Studio, llama.cpp) | "ollama"
    mode: str = "verify"                # "verify" | "extend"
    none_threshold: float = 0.5         # drop/skip a candidate when P(NONE) >= this
    accept_threshold: float = 0.5       # extend: add a window when P(best label) >= this
    skip_verified: bool = True          # regex spans bypass the LLM
    temperature: float = 1.0            # logprob calibration
    context_chars: int = 200            # context kept each side of a candidate
    max_ngram: int = 3                  # extend: window size in whitespace tokens
    max_candidates: int = 64            # extend: cap on windows scored per text
    timeout: float = 30.0
    concurrency: int = 4
    labels: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABELS))

    def __post_init__(self) -> None:
        if self.mode not in ("verify", "extend"):
            raise ValueError(f"mode must be 'verify' or 'extend', got {self.mode!r}")

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    @classmethod
    def from_yaml(cls, path: str) -> "LocalLLMConfig":
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_env(cls) -> "LocalLLMConfig":
        """``PRIVYSCOPE_LOCAL_LLM_CONFIG`` (YAML path) wins; else URL/MODEL/MODE vars."""
        path = os.environ.get("PRIVYSCOPE_LOCAL_LLM_CONFIG")
        if path:
            return cls.from_yaml(path)
        return cls(
            base_url=os.environ.get("PRIVYSCOPE_LOCAL_LLM_URL"),
            model=os.environ.get("PRIVYSCOPE_LOCAL_LLM_MODEL", ""),
            mode=os.environ.get("PRIVYSCOPE_LOCAL_LLM_MODE", "verify"),
        )

    def make_provider(self):
        from .providers import OllamaProvider, OpenAICompatProvider

        kw = dict(api_key=self.api_key, timeout=self.timeout, concurrency=self.concurrency)
        if self.provider == "ollama":
            return OllamaProvider(self.model, **({"base_url": self.base_url, **kw}))
        if self.provider == "openai":
            return OpenAICompatProvider(self.base_url, self.model, **kw)
        raise ValueError(f"unknown provider {self.provider!r}")
