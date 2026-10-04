"""Provider contract: first-token log-probabilities for a batch of prompts."""
from __future__ import annotations

from typing import Dict, List, Protocol, Sequence


class Provider(Protocol):
    def score(self, prompts: Sequence[str]) -> List[Dict[str, float]]:
        """Return, per prompt, ``{token: logprob}`` for the first generated token."""
        ...
