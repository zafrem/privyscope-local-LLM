"""Single-pass candidate scoring.

Each candidate is one prompt; the answer is one letter whose first-token logprob we
read. Letters (not entity codes) keep every option a single token, and the fixed
instruction + option menu comes first so servers can reuse it as a cached prefix.
"""
from __future__ import annotations

import math
import string
from typing import Dict, List, Mapping, Sequence, Tuple

NONE = "NONE"


class ScoringError(RuntimeError):
    pass


class Scorer:
    def __init__(self, provider, labels: Mapping[str, str], temperature: float = 1.0) -> None:
        if len(labels) + 1 > len(string.ascii_uppercase):
            raise ValueError("too many labels for single-letter options")
        self.provider = provider
        self.temperature = temperature
        names = list(labels) + [NONE]
        self.letter_of: Dict[str, str] = dict(zip(names, string.ascii_uppercase))
        menu = "\n".join(
            f"{self.letter_of[n]}) {n} - {labels[n]}" for n in labels
        ) + f"\n{self.letter_of[NONE]}) {NONE} - not personal information"
        self.prefix = (
            "You check candidates for a personal-information detector.\n"
            "Given a text and a candidate span from it, choose the option that best "
            "describes the span as it is used in the text.\n\n"
            f"Options:\n{menu}\n\n"
        )

    def prompt(self, context: str, candidate: str) -> str:
        return f"{self.prefix}Text: {context}\nCandidate: {candidate}\nAnswer:"

    def probabilities(self, logprobs: Mapping[str, float]) -> Dict[str, float]:
        """Softmax over the option letters found in the top logprobs."""
        raw: Dict[str, List[float]] = {}
        for tok, lp in logprobs.items():
            key = tok.strip().upper()
            if key in self.letter_of.values():
                raw.setdefault(key, []).append(lp / self.temperature)
        if not raw:
            raise ScoringError("no option letter in returned logprobs")
        score = {k: math.log(sum(math.exp(v) for v in vs)) for k, vs in raw.items()}
        m = max(score.values())
        z = sum(math.exp(s - m) for s in score.values())
        by_letter = {k: math.exp(s - m) / z for k, s in score.items()}
        return {name: by_letter.get(letter, 0.0) for name, letter in self.letter_of.items()}

    def score(self, items: Sequence[Tuple[str, str]]) -> List[Dict[str, float]]:
        """``items`` are ``(context, candidate)``; returns ``{label_or_NONE: prob}`` each."""
        if not items:
            return []
        results = self.provider.score([self.prompt(c, s) for c, s in items])
        if len(results) != len(items):
            raise ScoringError("provider returned a different number of results")
        return [self.probabilities(r) for r in results]
