"""The ``privyscope.stages`` refinement stage."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .candidates import candidate_windows, context_for
from .config import LocalLLMConfig
from .scorer import NONE, Scorer
from .types import Span

log = logging.getLogger(__name__)


class LocalLLMStage:
    """Verify (and optionally extend) detections with a local LLM.

    ``verify``: scores spans other stages proposed; drops false positives, re-labels.
    ``extend``: additionally scores token windows so the LLM can add spans.
    Any provider/scoring error propagates; the core skips a failing stage and keeps
    the incoming spans.
    """

    def __init__(self, config: LocalLLMConfig, provider=None) -> None:
        self.config = config
        self.scorer = Scorer(provider or config.make_provider(), config.labels, config.temperature)

    @staticmethod
    def _best(probs) -> str:
        return max((k for k in probs if k != NONE), key=probs.get)

    def refine(self, text: str, spans: Sequence[Span], regex_spans: Sequence[Span] = ()) -> List[Span]:
        cfg = self.config
        spans = list(spans)
        trusted = set(regex_spans) if cfg.skip_verified else set()
        to_check = [s for s in spans if s not in trusted]
        windows = (
            candidate_windows(text, spans, cfg.max_ngram, cfg.max_candidates)
            if cfg.mode == "extend"
            else []
        )
        items = [(context_for(text, s, cfg.context_chars), text[s.start : s.end]) for s in to_check]
        items += [
            (context_for(text, Span("", a, b), cfg.context_chars), text[a:b]) for a, b in windows
        ]
        probs = self.scorer.score(items)

        verdict = {}
        for s, p in zip(to_check, probs):
            if p[NONE] >= cfg.none_threshold:
                verdict[s] = None
            else:
                verdict[s] = Span(self._best(p), s.start, s.end)
        out = [verdict.get(s, s) for s in spans]
        out = [s for s in out if s is not None]

        added = []
        for (a, b), p in zip(windows, probs[len(to_check):]):
            label = self._best(p)
            if p[NONE] < cfg.none_threshold and p[label] >= cfg.accept_threshold:
                added.append((p[label], Span(label, a, b)))
        taken: List[Span] = []
        for _score, cand in sorted(added, key=lambda t: (-t[0], -(t[1].end - t[1].start))):  # best, then longest; drop overlaps
            if not any(not (cand.end <= t.start or cand.start >= t.end) for t in taken):
                taken.append(cand)
        return sorted(out + taken, key=lambda s: s.start)


class _DefaultStage:
    """Entry-point object. A no-op until a server is configured via env/YAML."""

    def __init__(self) -> None:
        self._inner: Optional[LocalLLMStage] = None
        self._resolved = False

    def refine(self, text, spans, regex_spans=()):
        if not self._resolved:
            cfg = LocalLLMConfig.from_env()
            self._inner = LocalLLMStage(cfg) if cfg.configured else None
            self._resolved = True
        return list(spans) if self._inner is None else self._inner.refine(text, spans, regex_spans)


default_stage = _DefaultStage()
