"""Calibration sweep: pick temperature / thresholds / mode on labelled data.

No training. The expensive part is the LLM, so raw logprobs are cached per prompt and
every grid point is replayed offline over spans captured once from the real pipeline.
Tuning and reporting use disjoint splits so the chosen setting is not scored on the
data that selected it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import LocalLLMConfig
from .stage import LocalLLMStage
from .types import Span

# (text, gold spans, merged spans from the pipeline, regex spans from the pipeline)
Record = Tuple[str, Sequence[Span], Sequence[Span], Sequence[Span]]


class CachingProvider:
    """Memoise raw first-token logprobs by prompt; misses go out in one batch."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.cache: Dict[str, Dict[str, float]] = {}
        self.misses = 0

    def score(self, prompts: Sequence[str]) -> List[Dict[str, float]]:
        missing = list(dict.fromkeys(p for p in prompts if p not in self.cache))
        if missing:
            self.misses += len(missing)
            for p, r in zip(missing, self.inner.score(missing)):
                self.cache[p] = r
        return [self.cache[p] for p in prompts]


def micro_f1(pairs: Iterable[Tuple[Sequence[Span], Sequence[Span]]]) -> Dict[str, float]:
    """Strict (label, start, end) micro P/R/F1 — same rule as ``privyscope eval``."""
    tp = fp = fn = 0
    for gold, pred in pairs:
        g = {(s.label, s.start, s.end) for s in gold}
        p = {(s.label, s.start, s.end) for s in pred}
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


@dataclass
class Result:
    config: LocalLLMConfig
    metrics: Dict[str, float]
    failures: int = 0


def grid(
    base: LocalLLMConfig,
    modes: Sequence[str] = ("verify",),
    temperatures: Sequence[float] = (0.5, 1.0, 2.0),
    none_thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
    skip_verified: Sequence[bool] = (True, False),
    accept_thresholds: Sequence[float] = (0.5, 0.7, 0.9),
) -> List[LocalLLMConfig]:
    """Config variants, cheapest mode first. ``accept_threshold`` only varies for extend."""
    out: List[LocalLLMConfig] = []
    for mode in modes:
        accepts = accept_thresholds if mode == "extend" else (base.accept_threshold,)
        for t, n, sv, a in itertools.product(temperatures, none_thresholds, skip_verified, accepts):
            out.append(replace(base, mode=mode, temperature=t, none_threshold=n,
                               skip_verified=sv, accept_threshold=a))
    return out


def evaluate(cfg: LocalLLMConfig, provider, records: Sequence[Record]) -> Result:
    """Replay the stage over captured spans. A failing document keeps its incoming spans,
    exactly as the core does at runtime, and is counted."""
    stage = LocalLLMStage(cfg, provider)
    failures = 0
    pairs = []
    for text, gold, merged, regex_spans in records:
        try:
            pred = stage.refine(text, merged, regex_spans)
        except Exception:  # noqa: BLE001 - mirror core fallback
            failures += 1
            pred = merged
        pairs.append((gold, pred))
    return Result(cfg, micro_f1(pairs), failures)


def split(records: Sequence[Record], tune_fraction: float = 0.5) -> Tuple[List[Record], List[Record]]:
    """Interleaved split (not head/tail), so ordered datasets don't bias either side."""
    if not 0 < tune_fraction < 1:
        raise ValueError("tune_fraction must be in (0, 1)")
    tune, held = [], []
    acc = 0.0
    for r in records:
        acc += tune_fraction
        if acc >= 1.0:
            acc -= 1.0
            tune.append(r)
        else:
            held.append(r)
    return tune, held


@dataclass
class Report:
    baseline_tune: Dict[str, float]
    baseline_held: Dict[str, float]
    best: Optional[Result]            # chosen on the tune split
    best_held: Optional[Result]       # same config, scored on the held-out split
    results: List[Result]             # every grid point, tune split
    server_calls: int
    @property
    def helps(self) -> bool:
        return bool(self.best_held and self.best_held.metrics["f1"] > self.baseline_held["f1"])


def calibrate(
    records: Sequence[Record],
    provider,
    configs: Sequence[LocalLLMConfig],
    tune_fraction: float = 0.5,
) -> Report:
    cache = provider if isinstance(provider, CachingProvider) else CachingProvider(provider)
    tune, held = split(records, tune_fraction)
    base_tune = micro_f1((g, m) for _t, g, m, _r in tune)
    base_held = micro_f1((g, m) for _t, g, m, _r in held)
    results = [evaluate(c, cache, tune) for c in configs]
    best = max(results, key=lambda r: r.metrics["f1"], default=None)  # first max = cheapest
    best_held = evaluate(best.config, cache, held) if best else None
    return Report(base_tune, base_held, best, best_held, results, cache.misses)


def config_to_yaml(cfg: LocalLLMConfig) -> str:
    import yaml

    d = {k: v for k, v in vars(cfg).items()}
    return yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
