import math

import pytest

from privyscope_local_llm.config import LocalLLMConfig
from privyscope_local_llm.scorer import Scorer, ScoringError
from privyscope_local_llm.stage import LocalLLMStage, _DefaultStage
from privyscope_local_llm.types import Span

LABELS = {"PER": "person", "LOC": "address"}  # letters: A=PER B=LOC C=NONE


class FakeProvider:
    """Answers by candidate text: {candidate: letter}; default NONE."""

    def __init__(self, answers):
        self.answers = answers
        self.prompts = []

    def score(self, prompts):
        self.prompts += list(prompts)
        out = []
        for p in prompts:
            cand = p.split("Candidate: ")[1].split("\n")[0]
            letter = self.answers.get(cand, "C")
            out.append({f" {letter}": -0.05, " Z": -9.0})
        return out


def make(answers, **cfg):
    cfg.setdefault("labels", LABELS)
    prov = FakeProvider(answers)
    return LocalLLMStage(LocalLLMConfig(base_url="x", **cfg), prov), prov


TEXT = "Contact Alice Kim at Main Street 5. Ask Bob."


def test_probabilities_normalise_and_missing_letters_are_zero():
    sc = Scorer(None, LABELS)
    p = sc.probabilities({" A": math.log(0.6), "A": math.log(0.2), " C": math.log(0.2)})
    assert p["PER"] == pytest.approx(0.8) and p["NONE"] == pytest.approx(0.2)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["LOC"] == 0.0 and p["PER"] > p["NONE"]


def test_no_option_letter_raises():
    with pytest.raises(ScoringError):
        Scorer(None, LABELS).probabilities({" the": -0.1})


def test_verify_drops_false_positive_and_relabels():
    stage, _ = make({"Alice Kim": "A", "Main": "C", "Street 5": "A"})
    spans = [Span("LOC", 8, 17), Span("PER", 21, 25), Span("LOC", 26, 34)]
    out = stage.refine(TEXT, spans, [])
    assert out == [Span("PER", 8, 17), Span("PER", 26, 34)]


def test_verify_never_adds_spans():
    stage, prov = make({"Bob": "A"})
    assert stage.refine(TEXT, [], []) == []
    assert prov.prompts == []


def test_skip_verified_bypasses_llm():
    stage, prov = make({}, skip_verified=True)
    s = Span("PER", 8, 17)
    assert stage.refine(TEXT, [s], [s]) == [s]
    assert prov.prompts == []
    stage2, prov2 = make({}, skip_verified=False)
    assert stage2.refine(TEXT, [s], [s]) == []  # LLM says NONE
    assert len(prov2.prompts) == 1


def test_extend_adds_span_and_respects_cap():
    stage, prov = make({"Bob": "A"}, mode="extend", max_ngram=1)
    out = stage.refine(TEXT, [], [])
    assert out == [Span("PER", 40, 43)]
    capped, prov2 = make({}, mode="extend", max_ngram=2, max_candidates=3)
    capped.refine(TEXT, [], [])
    assert len(prov2.prompts) == 3


def test_extend_windows_avoid_existing_spans_and_overlaps_resolve():
    stage, _ = make({"Alice": "A", "Alice Kim": "A", "Kim": "A"}, mode="extend", max_ngram=2)
    existing = Span("LOC", 26, 34)
    out = stage.refine(TEXT, [existing], [existing])
    assert existing in out
    people = [s for s in out if s.label == "PER"]
    assert people == [Span("PER", 8, 17)]  # ties resolve to the longest window


def test_prompt_prefix_is_shared_and_first():
    stage, prov = make({}, skip_verified=False)
    stage.refine(TEXT, [Span("PER", 8, 17), Span("LOC", 26, 34)], [])
    pre = stage.scorer.prefix
    assert all(p.startswith(pre) for p in prov.prompts)


def test_unconfigured_default_stage_is_noop(monkeypatch):
    for v in ("PRIVYSCOPE_LOCAL_LLM_URL", "PRIVYSCOPE_LOCAL_LLM_CONFIG"):
        monkeypatch.delenv(v, raising=False)
    s = [Span("PER", 0, 3)]
    assert _DefaultStage().refine("abc", s, []) == s


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        LocalLLMConfig(mode="nope")
