import pytest

from privyscope_local_llm.calibrate import (CachingProvider, calibrate, config_to_yaml, evaluate,
                                            grid, micro_f1, split)
from privyscope_local_llm.config import LocalLLMConfig
from privyscope_local_llm.types import Span

LABELS = {"PER": "person", "LOC": "address"}  # A=PER B=LOC C=NONE


class Oracle:
    """Says PER for 'Alice', NONE otherwise; counts prompts it actually receives."""

    def __init__(self):
        self.seen = 0

    def score(self, prompts):
        self.seen += len(prompts)
        out = []
        for p in prompts:
            cand = p.split("Candidate: ")[1].split("\n")[0]
            out.append({" A": -0.05, " C": -4.0} if cand == "Alice" else {" C": -0.05, " A": -4.0})
        return out


def rec(i):
    text = f"Hi Alice and Table {i}."
    gold = [Span("PER", 3, 8)]
    merged = [Span("PER", 3, 8), Span("PER", 13, 18)]  # second is a false positive
    return text, gold, merged, []


def test_micro_f1_strict():
    m = micro_f1([([Span("A", 0, 1)], [Span("A", 0, 1), Span("A", 2, 3)])])
    assert (m["precision"], m["recall"]) == (0.5, 1.0)


def test_split_is_disjoint_interleaved_and_complete():
    recs = [rec(i) for i in range(10)]
    tune, held = split(recs, 0.5)
    assert len(tune) == len(held) == 5 and {id(r) for r in tune}.isdisjoint({id(r) for r in held})
    assert recs[0] in held and recs[1] in tune  # interleaved, not head/tail
    with pytest.raises(ValueError):
        split(recs, 1.0)


def test_cache_queries_each_prompt_once():
    inner = Oracle()
    c = CachingProvider(inner)
    c.score(["Candidate: x\n", "Candidate: x\n", "Candidate: y\n"])
    c.score(["Candidate: x\n"])
    assert inner.seen == 2 and c.misses == 2


def test_grid_sizes_and_extend_only_varies_accept():
    base = LocalLLMConfig(base_url="x", labels=LABELS)
    assert len(grid(base)) == 3 * 4 * 2
    assert len(grid(base, modes=("verify", "extend"))) == 24 + 72
    assert {c.accept_threshold for c in grid(base)} == {base.accept_threshold}


def test_calibrate_picks_config_that_beats_baseline_without_resending_prompts():
    recs = [rec(i) for i in range(20)]
    base = LocalLLMConfig(base_url="x", labels=LABELS, skip_verified=False)
    inner = Oracle()
    rep = calibrate(recs, inner, grid(base), 0.5)
    assert rep.baseline_held["f1"] < 1.0           # false positives hurt the baseline
    assert rep.best_held.metrics["f1"] == 1.0 and rep.helps
    assert inner.seen == rep.server_calls
    # 24 configs × 10 docs × 2 candidates would be 480 prompts; the cache keeps it far lower
    assert inner.seen < 480 / 10


def test_calibrate_reports_no_help_when_llm_is_useless():
    class AlwaysNone:
        def score(self, prompts):
            return [{" C": -0.01} for _ in prompts]

    recs = [rec(i) for i in range(20)]
    base = LocalLLMConfig(base_url="x", labels=LABELS, skip_verified=False)
    rep = calibrate(recs, AlwaysNone(), grid(base, none_thresholds=(0.5,), temperatures=(1.0,),
                                              skip_verified=(False,)), 0.5)
    assert not rep.helps  # it deletes true positives too


def test_failing_provider_counts_failures_and_keeps_baseline():
    class Down:
        def score(self, prompts):
            raise RuntimeError("down")

    cfg = LocalLLMConfig(base_url="x", labels=LABELS, skip_verified=False)
    res = evaluate(cfg, Down(), [rec(0), rec(1)])
    assert res.failures == 2
    assert res.metrics == micro_f1((g, m) for _t, g, m, _r in [rec(0), rec(1)])


def test_yaml_roundtrip(tmp_path):
    cfg = LocalLLMConfig(base_url="http://x/v1", model="m", mode="extend", temperature=2.0)
    p = tmp_path / "c.yaml"
    p.write_text(config_to_yaml(cfg))
    assert LocalLLMConfig.from_yaml(str(p)) == cfg
