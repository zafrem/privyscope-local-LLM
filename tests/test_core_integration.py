"""The stage plugged into a real privyscope engine (skipped if the core isn't importable)."""
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[2] / "privyscope"
sys.path.insert(0, str(CORE))
privyscope = pytest.importorskip("privyscope")

from privyscope._api import Privyscope  # noqa: E402
from privyscope._core.regex_filter import RegexFilter  # noqa: E402

from privyscope_local_llm.config import LocalLLMConfig  # noqa: E402
from privyscope_local_llm.stage import LocalLLMStage  # noqa: E402

RULES = CORE / "tests" / "fixtures" / "regex_rules.min.yaml"


class Always:
    def __init__(self, letter):
        self.letter = letter

    def score(self, prompts):
        return [{f" {self.letter}": -0.01} for _ in prompts]


def test_stage_runs_inside_real_engine_and_trusts_regex_spans():
    cfg = LocalLLMConfig(base_url="x")  # NONE for everything, but regex spans are skipped
    stage = LocalLLMStage(cfg, Always("I"))  # 8 labels → I is NONE
    eng = Privyscope(RegexFilter.from_yaml(RULES), stages=[stage])
    assert eng.redact("call 555-123-4567").redacted_text == "call <PHONE>"


def test_llm_can_veto_when_not_skipping_verified():
    stage = LocalLLMStage(LocalLLMConfig(base_url="x", skip_verified=False), Always("I"))
    eng = Privyscope(RegexFilter.from_yaml(RULES), stages=[stage])
    assert eng.redact("call 555-123-4567").redacted_text == "call 555-123-4567"
