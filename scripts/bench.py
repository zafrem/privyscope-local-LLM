"""Opt-in benchmark against a real server: calls and latency per document, per mode.

    python scripts/bench.py --url http://localhost:8000/v1 --model NAME texts.txt

``texts.txt`` is one document per line. Requires privyscope installed (uses its regex-only
engine on the packaged fixture is not assumed; pass --rules to supply a regex_rules.yaml).
"""
import argparse
import statistics
import time

from privyscope._api import Privyscope
from privyscope._core.regex_filter import RegexFilter
from privyscope_local_llm import LocalLLMConfig, LocalLLMStage


class Counting:
    def __init__(self, inner):
        self.inner, self.calls, self.secs = inner, 0, []

    def score(self, prompts):
        self.calls += len(prompts)
        t = time.perf_counter()
        out = self.inner.score(prompts)
        self.secs.append(time.perf_counter() - t)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rules", required=True, help="regex_rules.yaml for Stage 1")
    a = ap.parse_args()
    docs = [l.rstrip("\n") for l in open(a.texts, encoding="utf-8") if l.strip()]
    for mode in ("verify", "extend"):
        cfg = LocalLLMConfig(base_url=a.url, model=a.model, mode=mode, skip_verified=False)
        prov = Counting(cfg.make_provider())
        eng = Privyscope(RegexFilter.from_yaml(a.rules), stages=[LocalLLMStage(cfg, prov)])
        t = time.perf_counter()
        for d in docs:
            eng.redact(d)
        wall = time.perf_counter() - t
        per = statistics.mean(prov.secs) if prov.secs else 0.0
        print(f"{mode:7s} docs={len(docs)} calls={prov.calls} "
              f"calls/doc={prov.calls/len(docs):.1f} wall={wall:.2f}s mean_batch={per*1000:.0f}ms")


if __name__ == "__main__":
    main()
