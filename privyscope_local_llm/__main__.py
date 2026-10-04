"""``python -m privyscope_local_llm calibrate ...``"""
from __future__ import annotations

import argparse
import sys

from .config import LocalLLMConfig


def _collect(args):
    """Run the real pipeline once per document, capturing merged + regex spans."""
    from privyscope import Privyscope
    from privyscope._eval.dataset import read_eval_jsonl

    captured = []

    class Recorder:
        def refine(self, text, spans, regex_spans):
            captured.append((list(spans), list(regex_spans)))
            return list(spans)

    # explicit stages=[...] also stops auto-discovery from applying the installed stage twice
    engine = Privyscope.from_pretrained(
        lang=args.lang, regex_only=args.regex_only, regex_rules=args.regex_rules, stages=[Recorder()]
    )
    records = []
    for text, gold in read_eval_jsonl(args.dataset, limit=args.limit):
        engine.redact(text)
        merged, regex_spans = captured.pop()
        records.append((text, gold, merged, regex_spans))
    return records


def _cmd_calibrate(args) -> int:
    from .calibrate import CachingProvider, calibrate, config_to_yaml, grid

    base = LocalLLMConfig(base_url=args.url, model=args.model, provider=args.provider,
                          context_chars=args.context_chars, max_ngram=args.max_ngram,
                          max_candidates=args.max_candidates, timeout=args.timeout)
    records = _collect(args)
    if len(records) < 20:
        print(f"warning: only {len(records)} records; the held-out split will be too small to trust",
              file=sys.stderr)
    configs = grid(base, modes=tuple(args.modes.split(",")))
    print(f"{len(records)} records, {len(configs)} configs, tune={args.tune_fraction:.0%}")
    rep = calibrate(records, CachingProvider(base.make_provider()), configs, args.tune_fraction)

    f = lambda m: f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"  # noqa: E731
    print(f"server calls: {rep.server_calls}")
    print(f"baseline (no stage)   tune: {f(rep.baseline_tune)}   held-out: {f(rep.baseline_held)}")
    if rep.best is None:
        print("no configs evaluated")
        return 1
    c = rep.best.config
    print(f"best on tune: mode={c.mode} temperature={c.temperature} none_threshold={c.none_threshold} "
          f"skip_verified={c.skip_verified} accept_threshold={c.accept_threshold}  {f(rep.best.metrics)}")
    print(f"best on held-out:                                    {f(rep.best_held.metrics)}"
          f"   (failures: {rep.best_held.failures})")
    if not rep.helps:
        print("RESULT: the stage does not beat the baseline on held-out data; don't enable it "
              "for this model/language.")
        return 2
    print("RESULT: improves held-out F1.")
    if args.write_config:
        c.api_key = None  # never write credentials into the tuned config
        with open(args.write_config, "w", encoding="utf-8") as fh:
            fh.write(config_to_yaml(c))
        print(f"wrote {args.write_config}  (use via PRIVYSCOPE_LOCAL_LLM_CONFIG)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="privyscope_local_llm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="sweep temperature/thresholds/mode on labelled JSONL")
    c.add_argument("dataset", help='JSONL: {"text": ..., "spans": [{"label","start","end"}]}')
    c.add_argument("--url", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    c.add_argument("--lang", default=None)
    c.add_argument("--regex-only", action="store_true", help="skip NER weights")
    c.add_argument("--regex-rules", default=None)
    c.add_argument("--modes", default="verify", help="comma list; 'verify,extend' costs many more calls")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--tune-fraction", type=float, default=0.5)
    c.add_argument("--context-chars", type=int, default=200)
    c.add_argument("--max-ngram", type=int, default=3)
    c.add_argument("--max-candidates", type=int, default=64)
    c.add_argument("--timeout", type=float, default=30.0)
    c.add_argument("--write-config", default=None)
    args = ap.parse_args(argv)
    return _cmd_calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
