# privyscope-local-llm

Connect [privyscope](https://github.com/zafrem/privyscope) to a local LLM you already run.
The LLM never generates text: for each PII candidate it reads one next-token
log-probability distribution over a fixed option list (the
[open-jev](https://github.com/zefan-cai/open-jev) idea), so there is no JSON to parse and
each check is a single short forward pass. No weights, no torch.

```bash
pip install privyscope-local-llm
export PRIVYSCOPE_LOCAL_LLM_URL=http://localhost:8000/v1   # vLLM, LM Studio, llama.cpp server, Ollama /v1
export PRIVYSCOPE_LOCAL_LLM_MODEL=<model name your server exposes>
```

Once the URL is set, every `privyscope` engine runs the stage automatically (it registers under the
`privyscope.stages` entry point). With no URL set it does nothing and makes no requests.
Needs a `privyscope` core that has the `privyscope.stages` hook.

## Modes

| `mode` | Candidates | Effect | Cost |
|---|---|---|---|
| `verify` (default) | spans regex/NER already found | drops false positives, re-labels | calls ≤ candidates |
| `extend` | the above + whitespace n-gram windows | can also add spans | more calls; weak for CJK (no spaces) |

`skip_verified` (default on) sends regex hits straight through without asking the LLM.

## Config

YAML via `PRIVYSCOPE_LOCAL_LLM_CONFIG=/path/cfg.yaml`, or `LocalLLMConfig(...)` in code
(`base_url`, `model`, `provider` = `openai`|`ollama`, `mode`, `none_threshold`, `accept_threshold`,
`skip_verified`, `temperature`, `context_chars`, `max_ngram`, `max_candidates`, `timeout`, `concurrency`).

```python
from privyscope import Privyscope
from privyscope_local_llm import LocalLLMConfig, LocalLLMStage

cfg = LocalLLMConfig(base_url="http://localhost:8000/v1", model="qwen2.5-7b", mode="extend")
engine = Privyscope.from_pretrained(lang="en", stages=[LocalLLMStage(cfg)])
```

## Calibrate (no training)

Defaults are untuned guesses and every model's logprobs are scaled differently. Sweep
temperature, thresholds and `skip_verified` on labelled data (same JSONL as `privyscope eval`):

```bash
python -m privyscope_local_llm calibrate val.jsonl --lang ko \
    --url http://localhost:8000/v1 --model <name> --write-config tuned.yaml
export PRIVYSCOPE_LOCAL_LLM_CONFIG=tuned.yaml
```

The core pipeline runs once per document; each grid point is replayed offline over cached
logprobs, so the server sees each distinct prompt once. Settings are chosen on half the data
and scored on the other half. `tuned.yaml` is written only if the best setting beats the
no-stage baseline on the held-out half; otherwise the command exits 2 and says not to enable
the stage. Add `--modes verify,extend` to also sweep `extend` (many more server calls).
Use 200+ records: with fewer, the held-out result is too noisy to trust.

## Behavior and limits

- If the server is unreachable or returns no logprobs, the core skips the stage and keeps the
  original spans. An LLM failure never removes detections.
- The server must return logprobs for the first token (`logprobs` on `/v1/completions`).
  Ollama support depends on its version and has not been verified against a live server.
- Speed notes (prefix caching, fewer calls) are reasoning, not measurements. Measure with
  `python scripts/bench.py --url ... --model ... --rules regex_rules.yaml texts.txt`.
