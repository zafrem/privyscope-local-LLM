# privyscope-jev — Software Requirements Specification

## 1. Introduction

privyscope-jev is a companion package to privyscope that answers typed questions about a piece of text with calibrated probabilities, without generating text. It verifies and classifies PII candidates; it does not locate spans.

### 1.1 Purpose

This document specifies the requirements for version 0.1 of privyscope-jev: the inference API, the model and its backends, the training and calibration pipeline, the data format, and the acceptance criteria. Numeric targets marked *proposed* are starting points to confirm before implementation.

### 1.2 Scope

In scope:

- A Python package that takes a state and a set of typed questions and returns typed answers with probabilities.
- Three question types: Noul (yes/no), Choice (one of N declared options) and Score (ordered levels).
- An independent implementation trained separately on a pretrained open-weight base model.
- Pluggable inference backends: in-process by default, with optional local serving platforms.
- A synthetic data generator, a training pipeline, calibration and evaluation tooling.
- Korean, English, Japanese and Simplified Chinese.

Out of scope for 0.1:

- Span extraction or masking. privyscope's existing detectors keep that role.
- Text generation of any kind.
- A hosted service, user accounts or telemetry.
- Training a language model from randomly initialised weights.

### 1.3 Definitions

| Term | Meaning |
| --- | --- |
| State | The input being judged: plain text, or a JSON object such as a detected value with its surrounding context. |
| Question | A typed query about the state, with instructions and, for Choice and Score, a list of candidates. |
| Noul | A yes/no question. The answer is P(yes). |
| Choice | A question with 2 to 255 declared options. The answer is a distribution over exactly those options. |
| Score | A question with 2 to 10 ordered levels. The answer is a distribution over levels and its expected value. |
| Candidate | One option of a Choice or one level of a Score. |
| Backend | The component that turns a prompt into a raw score. Backends are interchangeable behind one interface. |
| Calibration | A post-training step that fits a temperature so that reported probabilities match observed frequencies. |
| Soft target | A training label expressed as a probability distribution instead of a single class. |
| Group | A set of records that share a template or entity and must stay in the same data split. |

### 1.4 Relationship to Jev and Open-Jev

Jev is TypeSafe AI's hosted model; its architecture and its RLCD training method are not public. privyscope-jev implements the same interface idea independently. It uses no code, weights or data from Jev or from any Open-Jev repository.

## 2. Overall description

privyscope-jev sits after detection: privyscope finds candidate PII, and privyscope-jev judges each candidate or each column with a probability the caller can threshold.

### 2.1 Product perspective

The package is distributed separately from privyscope and installed as an optional companion. privyscope must keep working when it is absent. The two share entity type names and language codes, and nothing else.

### 2.2 Primary use cases

| ID | Use case | Question type |
| --- | --- | --- |
| UC-1 | False-positive filtering: is this detected value really a person's phone number, or an order number that looks like one? | Noul |
| UC-2 | Type disambiguation: which PII type does this value belong to, including "none"? | Choice |
| UC-3 | Column classification: given a column name and sample values, which PII type does the column hold? | Choice |
| UC-4 | Sensitivity grading: how sensitive is this record, from public to uniquely identifying? | Score |
| UC-5 | Context checks: does this text refer to a real individual, or to a fictional or generic one? | Noul |

### 2.3 Users

- Developers who already use privyscope and want fewer false positives.
- Data engineers who classify columns in bulk and need a probability per decision.
- The maintainer, who trains, calibrates and releases model packages.

### 2.4 Constraints

- The default installation must run on CPU with no GPU, no server and no network access at inference time.
- Every base model and dependency must carry a licence compatible with open-source redistribution.
- Training and evaluation data must be synthetic or openly licensed. No real personal data enters the repository or the released datasets.
- The implementation must stay cleanly separate from any employer code, data or infrastructure.

### 2.5 Assumptions

- A pretrained multilingual base model with adequate Korean, Japanese and Chinese coverage is available under a permissive licence.
- A local teacher LLM is available to the maintainer for producing soft labels during data preparation. End users never need it.
- Probabilities from different backends are not interchangeable. Each backend ships its own calibration.

## 3. System architecture

The runtime path is a thin layer over one backend interface, and everything model-specific lives in the model package that the offline pipeline publishes.

&#91;embedded content: architecture · runtime path, four backends, offline training pipeline\]

The Judge API builds one prompt per candidate, sends them through whichever backend is selected, then applies that backend's calibration and normalises within each question. Training never runs on the user's machine.

| ID | Requirement |
| --- | --- |
| AR-1 | The prompt builder must be the single source of prompt text for training and for every backend. |
| AR-2 | Backends must be loadable by name, and a missing optional dependency must produce a clear install hint. |
| AR-3 | The core package must import without PyTorch installed. |
| AR-4 | The model package format must be the only contract between the training pipeline and the runtime. |

## 4. Functional requirements

Each requirement has an ID for traceability. "Must" is required for 0.1; "should" is desirable and may slip.

### 4.1 Inference API

| ID | Requirement |
| --- | --- |
| FR-API-1 | The API must accept a state as plain text, a JSON object or an array. |
| FR-API-2 | The API must accept a mapping of question ID to definition: type, instructions and, for Choice and Score, criteria. |
| FR-API-3 | Questions must be evaluated independently. Adding or removing one question must not change another question's answer. |
| FR-API-4 | A Noul answer must contain P(yes). A Choice answer must contain the selected option, the full distribution and a confidence value. A Score answer must contain the expected level, the distribution and the level legend. |
| FR-API-5 | A Choice answer must be one of the declared options. Probabilities must sum to 1 within 1e-6. |
| FR-API-6 | Input longer than the model's limit must raise an error. The package must never truncate silently. |
| FR-API-7 | The API must accept a batch of states evaluated against the same questions. |
| FR-API-8 | The same input, model package and backend on the same hardware must give the same probabilities within 1e-6. |
| FR-API-9 | The package must ship ready-made question presets for each supported entity type and language. |
| FR-API-10 | The API should offer an explicit windowing option for long text, returning one answer per window. |

### 4.2 Model

| ID | Requirement |
| --- | --- |
| FR-MOD-1 | The model must consist of a pretrained base and a trained decision component that outputs one scalar score per candidate. |
| FR-MOD-2 | Noul must be computed as sigmoid of the score. Choice and Score must be computed as a softmax over the candidate scores of that question only. |
| FR-MOD-3 | The base model must be selectable by configuration. Base-specific code must be confined to one adapter module. |
| FR-MOD-4 | Two model families must be supported: an encoder cross-encoder with a classification head, and a decoder LLM with a LoRA adapter whose score is the yes-minus-no logit of the next token. |
| FR-MOD-5 | A model package must contain the trained weights or adapter, the calibration temperature, the base model ID with a pinned revision, a hash of the prompt template and a model card. |
| FR-MOD-6 | Loading must fail when the base revision or template hash differs from the one recorded in the package. |

### 4.3 Backends

| ID | Requirement |
| --- | --- |
| FR-BE-1 | All backends must implement one interface that maps a list of prompts to a list of raw scores and declares its limits: batch size and maximum length. |
| FR-BE-2 | An ONNX Runtime CPU backend must be the default and must need no other runtime. |
| FR-BE-3 | A PyTorch backend must support training and GPU inference. |
| FR-BE-4 | An Ollama backend should read yes and no log-probabilities from a one-token completion of a merged, exported model. It must raise an error when either token is missing from the returned log-probabilities. |
| FR-BE-5 | An OpenAI-compatible backend should support servers such as vLLM and the llama.cpp server through the same log-probability method. |
| FR-BE-6 | Each backend must load a calibration fitted on that backend and model export. Using a calibration fitted elsewhere must be refused. |
| FR-BE-7 | The prompt sent by each backend must be byte-identical to the training prompt. A parity test must verify this. |

### 4.4 Training

| ID | Requirement |
| --- | --- |
| FR-TR-1 | The loss must be soft-target cross-entropy plus a Brier term with a configurable weight, default 0.1. |
| FR-TR-2 | For decoder bases, the base weights must stay frozen and only the LoRA adapter is trained. For encoder bases, full fine-tuning is allowed. |
| FR-TR-3 | Training must be resumable from a checkpoint. Resuming must fail when the data hash or the configuration differs. |
| FR-TR-4 | Every run must record its configuration, data hash, base revision, seed and source commit. |
| FR-TR-5 | Training the default model should fit on one 16 GB consumer GPU (*proposed*). |

### 4.5 Calibration

| ID | Requirement |
| --- | --- |
| FR-CAL-1 | The temperature must be fitted on the calibration split only. |
| FR-CAL-2 | Calibration must report expected calibration error and Brier score per question type and per language. |
| FR-CAL-3 | A recalibration command must exist for exported or quantised models, because export changes the probabilities. |

### 4.6 Evaluation

| ID | Requirement |
| --- | --- |
| FR-EV-1 | Evaluation must report accuracy, F1, AUROC, expected calibration error and Brier score, broken down by language, entity type and data source. |
| FR-EV-2 | Test and out-of-distribution results must be reported separately and never pooled. |
| FR-EV-3 | Evaluation must compare privyscope detection alone against detection followed by privyscope-jev, reporting the change in precision and recall. |
| FR-EV-4 | A latency benchmark must report P50 and P95 by context length and number of candidates, per backend. |
| FR-EV-5 | A backend parity test must report how many decisions differ from the PyTorch reference backend. |

## 5. Data requirements

Training data is one JSONL record per decision, with a probability distribution as the target and a group ID that controls splitting.

### 5.1 Record schema

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique record ID. |
| `group_id` | string | Template or entity family. All records of a group share one split. |
| `split` | string | One of `train`, `calibration`, `validation`, `test`, `ood`. |
| `source` | string | Name of the generator or imported dataset. |
| `language` | string | `ko`, `en`, `ja` or `zh-Hans`. |
| `state` | object or string | The input being judged. |
| `question` | string | Instructions for this decision. |
| `kind` | string | `noul`, `choice` or `score`. |
| `options` | array of strings | Candidates. For Noul, exactly `["no", "yes"]`. |
| `target` | array of numbers | Probability per option, summing to 1. |
| `metadata` | object | Provenance, licence, target basis, entity type. |

```json
{"id": "ko-phone-000123", "group_id": "ko-phone-tpl-17", "split": "train",
 "source": "synthetic-ko-v1", "language": "ko",
 "state": {"text": "주문번호 010-2345-6789 건 배송 문의드립니다.", "span": "010-2345-6789", "detected_as": "phone"},
 "question": "Is the span a phone number that belongs to a person?",
 "kind": "noul", "options": ["no", "yes"], "target": [0.85, 0.15],
 "metadata": {"entity_type": "phone", "target_basis": "teacher_soft", "license": "CC0-1.0"}}
```

### 5.2 Requirements

| ID | Requirement |
| --- | --- |
| DR-1 | A validator must reject records with missing fields, duplicate IDs, targets that do not sum to 1, or duplicate options. |
| DR-2 | A group must never appear in more than one split. |
| DR-3 | The same state, question and kind must never appear in more than one split, regardless of option order. |
| DR-4 | Every record must carry provenance: generator version, seed, template ID and licence. |
| DR-5 | A generator must exist per language and must produce locale-correct formats for each entity type, using synthetic values only. |
| DR-6 | Hard negatives must make up at least 30% of training records (*proposed*): order and tracking numbers, product codes, business names that resemble personal names, fictional and generic persons. |
| DR-7 | Labels derived from generator rules must stay hard. Labels for ambiguous cases may be soft distributions from one or more local teacher LLMs. `target_basis` must record which. |
| DR-8 | The OOD split must hold templates, formats and domains that are absent from training. |
| DR-9 | Choice records must shuffle option order per record so that position carries no signal. |
| DR-10 | A release check must confirm that every record is synthetic or openly licensed before a dataset is published. |
| DR-11 | Version 0.1 should have at least 20,000 training records per language (*proposed*). |

## 6. External interfaces

The package exposes a Python API, a command-line tool and a hook that privyscope can call after detection.

### 6.1 Python API

```python
from privyscope_jev import Judge, Noul, Choice, Score

judge = Judge.load("privyscope-jev-base")  # default: ONNX Runtime on CPU

result = judge.ask(
    state={"text": "배송 문의: 010-2345-6789로 연락 주세요.", "span": "010-2345-6789"},
    questions={
        "is_phone": Noul("Is the span a phone number that belongs to a person?"),
        "pii_type": Choice("Which PII type is the span?",
                           options={"phone": "Personal phone number",
                                    "order_id": "Order or tracking number",
                                    "none": "Not personal data"}),
        "sensitivity": Score("How sensitive is this record?",
                             levels=["Public", "Internal", "Sensitive", "Uniquely identifying"]),
    },
)

result["is_phone"].probability      # float in [0, 1]
result["pii_type"].choice           # one of the declared option keys
result["pii_type"].probabilities    # dict of option key to probability
result["sensitivity"].score         # expected level index
```

| ID | Requirement |
| --- | --- |
| IF-PY-1 | `Judge.load` must accept a local path or a Hugging Face model ID and an optional backend name. |
| IF-PY-2 | `Judge.ask` must take one state; `Judge.ask_batch` must take a list of states. |
| IF-PY-3 | Answer objects must be typed and serialisable to JSON. |
| IF-PY-4 | Errors must be specific exception types: input too long, invalid question, backend unavailable, package mismatch. |

### 6.2 Command line

| Command | Purpose |
| --- | --- |
| `privyscope-jev ask` | Answer questions for one state or a JSONL file of states. |
| `privyscope-jev data build` | Generate synthetic records for a language and entity set. |
| `privyscope-jev data validate` | Run the schema and split checks from section 5. |
| `privyscope-jev train` | Train or resume a model from a configuration file. |
| `privyscope-jev calibrate` | Fit the temperature for a model and backend. |
| `privyscope-jev eval` | Produce the metrics report from section 4.6. |
| `privyscope-jev export` | Export to ONNX, or merge an adapter and export for a serving platform. |
| `privyscope-jev bench` | Run the latency benchmark. |

### 6.3 privyscope integration

| ID | Requirement |
| --- | --- |
| IF-PS-1 | privyscope must discover privyscope-jev at runtime and work unchanged when it is not installed. |
| IF-PS-2 | A verifier hook must take privyscope detections and return each detection with a probability and a keep or drop decision. |
| IF-PS-3 | Thresholds must be configurable per entity type, with defaults shipped in the model package. |
| IF-PS-4 | Entity type names and language codes must match privyscope's. |

### 6.4 Serving platforms

| Platform | Model family | How the score is obtained |
| --- | --- | --- |
| ONNX Runtime, in process | Encoder | Head output. Default. |
| PyTorch, in process | Both | Head output or yes-minus-no logit. Reference backend. |
| Ollama | Decoder | Log-probabilities of yes and no from a one-token completion. |
| vLLM, llama.cpp server | Decoder | Same, through the OpenAI-compatible API. |

## 7. Non-functional requirements

The default path must be small, offline and CPU-only; everything heavier is optional.

### 7.1 Performance

| ID | Requirement |
| --- | --- |
| NFR-P-1 | Default backend, one Noul on a 256-token state, 4 CPU cores: P50 at or below 50 ms and P95 at or below 150 ms (*proposed*). |
| NFR-P-2 | Default model package size at or below 1.5 GB on disk (*proposed*). |
| NFR-P-3 | Peak memory for the default backend at or below 3 GB (*proposed*). |
| NFR-P-4 | Batch throughput must scale with batch size until the backend's declared limit, with no per-item model reload. |
| NFR-P-5 | Cost grows with context length times candidate count. The documentation must state this and give measured numbers. |

### 7.2 Security and privacy

| ID | Requirement |
| --- | --- |
| NFR-S-1 | Inference must make no network calls unless the user selects a remote backend. Model download is a separate, explicit step. |
| NFR-S-2 | The package must never log, cache to disk or transmit state content. Logs may contain IDs, lengths and timings only. |
| NFR-S-3 | There must be no telemetry. |
| NFR-S-4 | Remote backends must default to loopback addresses. A non-loopback address must require an explicit option. |
| NFR-S-5 | Model files must be loaded from safetensors or ONNX only. Pickle-based formats must be refused. |
| NFR-S-6 | Model packages must ship a manifest of file hashes, verified at load. |
| NFR-S-7 | The documentation must state that a probability is not an authorisation decision and that typed output prevents malformed answers, not wrong ones. |

### 7.3 Portability and compatibility

| ID | Requirement |
| --- | --- |
| NFR-C-1 | Python 3.10 or newer on Linux, macOS and Windows for the default backend. |
| NFR-C-2 | Core dependencies limited to ONNX Runtime, a tokenizer library and NumPy. PyTorch, PEFT and serving clients are optional extras. |
| NFR-C-3 | Dependency versions must be given as ranges for the core and may be pinned only inside the training extra. |
| NFR-C-4 | Model packages must carry a format version. A newer package on an older library must fail with a clear message. |

### 7.4 Maintainability and quality

| ID | Requirement |
| --- | --- |
| NFR-M-1 | Unit tests must cover prompt construction, probability normalisation, schema validation and split isolation. |
| NFR-M-2 | Continuous integration must run the tests and a small CPU inference smoke test on every change. |
| NFR-M-3 | Every released model must have a model card with training data summary, metrics by language, known weaknesses and calibration details. |

### 7.5 Licensing

| ID | Requirement |
| --- | --- |
| NFR-L-1 | Code under a permissive open-source licence consistent with privyscope. |
| NFR-L-2 | Base models limited to licences that allow redistribution of derived adapters, such as Apache 2.0 or MIT. |
| NFR-L-3 | Generated datasets released under CC0 or an equivalent, with third-party sources listed separately. |

## 8. Acceptance criteria

Version 0.1 is accepted when every row below passes on the held-out test and OOD splits. All thresholds are *proposed* and should be set once a first baseline exists.

| ID | Criterion | Threshold |
| --- | --- | --- |
| AC-1 | Noul F1 on the test split, each language | At least 0.95 |
| AC-2 | Noul F1 on the OOD split, each language | At least 0.88 |
| AC-3 | Choice accuracy on the test split, each language | At least 0.93 |
| AC-4 | Expected calibration error after calibration, each question type | At most 0.05 |
| AC-5 | False positives removed when used after privyscope detection | At least 50% |
| AC-6 | Recall lost when used after privyscope detection | At most 2 percentage points |
| AC-7 | Gap between the best and worst language on AC-1 | At most 0.03 |
| AC-8 | Decisions that differ between the default backend and the PyTorch reference | At most 0.5% |
| AC-9 | Performance requirements NFR-P-1 to NFR-P-3 | Met on the reference machine |
| AC-10 | Schema, split-isolation and no-network tests | All pass |
| AC-11 | Result on real-world text | Measured on a small, openly licensed or hand-written natural sample per language and reported beside the synthetic results |

AC-11 has no pass threshold in 0.1. Synthetic scores overstate real-world quality, so the natural-text number must be published even when it is lower.

## 9. Milestones, risks and open questions

The work runs in five phases, and each phase ends at a gate tied to the acceptance criteria. No dates are set yet.

### 9.1 Milestones

&#91;embedded content: roadmap · 5 phases, each closed by a gate\]

Phases 1 to 4 deliver version 0.1 on the encoder path. Phase 5 adds the decoder path and the serving-platform backends and can ship later without changing the public API.

### 9.2 Risks

| ID | Risk | Mitigation |
| --- | --- | --- |
| R-1 | Synthetic scores overstate real-world quality. | Natural-text sample per language (AC-11), hard negatives (DR-6). |
| R-2 | Template leakage inflates test results. | Group-level splits (DR-2, DR-3) and a separate OOD split (DR-8). |
| R-3 | Export or quantisation shifts probabilities. | Per-backend calibration (FR-BE-6) and a recalibration command (FR-CAL-3). |
| R-4 | Latency grows with context length times candidate count. | Small option sets in presets, batching, measured numbers in the docs (NFR-P-5). |
| R-5 | A base model's licence or availability changes. | Pinned revisions (FR-MOD-5) and a swappable base (FR-MOD-3). |
| R-6 | Teacher LLM labels carry systematic bias. | Rule-derived labels stay hard (DR-7), more than one teacher, a hand-audited sample. |
| R-7 | Serving platforms change their log-probability output between versions. | Version check at backend start; these backends stay optional. |

### 9.3 Open questions

- [ ] Package name: is "privyscope-jev" acceptable given that Jev is TypeSafe AI's product name, or should the package take a neutral name?
- [ ] Default encoder base: mmBERT, mDeBERTa-v3 or a multilingual reranker?
- [ ] Decoder base for phase 5: a small Gemma-class model or another permissively licensed one?
- [ ] Entity types in 0.1: the full privyscope set or a starting subset?
- [ ] Reference machine for the performance targets in section 7.1.
- [ ] Confirm or replace every threshold marked *proposed* once the phase 3 baseline exists.

