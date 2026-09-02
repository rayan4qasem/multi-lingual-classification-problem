# docrouter — تصنيف وتوجيه الوثائق الحكومية

[![CI](https://github.com/rayan4qasem/multi-lingual-classification-problem/actions/workflows/ci.yml/badge.svg)](https://github.com/rayan4qasem/multi-lingual-classification-problem/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org/)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-d7ff64)](https://docs.astral.sh/ruff/)

Routes Arabic government documents to the Saudi institution responsible for
handling them: police, health, courts, prosecution, tax, municipal, and so on.

Handles mixed input — digital PDFs and DOCX go straight to text, scanned PDFs
and images are OCR'd first. Classification runs on the Claude API with
structured output constrained to the institution list, so the model cannot
invent a destination that doesn't exist.

## Status

Working scaffold, end to end, on **mock data**. The taxonomy is real; the
documents are synthetic. Before anyone quotes an accuracy number, it needs to
be re-measured on real correspondence — see [Next steps](#next-steps).

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Set your key (`ANTHROPIC_API_KEY`), or copy `.env.example` to `.env`.

## Quick start

```bash
docrouter taxonomy
```

```bash
docrouter generate --engine curated
```

```bash
docrouter train-baseline
```

```bash
docrouter classify data/generated/mock.jsonl --backend llm --limit 25
```

```bash
docrouter evaluate --show-confusion
```

## How it fits together

```
file (pdf/docx/image/txt)
   │
   ├─ ingest.py ──── digital text layer? ──── yes ──► text
   │                        │
   │                        no
   │                        ▼
   │                 rasterize @200dpi ──► OCR (Claude vision, or Tesseract)
   ▼
normalize.light()          strips diacritics, tatweel, presentation forms
   │
   ▼
classify/llm.py            system prompt = taxonomy catalogue (cached)
   │                       output constrained to a json_schema enum of ids
   ▼
Prediction                 institution_id · confidence · rationale_ar · alternatives
   │
   ├─ confidence ≥ threshold ──► routed automatically
   └─ confidence <  threshold ──► held for human review
```

### The pieces

| File | Does |
|---|---|
| `config/taxonomy.yaml` | The institution list. **Edit this first** — prompts, labels, mock data and metrics all derive from it. |
| `taxonomy.py` | Loads and validates it; renders the catalogue block for the prompt. |
| `ingest.py` | File → text. Per-page decision on whether OCR is needed. |
| `normalize.py` | Two strengths: `light` for the LLM, `aggressive` for bag-of-words. |
| `classify/llm.py` | Claude backend. Sync path and Batches path share one prompt. |
| `classify/baseline.py` | TF-IDF + calibrated SVM. Offline reference point. |
| `data/curated/` | 86 hand-authored Arabic documents, incl. 16 boundary cases. |
| `mockdata.py` | Dataset engines — curated, template and LLM. |
| `evaluate.py` | Scoring, with auto-routed vs held-for-review separated. |
| `threshold.py` | Cut-off sweep, calibration check, held-out validation. |
| `fewshot.py` | Gold labels → cached few-shot examples, with redaction. |
| `labeling/store.py` | Append-only label store. Decisions only, no text. |
| `labeling/prioritize.py` | Builds the review batch: priority lane + random lane. |
| `labeling/review.py` | Local, loopback-only review UI (RTL, keyboard-driven). |

## Architecture

The swappable parts are defined as `typing.Protocol` contracts in
[`protocols.py`](src/docrouter/protocols.py) — structural, so an implementation
never has to import or subclass anything from this package.

**Two registries carry the extensibility.** `ExtractorRegistry` maps file
suffixes to extractors and `OcrRegistry` maps names to OCR backends, so adding
a format or a transcription engine is a registration rather than an edit to a
dispatch chain. `ClassifierRegistry` does the same for backends — an on-prem
fine-tune would register alongside `llm` and `baseline` without the CLI
changing. Tests register throwaway implementations at runtime to prove it.

**The classifier contract is split deliberately.** `Classifier` is the narrow
interface every backend satisfies; `BatchClassifier` adds the Batches API. The
offline baseline has no notion of an async batch, and is not made to grow stub
methods to satisfy one fat interface — it simply is not a `BatchClassifier`,
and `docrouter batch` checks for the narrower type rather than assuming it.

**Policies are injected, not hardcoded.** `PdfExtractor(is_usable=...)` decides
whether a page's text layer is good enough; the default requires real Arabic,
but "usable" is a deployment policy and a bilingual archive would want another.
The Claude OCR backend takes its client rather than constructing one, so no API
object exists until a document actually needs transcribing.

**Rendering is separate from the CLI.** [`reporting.py`](src/docrouter/reporting.py)
takes domain objects and returns Rich tables — no I/O, no `sys.exit`, no Typer.
That is why the tables are unit-tested without invoking a command.

## Development

```bash
uv pip install -e ".[dev]"
```

```bash
ruff check . && ruff format --check . && mypy && pytest --cov
```

CI runs on every push and pull request:

| job | does |
|---|---|
| **lint & types** | `ruff check`, `ruff format --check`, `mypy` |
| **tests** | pytest on Python 3.11 / 3.12 / 3.13 × Ubuntu / Windows, coverage gate at 85% |
| **cli smoke** | drives the whole offline pipeline end to end with no API key |
| **secrets** | fails if a credential or a real document is ever committed |

Windows is a first-class target in the matrix — it is where this runs today,
and the reason OCR defaults to a backend needing no external binaries. The
test suite sets `ANTHROPIC_API_KEY` to empty in CI, so any test that starts
reaching for the network fails loudly instead of silently costing money.

## Design decisions worth knowing

**The taxonomy is config, not code.** Adding an institution is a YAML edit.
Nothing downstream hardcodes an institution id.

**Output is schema-constrained.** `institution_id` is a JSON-schema `enum` of
the ids in the taxonomy, so the model physically cannot return a destination
outside the list — no string matching on free text, no fuzzy id repair.

**Confusion pairs are declared, not discovered.** `confusion_pairs` in the
taxonomy feeds both the prompt (as explicit tie-breakers) and the eval report
(as a targeted error count), so the pairs that actually matter — prosecution
vs. courts, labour vs. GOSI, tax vs. commerce — are tracked directly rather
than buried in a 14×14 matrix.

**The prompt prefix is byte-stable and cached.** The catalogue is a single
cached system block; per-document cost is roughly the document itself. There's
a test asserting the block doesn't vary between calls, because a silent cache
miss triples the bill without failing anything.

**Accuracy is not the headline metric.** Misrouting a document to the wrong
ministry costs much more than holding it for a clerk. `evaluate` reports
accuracy over the *auto-routed* subset alongside overall accuracy, plus how
many held documents would have been right anyway — that pair is what you tune
the threshold against.

**OCR defaults to Claude vision, not Tesseract.** No external binaries (a real
constraint on Windows), and it handles Arabic ligatures and handwriting
considerably better. Tesseract is available via `pip install '.[tesseract]'`
if you need OCR to stay local.

## Mock data

Three engines, answering different questions:

- **`--engine curated`** (default) — 86 hand-authored documents shipped in
  `data/curated/`, written to read like real correspondence rather than drawn
  from phrase pools. Covers all 14 institutions and includes **16 adversarial
  boundary cases**, two per declared confusion pair, each labeled with the
  institution that is actually competent while carrying the surface signals of
  its partner. Free, offline, deterministic, no API key.
- **`--engine template`** — combinatorial generator, unlimited volume. Use it
  to exercise the plumbing. Its bodies come from per-institution pools, so
  keyword models score unrealistically well; don't quote accuracy from it.
- **`--engine llm`** — Claude writes fresh correspondence at whatever volume
  you ask. Needs `ANTHROPIC_API_KEY`. Use it to scale the corpus up once the
  curated set has told you the pipeline discriminates.

### Is the curated set actually harder?

`python scripts/corpus_difficulty.py` trains the offline baseline on the
template corpus and applies it to each set:

| evaluation | n | accuracy | macro F1 |
|---|---:|---:|---:|
| templates → templates (held out) | 81 | 100.0% | 1.000 |
| templates → curated (all) | 86 | 82.6% | 0.830 |
| templates → curated (**hard only**) | 44 | 65.9% | 0.640 |

A keyword-ish model saturates on the templates and loses a third of its
accuracy on the boundary cases — and its errors land on the pairs declared in
the taxonomy (prosecution vs. police 3, labour vs. GOSI 2, tax vs. commerce
2). That gap is the benchmark. A test asserts it stays there.

`--hard-only` gives you just the 44 boundary documents:

```bash
docrouter generate --engine curated --hard-only --out data/generated/hard.jsonl
```

## Feeding confirmed labels back into the prompt

Once a gold set exists, it becomes few-shot examples:

```bash
docrouter prompt build ./incoming --predictions runs/predictions.jsonl
```

```bash
docrouter prompt show --count-tokens
```

```bash
docrouter classify ./new-batch --examples data/prompt/examples.json
```

**Selection favours corrections, not confirmations.** A document the model
already got right teaches it little; a document a reviewer *overrode* teaches
a boundary it got wrong. Coverage comes first — every institution with gold
gets at least one example — then the remaining budget goes to overrides,
which is exactly where the prosecution-vs-police and labour-vs-GOSI calls
live. Built from the curated corpus, 16 of 18 selected examples are
corrections, and each one names the mistake in the prompt:

> `### مثال 7 ← mci_commerce (وزارة التجارة)`
> `تنبيه: صُنّفت خطأً على أنها zatca (هيئة الزكاة والضريبة والجمارك)، والصواب ما هو مثبت أعلاه.`

**Examples are fixed, not retrieved per document.** Nearest-neighbour
retrieval would pick better examples for any single document, but it changes
the prompt prefix on every call — which invalidates the server-side cache and
multiplies the per-document cost. At 14 classes a fixed set captures most of
the benefit and keeps the prefix byte-stable. Selection is sorted, never
shuffled, and a test asserts the rendered block is byte-identical across runs.
The examples join the same single cached block as the taxonomy, so they are
paid for once per cache window rather than once per document.

**Redaction is on by default.** Examples embed real document text and are sent
on every request, which is a meaningful escalation over sending one document
at a time. National IDs, phone numbers, IBANs, tax numbers and emails are
replaced with typed placeholders; dates and amounts survive because they carry
routing signal. `--no-redact` turns it off and warns.

**Leakage guard.** `classify --examples` warns when a document appears both in
the example set and in the run being scored — a model evaluated on documents
sitting in its own prompt looks better than it is.

### Does it actually help?

Unmeasured. The wiring is tested; the benefit is not. Establish it with an
A/B on documents *not* in the example set:

```bash
docrouter classify data/holdout.jsonl --backend llm --out runs/base.jsonl
```

```bash
docrouter classify data/holdout.jsonl --backend llm --examples data/prompt/examples.json --out runs/shot.jsonl
```

Then `docrouter evaluate` each. Few-shot examples are not automatically an
improvement — they can bias the model toward the classes they over-represent.
Treat the A/B as the deciding evidence, not the assumption.

## Choosing the threshold

```bash
docrouter threshold --dataset data/generated/curated.jsonl --predictions runs/preds.jsonl
```

The auto-route cut-off is the only knob that trades one harm for another, so
it is set two ways rather than guessed:

- **target mode** — `--target-auto-accuracy 0.95` finds the lowest cut-off
  meeting that bar, so coverage is as high as the bar allows. If nothing
  reaches it, that is reported as a real answer: the model cannot support
  that SLA on this data at any threshold.
- **cost mode** — `--misroute-cost 20 --review-cost 1` minimises expected
  cost. This is where the asymmetry gets stated explicitly instead of hiding
  inside a default.

Two guards run alongside it:

**Calibration.** A threshold on a confidence score is meaningless unless the
score tracks reality. The command reports reliability bins and ECE first, and
warns when confidence is not worth thresholding. On the shipped baseline it
immediately catches that the model is badly *under*-confident (ECE 0.32 — it
says 0.46 and is right 100% of the time), which is exactly the situation where
a sensible-looking cut-off silently holds half the archive for no gain.

**Held-out validation.** Picking a threshold on the same documents you measure
it on flatters the result, so the cut-off is chosen on one half and reported
on the other. The optimism gap is printed; on the baseline it is +8%.

`--per-class` fits a cut-off per institution, which usually dominates a single
global one — the model is not equally trustworthy everywhere. It needs enough
labeled documents per class, and marks the ones too thin to trust.

### What this found about the shipped default

The `0.55` default the repo has been carrying is wrong for the baseline, and
the sweep says so plainly:

| threshold | coverage | auto accuracy | misrouted | held |
|---:|---:|---:|---:|---:|
| **0.40** | 57% | 98.0% | 1 | 37 |
| 0.55 (old default) | 44% | 97.4% | 1 | 48 |

0.55 holds eleven more documents to buy nothing. That is the placeholder the
"Next steps" section warned about, caught by measurement rather than opinion.

## Labeling real documents

The loop that turns an unlabeled archive into ground truth:

```
real documents ──► prelabel ──► queue ──► review (local UI) ──► label store
                      ▲                                              │
                      └──────── retrain / re-prompt ◄──── export gold┘
```

```bash
docrouter label prelabel ./incoming --size 50
```

```bash
docrouter label review --reviewer "اسم المراجع"
```

```bash
docrouter label status
```

```bash
docrouter label export ./incoming
```

`export` writes a gold dataset that `evaluate` and `train-baseline` already
consume, so the loop closes without any glue.

### Two lanes, and why it matters

**You should not label everything.** The model already gets the easy documents
right; confirming them teaches nothing. The **priority lane** ranks candidates
by how much a label would be worth — model uncertainty, a thin margin between
top-1 and top-2, disagreement between the LLM and the offline baseline, and
whether the top two form a confusion pair declared in the taxonomy. A
diversity term stops the queue filling with fifty copies of the same form.

But a queue ranked by difficulty is a **biased sample**, so agreement measured
on it understates real accuracy. That is what the **random lane** is for: a
uniform sample of the pool, sized by `--random-ratio` (default 20%). It is the
only thing here that yields an honest accuracy estimate. Don't set it to zero.

The dry run makes the gap concrete — same model, same session:

| lane | agreement | reading |
|---|---:|---|
| random | 100% (n=10) | what the model actually does on representative mail |
| priority | 53.6% (n=28) | the hard cases, which is the point of the lane |

Report the random-lane number. `label status` prints it with a Wilson
interval, because "100%" off ten documents is really 72–100%.

### Anchoring

Random-lane documents are served **blind**: the model's prediction is stripped
from the payload server-side, not merely hidden in the page. A reviewer who
never saw the suggestion cannot have been nudged into agreeing with it, which
is what makes that lane a genuine independent measurement. The guess is
revealed after the decision is recorded. `--no-blind-random` turns it off.

### Privacy

Real documents mean real citizen data, so:

- The review UI is a **local** server bound to `127.0.0.1`. Nothing is
  published, and there are no CDN or outbound requests in the page.
- The label store records **decisions and file references, never document
  text** — so it can be shared with an auditor without shipping the documents.
- Review queues and gold sets *do* embed text and are gitignored.
- Note that `--backend llm` and Claude-vision OCR send document content to the
  API. Use `--backend baseline` or `--backend none` for material that must not
  leave the network.

### Cold start

No API key and no trained model yet? `--backend none` builds a pure-random
queue so labeling can begin immediately; the priority signals switch on as
soon as there is a model to disagree with.

## Cost

The classifier defaults to `claude-opus-5`. For a high-volume archive sweep,
two levers, in this order:

1. `docrouter batch submit` — the Batches API is **50% cheaper** and returns
   within the hour. This is the right path for anything non-interactive.
2. `--effort low` — classification is a task where low effort usually holds
   quality. Measure on a labeled slice before making it the default.

Model choice is a third lever (`--model claude-sonnet-5`), but measure it
against the same slice rather than assuming.

## Tests

```bash
uv run pytest -q
```

Everything offline — no API calls, no key needed.

## Next steps

1. **Get real labeled documents.** Everything below depends on this. The
   labeling loop above is built and tested; point `label prelabel` at a real
   folder and start. Aim for ~25/class before trusting any per-class number.
2. **Tune the review threshold** on that real slice — the 0.55 default is a
   placeholder, not a finding. Once the gold set exists, sweep it against the
   auto-routed accuracy that `evaluate` reports.
3. **Verify OCR quality on real scans.** Simulated noise is not the same as a
   1998 fax of a stamped form.
4. **Confirm data residency.** Documents currently leave the network to reach
   the API. If that is not acceptable for production, the backend is swappable
   — `classify/baseline.py` already implements the same interface offline, and
   a fine-tuned AraBERT/CAMeLBERT would slot in the same way.
5. **Add the routing sink** — right now predictions go to JSONL. Where should
   a routed document actually land?
