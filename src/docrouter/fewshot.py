"""Turning confirmed labels into few-shot examples.

Selection is deliberately **fixed**, not retrieved per document. Nearest-
neighbour retrieval would pick better examples for any single document, but
it changes the prompt prefix on every call, which invalidates the server-side
cache and multiplies the per-document cost. With 14 classes a well-chosen
fixed set captures most of the benefit and keeps the prefix byte-stable, so
that is what this does. Ordering is sorted, never shuffled, for the same
reason — see `test_examples_block_is_byte_stable`.

What gets picked matters more than how many. Priority goes to **human
overrides** — documents where a reviewer disagreed with the model. A
confirmation teaches the model something it already knew; a correction
teaches it a boundary it got wrong, and those are exactly the
prosecution-vs-courts and labour-vs-GOSI calls the taxonomy flags.

Because examples embed real document text into every request, `redact()`
runs over them by default: national IDs, phone numbers, IBANs, tax numbers
and emails are replaced with typed placeholders. Dates and amounts survive,
since those carry routing signal.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Document
from .taxonomy import Taxonomy
from .taxonomy import load as load_taxonomy

# Ordered: the specific patterns must win before the generic digit-run rule.
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[بريد]"),
    (re.compile(r"\bSA\d{22}\b", re.I), "[آيبان]"),
    (re.compile(r"\b3\d{14}\b"), "[رقم ضريبي]"),
    (re.compile(r"(?:\+?966|00966|0)5\d{8}\b"), "[جوال]"),
    (re.compile(r"\b[12]\d{9}\b"), "[رقم هوية]"),
    # Anything else long enough to identify a person or a file.
    (re.compile(r"\b\d{9,}\b"), "[رقم]"),
]


def redact(text: str) -> str:
    """Mask identifiers while leaving dates and amounts intact."""
    for pattern, placeholder in _REDACTIONS:
        text = pattern.sub(placeholder, text)
    return text


class Example(BaseModel):
    doc_id: str
    label: str
    text: str
    # Set when a reviewer overrode the model; carries the wrong answer so the
    # rendered prompt can name the boundary explicitly.
    corrected_from: str | None = None
    redacted: bool = True
    truncated: bool = False

    @property
    def is_override(self) -> bool:
        return self.corrected_from is not None


class ExampleSet(BaseModel):
    examples: list[Example] = Field(default_factory=list)
    taxonomy_version: int = 1

    @property
    def doc_ids(self) -> set[str]:
        return {e.doc_id for e in self.examples}

    def per_class(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self.examples:
            counts[e.label] += 1
        return dict(counts)


def select_examples(
    docs: list[Document],
    gold: dict[str, str],
    model_labels: dict[str, str] | None = None,
    taxonomy: Taxonomy | None = None,
    per_class: int = 1,
    max_examples: int = 20,
    max_chars: int = 700,
    do_redact: bool = True,
) -> ExampleSet:
    """Choose a fixed few-shot set from confirmed labels.

    `gold` is doc_id -> human-confirmed label. `model_labels` is what the
    model had said, so overrides can be identified and preferred.

    Guarantees, in order: at most `per_class` per institution first so every
    class is represented, then overrides, then the rest — all within
    `max_examples`. The result is sorted by (label, doc_id) so the rendered
    block is byte-identical across runs.
    """
    tax = taxonomy or load_taxonomy()
    model_labels = model_labels or {}
    known = set(tax.ids)
    by_id = {d.doc_id: d for d in docs}

    def build(doc_id: str, label: str) -> Example | None:
        doc = by_id.get(doc_id)
        if doc is None or not doc.text.strip():
            return None
        text = doc.text.strip()
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars].rstrip() + " …"
        if do_redact:
            text = redact(text)
        was = model_labels.get(doc_id)
        return Example(
            doc_id=doc_id,
            label=label,
            text=text,
            corrected_from=was if was and was != label and was in known else None,
            redacted=do_redact,
            truncated=truncated,
        )

    candidates: dict[str, list[str]] = defaultdict(list)
    for doc_id, label in gold.items():
        if label in known and doc_id in by_id:
            candidates[label].append(doc_id)

    # Within a class, corrections first, then stable order.
    for label, ids in candidates.items():
        ids.sort(key=lambda i: (model_labels.get(i, label) == label, i))

    chosen: list[Example] = []
    taken: set[str] = set()

    # Pass 1 — coverage. Every class that has any gold gets representation.
    for label in tax.ids:
        for doc_id in candidates.get(label, [])[:per_class]:
            if len(chosen) >= max_examples:
                break
            example = build(doc_id, label)
            if example:
                chosen.append(example)
                taken.add(doc_id)

    # Pass 2 — remaining budget goes to corrections, they teach the most.
    leftovers = [
        (doc_id, label)
        for label, ids in candidates.items()
        for doc_id in ids
        if doc_id not in taken
    ]
    leftovers.sort(
        key=lambda pair: (
            model_labels.get(pair[0], pair[1]) == pair[1],  # overrides first
            pair[1],
            pair[0],
        )
    )
    for doc_id, label in leftovers:
        if len(chosen) >= max_examples:
            break
        example = build(doc_id, label)
        if example:
            chosen.append(example)
            taken.add(doc_id)

    chosen.sort(key=lambda e: (e.label, e.doc_id))
    return ExampleSet(examples=chosen, taxonomy_version=tax.version)


def render(example_set: ExampleSet, taxonomy: Taxonomy | None = None) -> str:
    """The examples block, appended to the cached system prompt."""
    if not example_set.examples:
        return ""
    tax = taxonomy or load_taxonomy()

    lines = [
        "",
        "## أمثلة معتمدة من مراجعة بشرية",
        "هذه وثائق حقيقية اعتمد تصنيفها مراجع مختص. استرشد بها عند التقارب،",
        "ولا تنسخ منها التصنيف إلا إذا كان موضوع الوثيقة المعروضة مماثلاً.",
        "(حُجبت أرقام الهوية والجوال ونحوها بعلامات بين قوسين.)",
    ]
    for n, example in enumerate(example_set.examples, 1):
        header = f"### مثال {n} ← {example.label} ({tax.name_ar(example.label)})"
        lines.append("")
        lines.append(header)
        if example.corrected_from:
            lines.append(
                f"تنبيه: صُنّفت خطأً على أنها {example.corrected_from} "
                f"({tax.name_ar(example.corrected_from)})، والصواب ما هو مثبت أعلاه."
            )
        lines.append(example.text)
    return "\n".join(lines)


def check_leakage(example_set: ExampleSet, eval_docs: list[Document]) -> set[str]:
    """Documents that are both in the prompt and in the evaluation set.

    Scoring a model on documents sitting in its own prompt inflates the
    result. Callers are expected to warn or exclude.
    """
    return example_set.doc_ids & {d.doc_id for d in eval_docs}


def save(example_set: ExampleSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(example_set.model_dump_json(indent=2), encoding="utf-8")
    return path


def load(path: str | Path) -> ExampleSet:
    return ExampleSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
