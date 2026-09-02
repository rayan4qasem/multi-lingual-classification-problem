"""Scoring a set of predictions against known labels.

Accuracy alone is the wrong headline for a routing system: sending a document
to the wrong ministry costs far more than holding it for a clerk. So the
report separates three outcomes — routed correctly, routed incorrectly, and
held for review — and reports accuracy over the auto-routed subset as well as
overall. That pair is what a threshold is actually tuned against.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from .models import Document, Prediction
from .taxonomy import Taxonomy
from .taxonomy import load as load_taxonomy


class ClassMetrics(BaseModel):
    institution_id: str
    support: int
    precision: float
    recall: float
    f1: float


class Report(BaseModel):
    backend: str
    total: int
    accuracy: float
    # Documents the system routed without asking a human.
    auto_routed: int
    auto_accuracy: float
    held_for_review: int
    # Of the ones held, how many the model would have got right anyway.
    review_would_have_been_correct: int
    macro_f1: float
    per_class: list[ClassMetrics]
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    confusion_pair_errors: dict[str, int] = Field(default_factory=dict)
    mean_confidence_correct: float = 0.0
    mean_confidence_wrong: float = 0.0

    def summary_lines(self, taxonomy: Taxonomy) -> list[str]:
        lines = [
            f"backend            : {self.backend}",
            f"documents          : {self.total}",
            f"accuracy (all)     : {self.accuracy:.1%}",
            f"auto-routed        : {self.auto_routed} ({self.auto_routed / max(self.total, 1):.0%})",
            f"accuracy (auto)    : {self.auto_accuracy:.1%}",
            f"held for review    : {self.held_for_review}"
            f"  (of which {self.review_would_have_been_correct} were already correct)",
            f"macro F1           : {self.macro_f1:.3f}",
            f"confidence  correct: {self.mean_confidence_correct:.2f}"
            f"   wrong: {self.mean_confidence_wrong:.2f}",
        ]
        if self.confusion_pair_errors:
            lines.append("known confusion pairs:")
            for pair, count in sorted(self.confusion_pair_errors.items(), key=lambda kv: -kv[1]):
                if count:
                    lines.append(f"  {pair}: {count}")
        return lines


def evaluate(
    docs: list[Document],
    predictions: list[Prediction],
    taxonomy: Taxonomy | None = None,
) -> Report:
    tax = taxonomy or load_taxonomy()
    truth = {d.doc_id: d.true_label for d in docs if d.true_label}
    scored = [p for p in predictions if p.doc_id in truth]
    if not scored:
        raise ValueError("no predictions matched a labeled document")

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    support: Counter[str] = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    correct = 0
    auto_total = auto_correct = 0
    held = held_correct = 0
    conf_correct: list[float] = []
    conf_wrong: list[float] = []

    for pred in scored:
        gold = truth[pred.doc_id]
        got = pred.institution_id
        support[gold] += 1
        confusion[gold][got] += 1
        hit = gold == got

        if hit:
            correct += 1
            tp[gold] += 1
            conf_correct.append(pred.confidence)
        else:
            fp[got] += 1
            fn[gold] += 1
            conf_wrong.append(pred.confidence)

        if pred.needs_review:
            held += 1
            held_correct += hit
        else:
            auto_total += 1
            auto_correct += hit

    per_class = []
    for institution_id in tax.ids:
        t, f_p, f_n = tp[institution_id], fp[institution_id], fn[institution_id]
        precision = t / (t + f_p) if (t + f_p) else 0.0
        recall = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append(
            ClassMetrics(
                institution_id=institution_id,
                support=support[institution_id],
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    # Macro F1 over classes that actually appear, so an unused class in the
    # taxonomy does not silently drag the score to zero.
    present = [m for m in per_class if m.support]
    macro_f1 = sum(m.f1 for m in present) / len(present) if present else 0.0

    pair_errors = {}
    for a, b in tax.confusion_pairs:
        pair_errors[f"{a} <-> {b}"] = confusion[a][b] + confusion[b][a]

    return Report(
        backend=scored[0].backend,
        total=len(scored),
        accuracy=correct / len(scored),
        auto_routed=auto_total,
        auto_accuracy=auto_correct / auto_total if auto_total else 0.0,
        held_for_review=held,
        review_would_have_been_correct=held_correct,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion={k: dict(v) for k, v in confusion.items()},
        confusion_pair_errors=pair_errors,
        mean_confidence_correct=(sum(conf_correct) / len(conf_correct) if conf_correct else 0.0),
        mean_confidence_wrong=sum(conf_wrong) / len(conf_wrong) if conf_wrong else 0.0,
    )
