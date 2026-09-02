"""Choosing the auto-route / hold-for-review cut-off.

The threshold is the only knob in this system that trades one kind of harm
for another. Above it a document is routed with no human in the loop; below
it a clerk looks. Raising it buys accuracy on the routed subset and pays for
it in review load. That is a cost decision, not an accuracy decision, so
there are two ways to make it here:

  target mode   "≥95% accuracy on anything auto-routed" — the lowest cut-off
                meeting that bar, so coverage is as high as the bar allows.
  cost mode     a misroute costs N× a clerk review — minimise expected cost.

Two things this module refuses to let you skip.

First, **calibration**. A threshold on a confidence score is meaningless if
the score does not track reality; 0.8 has to mean roughly 80% right, or the
cut-off is built on sand. `calibration()` reports reliability bins and ECE,
and the CLI shows it above the sweep for that reason.

Second, **overfitting the cut-off**. Choosing a threshold on the same
documents you measure it on flatters the result. `split_validate()` picks on
one half and reports what that choice actually delivers on the other.
"""

from __future__ import annotations

import random
from collections import defaultdict

from pydantic import BaseModel

from .models import Document, Prediction
from .taxonomy import Taxonomy
from .taxonomy import load as load_taxonomy

DEFAULT_GRID = [round(0.05 * i, 2) for i in range(0, 21)]


class SweepPoint(BaseModel):
    threshold: float

    auto_routed: int
    coverage: float
    auto_correct: int
    auto_accuracy: float
    misrouted: int

    held: int
    held_correct: int

    expected_cost: float = 0.0
    # End-to-end accuracy assuming a reviewer resolves held documents
    # correctly. An optimistic ceiling, not a measurement.
    resolved_accuracy: float = 0.0

    @property
    def defined(self) -> bool:
        return self.auto_routed > 0


class CalibrationBin(BaseModel):
    low: float
    high: float
    n: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return self.mean_confidence - self.accuracy


class CalibrationReport(BaseModel):
    bins: list[CalibrationBin]
    ece: float
    mean_confidence: float
    accuracy: float

    @property
    def verdict(self) -> str:
        drift = self.mean_confidence - self.accuracy
        if self.ece < 0.05:
            return "well calibrated"
        if drift > 0.05:
            return "overconfident"
        if drift < -0.05:
            return "underconfident"
        return "poorly calibrated (non-monotonic)"


def _paired(docs: list[Document], predictions: list[Prediction]) -> list[tuple[Prediction, bool]]:
    """Pair each prediction with whether it was right. Unlabeled docs dropped."""
    truth = {d.doc_id: d.true_label for d in docs if d.true_label}
    out = []
    for p in predictions:
        gold = truth.get(p.doc_id)
        if gold is not None:
            out.append((p, p.institution_id == gold))
    if not out:
        raise ValueError("no predictions matched a labeled document")
    return out


def sweep(
    docs: list[Document],
    predictions: list[Prediction],
    grid: list[float] | None = None,
    misroute_cost: float = 20.0,
    review_cost: float = 1.0,
) -> list[SweepPoint]:
    """Operating characteristics across candidate thresholds.

    `needs_review` on the incoming predictions is ignored — it records the
    threshold in force when they were produced, which is the very thing
    being re-decided here. Everything is recomputed from `confidence`.
    """
    pairs = _paired(docs, predictions)
    total = len(pairs)
    points: list[SweepPoint] = []

    for threshold in grid or DEFAULT_GRID:
        auto = [(p, ok) for p, ok in pairs if p.confidence >= threshold]
        held = [(p, ok) for p, ok in pairs if p.confidence < threshold]

        auto_correct = sum(1 for _, ok in auto if ok)
        misrouted = len(auto) - auto_correct
        held_correct = sum(1 for _, ok in held if ok)

        points.append(
            SweepPoint(
                threshold=threshold,
                auto_routed=len(auto),
                coverage=len(auto) / total,
                auto_correct=auto_correct,
                auto_accuracy=auto_correct / len(auto) if auto else 0.0,
                misrouted=misrouted,
                held=len(held),
                held_correct=held_correct,
                expected_cost=misrouted * misroute_cost + len(held) * review_cost,
                resolved_accuracy=(auto_correct + len(held)) / total,
            )
        )
    return points


def recommend_for_target(
    points: list[SweepPoint], target_auto_accuracy: float
) -> SweepPoint | None:
    """Lowest threshold meeting the accuracy bar — i.e. the most coverage.

    None when no cut-off reaches the bar, which is a real answer: it means
    the model cannot support that SLA on this data at any threshold.
    """
    viable = [p for p in points if p.defined and p.auto_accuracy >= target_auto_accuracy]
    return max(viable, key=lambda p: (p.coverage, -p.threshold)) if viable else None


def recommend_min_cost(points: list[SweepPoint]) -> SweepPoint:
    """Cheapest cut-off under the supplied cost ratio."""
    return min(points, key=lambda p: (p.expected_cost, p.threshold))


def calibration(
    docs: list[Document], predictions: list[Prediction], n_bins: int = 10
) -> CalibrationReport:
    """Reliability bins and expected calibration error."""
    pairs = _paired(docs, predictions)
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)

    for p, ok in pairs:
        index = min(int(p.confidence * n_bins), n_bins - 1)
        buckets[index].append((p.confidence, ok))

    total = len(pairs)
    bins: list[CalibrationBin] = []
    ece = 0.0

    for index in sorted(buckets):
        entries = buckets[index]
        mean_conf = sum(c for c, _ in entries) / len(entries)
        accuracy = sum(1 for _, ok in entries if ok) / len(entries)
        ece += (len(entries) / total) * abs(mean_conf - accuracy)
        bins.append(
            CalibrationBin(
                low=index / n_bins,
                high=(index + 1) / n_bins,
                n=len(entries),
                mean_confidence=mean_conf,
                accuracy=accuracy,
            )
        )

    return CalibrationReport(
        bins=bins,
        ece=ece,
        mean_confidence=sum(p.confidence for p, _ in pairs) / total,
        accuracy=sum(1 for _, ok in pairs if ok) / total,
    )


class ClassThreshold(BaseModel):
    institution_id: str
    support: int
    threshold: float | None
    coverage: float
    auto_accuracy: float
    # True when there were too few predictions to trust the number.
    thin: bool


def per_class_thresholds(
    docs: list[Document],
    predictions: list[Prediction],
    target_auto_accuracy: float = 0.95,
    grid: list[float] | None = None,
    min_support: int = 10,
    taxonomy: Taxonomy | None = None,
) -> list[ClassThreshold]:
    """A cut-off per predicted institution.

    A single global threshold assumes the model is equally trustworthy
    everywhere, which it is not: it may be confident and right on tax
    filings and confident and wrong on prosecution referrals. Per-class
    cut-offs usually dominate a global one — at the price of needing enough
    labeled documents per class to set them, hence `min_support`.
    """
    tax = taxonomy or load_taxonomy()
    pairs = _paired(docs, predictions)
    by_class: dict[str, list[tuple[Prediction, bool]]] = defaultdict(list)
    for p, ok in pairs:
        by_class[p.institution_id].append((p, ok))

    out: list[ClassThreshold] = []
    for institution_id in tax.ids:
        entries = by_class.get(institution_id, [])
        if not entries:
            out.append(
                ClassThreshold(
                    institution_id=institution_id,
                    support=0,
                    threshold=None,
                    coverage=0.0,
                    auto_accuracy=0.0,
                    thin=True,
                )
            )
            continue

        best: tuple[float, float, float] | None = None
        for threshold in grid or DEFAULT_GRID:
            auto = [(p, ok) for p, ok in entries if p.confidence >= threshold]
            if not auto:
                continue
            accuracy = sum(1 for _, ok in auto if ok) / len(auto)
            coverage = len(auto) / len(entries)
            if accuracy >= target_auto_accuracy and (best is None or coverage > best[1]):
                best = (threshold, coverage, accuracy)

        out.append(
            ClassThreshold(
                institution_id=institution_id,
                support=len(entries),
                threshold=best[0] if best else None,
                coverage=best[1] if best else 0.0,
                auto_accuracy=best[2] if best else 0.0,
                thin=len(entries) < min_support,
            )
        )
    return out


class SplitValidation(BaseModel):
    chosen_threshold: float
    train_auto_accuracy: float
    train_coverage: float
    test_auto_accuracy: float
    test_coverage: float
    test_misrouted: int
    n_train: int
    n_test: int

    @property
    def optimism(self) -> float:
        """How much the choice flattered itself on the data that picked it."""
        return self.train_auto_accuracy - self.test_auto_accuracy


def split_validate(
    docs: list[Document],
    predictions: list[Prediction],
    target_auto_accuracy: float = 0.95,
    misroute_cost: float = 20.0,
    review_cost: float = 1.0,
    mode: str = "target",
    test_ratio: float = 0.4,
    seed: int = 7,
    grid: list[float] | None = None,
) -> SplitValidation | None:
    """Pick a threshold on one half; report what it delivers on the other.

    Returns None when the chosen mode cannot find a threshold on the
    training half.
    """
    pairs = _paired(docs, predictions)
    ids = [p.doc_id for p, _ in pairs]
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * (1 - test_ratio))
    train_ids, test_ids = set(ids[:cut]), set(ids[cut:])

    truth_docs = {d.doc_id: d for d in docs if d.true_label}
    by_id = {p.doc_id: p for p in predictions}

    def subset(keep: set[str]):
        d = [truth_docs[i] for i in keep if i in truth_docs]
        p = [by_id[i] for i in keep if i in by_id]
        return d, p

    train_docs, train_preds = subset(train_ids)
    test_docs, test_preds = subset(test_ids)
    if not train_docs or not test_docs:
        return None

    train_points = sweep(
        train_docs, train_preds, grid, misroute_cost=misroute_cost, review_cost=review_cost
    )
    chosen = (
        recommend_for_target(train_points, target_auto_accuracy)
        if mode == "target"
        else recommend_min_cost(train_points)
    )
    if chosen is None:
        return None

    test_point = sweep(
        test_docs,
        test_preds,
        [chosen.threshold],
        misroute_cost=misroute_cost,
        review_cost=review_cost,
    )[0]

    return SplitValidation(
        chosen_threshold=chosen.threshold,
        train_auto_accuracy=chosen.auto_accuracy,
        train_coverage=chosen.coverage,
        test_auto_accuracy=test_point.auto_accuracy,
        test_coverage=test_point.coverage,
        test_misrouted=test_point.misrouted,
        n_train=len(train_docs),
        n_test=len(test_docs),
    )
