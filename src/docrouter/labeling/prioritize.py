"""Choosing what a human should look at next.

Reviewing an archive at random wastes most of the effort: the model already
gets the easy documents right, and confirming them teaches nothing. So the
priority lane ranks candidates by how much a label would be worth.

Four signals, each a real failure mode:

  uncertainty   the model says it doesn't know
  margin        top-1 and top-2 are close — a coin flip dressed as a decision
  disagreement  the LLM and the offline baseline picked different institutions
  pair          the top two are a confusion pair declared in the taxonomy

Plus a diversity term. A government archive is full of near-identical forms;
without it the queue fills with fifty copies of the same letter. Selection is
greedy with a redundancy penalty against what's already been picked.

The random lane exists for a separate reason. Any queue ranked by difficulty
is a biased sample, so agreement measured on it *understates* real accuracy.
A parallel uniform-random sample is the only thing here that yields an honest
estimate, which is why `build_queue` always reserves part of the batch for it.
"""

from __future__ import annotations

import random

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import normalize
from ..models import Document, Prediction
from ..taxonomy import Taxonomy
from ..taxonomy import load as load_taxonomy
from .store import Lane


class Weights(BaseModel):
    uncertainty: float = 1.0
    margin: float = 0.8
    disagreement: float = 1.2
    confusion_pair: float = 0.6
    novelty: float = 0.5
    # Subtracted per unit of similarity to an already-selected item.
    redundancy_penalty: float = 1.5


class QueueItem(BaseModel):
    doc_id: str
    text: str
    # Constrained, not free text: the lane decides how the document is served
    # in review and how its agreement is interpreted afterwards.
    lane: Lane = "priority"
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)

    model_label: str | None = None
    model_confidence: float | None = None
    model_backend: str | None = None
    model_rationale_ar: str = ""
    # Ranked alternates, shown as one-key picks in the review UI.
    alternatives: list[str] = Field(default_factory=list)
    baseline_label: str | None = None

    path: str | None = None
    source: str = "plain"


def _pair_set(taxonomy: Taxonomy) -> set[frozenset[str]]:
    return {frozenset(p) for p in taxonomy.confusion_pairs}


def _score(
    prediction: Prediction | None,
    baseline: Prediction | None,
    pairs: set[frozenset[str]],
    weights: Weights,
) -> tuple[float, list[str]]:
    if prediction is None:
        return 1.0, ["بلا تنبؤ"]

    score = 0.0
    reasons: list[str] = []

    uncertainty = 1.0 - prediction.confidence
    score += weights.uncertainty * uncertainty
    if prediction.confidence < 0.6:
        reasons.append(f"ثقة منخفضة ({prediction.confidence:.2f})")

    if prediction.alternatives:
        margin = prediction.confidence - prediction.alternatives[0].confidence
        score += weights.margin * max(0.0, 1.0 - margin)
        if margin < 0.15:
            reasons.append(f"فارق ضئيل عن البديل ({margin:.2f})")

        top_two = frozenset({prediction.institution_id, prediction.alternatives[0].institution_id})
        if top_two in pairs:
            score += weights.confusion_pair
            reasons.append("زوج خلط معروف")

    if baseline is not None and baseline.institution_id != prediction.institution_id:
        score += weights.disagreement
        reasons.append("اختلاف بين النموذجين")

    return score, reasons


def _similarity_matrix(texts: list[str]) -> np.ndarray | None:
    """Cosine similarity over normalized text. None when it can't be built."""
    if len(texts) < 2:
        return None
    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50_000
        )
        matrix = vectorizer.fit_transform(normalize.aggressive(t) for t in texts)
    except ValueError:
        # Empty vocabulary — every document was blank or unreadable.
        return None
    return (matrix @ matrix.T).toarray()


def build_queue(
    docs: list[Document],
    predictions: list[Prediction] | None = None,
    baseline_predictions: list[Prediction] | None = None,
    already_labeled: set[str] | None = None,
    taxonomy: Taxonomy | None = None,
    size: int = 50,
    random_ratio: float = 0.2,
    per_class_cap: int | None = None,
    weights: Weights | None = None,
    seed: int = 7,
) -> list[QueueItem]:
    """Build a review batch: a priority lane plus an unbiased random lane.

    `random_ratio` of the batch is drawn uniformly from the unlabeled pool.
    Do not set it to zero — without it there is no honest accuracy estimate,
    only agreement on the documents the model already found hard.
    """
    tax = taxonomy or load_taxonomy()
    weights = weights or Weights()
    already_labeled = already_labeled or set()
    rng = random.Random(seed)

    by_id = {p.doc_id: p for p in (predictions or [])}
    base_by_id = {p.doc_id: p for p in (baseline_predictions or [])}
    pairs = _pair_set(tax)

    pool = [d for d in docs if d.doc_id not in already_labeled]
    if not pool:
        return []

    def make_item(doc: Document, lane: Lane, score: float, reasons: list[str]) -> QueueItem:
        pred = by_id.get(doc.doc_id)
        base = base_by_id.get(doc.doc_id)
        return QueueItem(
            doc_id=doc.doc_id,
            text=doc.text,
            lane=lane,
            score=round(score, 4),
            reasons=reasons,
            model_label=pred.institution_id if pred else None,
            model_confidence=pred.confidence if pred else None,
            model_backend=pred.backend if pred else None,
            model_rationale_ar=pred.rationale_ar if pred else "",
            alternatives=[a.institution_id for a in pred.alternatives] if pred else [],
            baseline_label=base.institution_id if base else None,
            path=doc.path,
            source=doc.source,
        )

    # --- random lane first, so it is never crowded out by the ranking ---
    n_random = min(len(pool), round(size * random_ratio))
    random_docs = rng.sample(pool, n_random) if n_random else []
    random_ids = {d.doc_id for d in random_docs}
    items = [make_item(d, "random", 0.0, ["عينة عشوائية"]) for d in random_docs]

    # --- priority lane ---
    remaining = [d for d in pool if d.doc_id not in random_ids]
    n_priority = size - len(items)
    if remaining and n_priority > 0:
        scored = []
        for doc in remaining:
            score, reasons = _score(
                by_id.get(doc.doc_id), base_by_id.get(doc.doc_id), pairs, weights
            )
            scored.append((score, reasons, doc))

        similarity = _similarity_matrix([d.text for d in remaining])
        index_of = {d.doc_id: i for i, d in enumerate(remaining)}

        # Novelty against nothing yet, so it only shapes intra-batch spread.
        chosen: list[int] = []
        per_class: dict[str, int] = {}
        candidates = sorted(range(len(scored)), key=lambda i: -scored[i][0])

        while len(items) < size and candidates:
            # Greedy pick: highest score after penalising similarity to what
            # has already been chosen, so near-duplicates do not stack up.
            best_i = candidates[0]
            best_adjusted = float("-inf")
            for i in candidates:
                score, _, doc = scored[i]
                adjusted = score
                if similarity is not None and chosen:
                    redundancy = max(
                        similarity[index_of[doc.doc_id]][index_of[remaining[j].doc_id]]
                        for j in chosen
                    )
                    adjusted -= weights.redundancy_penalty * float(redundancy)
                if adjusted > best_adjusted:
                    best_i, best_adjusted = i, adjusted

            score, reasons, doc = scored[best_i]
            candidates.remove(best_i)

            prediction = by_id.get(doc.doc_id)
            label = prediction.institution_id if prediction is not None else "?"
            if per_class_cap is not None and per_class.get(label, 0) >= per_class_cap:
                continue
            per_class[label] = per_class.get(label, 0) + 1

            chosen.append(best_i)
            if not reasons and similarity is not None and len(chosen) > 1:
                # Nothing else flagged it; it earned its slot on diversity.
                reasons = ["مختار لتنويع الدفعة"]
            items.append(make_item(doc, "priority", score, reasons))

    # Interleave so a reviewer meets both lanes throughout the session rather
    # than hitting a wall of hard documents first.
    rng.shuffle(items)
    return items
