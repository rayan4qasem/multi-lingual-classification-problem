"""Threshold sweep tests. All offline."""

from __future__ import annotations

import pytest

from docrouter.models import Document, Prediction
from docrouter.threshold import (
    calibration,
    per_class_thresholds,
    recommend_for_target,
    recommend_min_cost,
    split_validate,
    sweep,
)


def _case(n_correct_high, n_wrong_high, n_correct_low, n_wrong_low):
    """Build a set where confidence genuinely separates right from wrong."""
    docs, preds = [], []

    def add(prefix, count, confidence, correct):
        for i in range(count):
            doc_id = f"{prefix}{i}"
            gold = "moj_courts"
            docs.append(Document(doc_id=doc_id, text="نص", true_label=gold))
            preds.append(
                Prediction(
                    doc_id=doc_id,
                    institution_id=gold if correct else "gosi",
                    confidence=confidence,
                    backend="t",
                )
            )

    add("ch", n_correct_high, 0.9, True)
    add("wh", n_wrong_high, 0.9, False)
    add("cl", n_correct_low, 0.3, True)
    add("wl", n_wrong_low, 0.3, False)
    return docs, preds


def test_sweep_covers_grid_and_partitions_every_document():
    docs, preds = _case(10, 1, 5, 4)
    points = sweep(docs, preds)
    assert len(points) == 21
    for p in points:
        assert p.auto_routed + p.held == len(docs)
        assert p.auto_correct + p.misrouted == p.auto_routed


def test_coverage_falls_monotonically_as_threshold_rises():
    docs, preds = _case(10, 2, 6, 5)
    coverages = [p.coverage for p in sweep(docs, preds)]
    assert coverages == sorted(coverages, reverse=True)


def test_raising_the_threshold_buys_accuracy_and_costs_coverage():
    docs, preds = _case(10, 1, 2, 8)
    points = {p.threshold: p for p in sweep(docs, preds)}
    low, high = points[0.0], points[0.5]
    assert high.auto_accuracy > low.auto_accuracy
    assert high.coverage < low.coverage


def test_incoming_needs_review_flag_is_ignored():
    # It records the threshold in force when the predictions were made,
    # which is exactly what the sweep is re-deciding.
    docs, preds = _case(6, 0, 4, 0)
    for p in preds:
        p.needs_review = True
    point = sweep(docs, preds, grid=[0.0])[0]
    assert point.auto_routed == len(docs)
    assert point.held == 0


def test_target_mode_picks_the_widest_coverage_meeting_the_bar():
    docs, preds = _case(10, 0, 2, 8)
    points = sweep(docs, preds)
    chosen = recommend_for_target(points, 0.95)
    assert chosen is not None
    assert chosen.auto_accuracy >= 0.95
    # Anything below 0.9 lets the wrong low-confidence documents through.
    assert 0.3 < chosen.threshold <= 0.9
    for p in points:
        if p.defined and p.auto_accuracy >= 0.95:
            assert p.coverage <= chosen.coverage


def test_target_mode_returns_none_when_unreachable():
    docs, preds = _case(5, 5, 5, 5)  # 50% right at every confidence level
    assert recommend_for_target(sweep(docs, preds), 0.99) is None


def test_target_mode_ignores_zero_coverage_thresholds():
    docs, preds = _case(8, 2, 0, 0)
    chosen = recommend_for_target(sweep(docs, preds), 0.75)
    assert chosen is not None and chosen.auto_routed > 0


def test_cost_mode_follows_the_cost_ratio():
    docs, preds = _case(10, 1, 2, 8)
    # Misroutes catastrophic -> hold more.
    strict = recommend_min_cost(sweep(docs, preds, misroute_cost=500, review_cost=1))
    # Reviews expensive -> route more.
    loose = recommend_min_cost(sweep(docs, preds, misroute_cost=2, review_cost=1))
    assert strict.threshold >= loose.threshold
    assert strict.coverage <= loose.coverage


def test_expected_cost_arithmetic():
    docs, preds = _case(4, 1, 3, 2)
    point = sweep(docs, preds, grid=[0.5], misroute_cost=10, review_cost=2)[0]
    assert point.auto_routed == 5 and point.misrouted == 1
    assert point.held == 5
    assert point.expected_cost == pytest.approx(1 * 10 + 5 * 2)


def test_resolved_accuracy_assumes_reviewers_fix_held_documents():
    docs, preds = _case(4, 1, 0, 5)
    point = sweep(docs, preds, grid=[0.5])[0]
    # 4 auto-correct + 5 held resolved = 9 of 10.
    assert point.resolved_accuracy == pytest.approx(0.9)


# ---------- calibration ----------

def test_calibration_detects_overconfidence():
    docs, preds = _case(2, 8, 0, 0)  # says 0.9, right 20% of the time
    report = calibration(docs, preds)
    assert report.verdict == "overconfident"
    assert report.ece > 0.5


def test_calibration_detects_a_well_calibrated_model():
    docs, preds = _case(9, 1, 3, 7)  # 0.9 -> 90% right, 0.3 -> 30% right
    report = calibration(docs, preds)
    assert report.verdict == "well calibrated"
    assert report.ece < 0.05


def test_calibration_bins_partition_the_data():
    docs, preds = _case(5, 5, 5, 5)
    report = calibration(docs, preds)
    assert sum(b.n for b in report.bins) == len(docs)


# ---------- per class ----------

def test_per_class_flags_thin_support():
    docs, preds = _case(3, 0, 0, 0)
    rows = {r.institution_id: r for r in per_class_thresholds(docs, preds, min_support=10)}
    assert rows["moj_courts"].thin is True
    # A class with no predictions at all gets no threshold.
    assert rows["zatca"].threshold is None


def test_per_class_finds_a_cutoff_where_one_exists():
    docs, preds = _case(12, 0, 0, 6)
    rows = {r.institution_id: r for r in per_class_thresholds(docs, preds, target_auto_accuracy=0.9)}
    assert rows["moj_courts"].threshold is not None
    assert rows["moj_courts"].auto_accuracy >= 0.9


# ---------- split validation ----------

def test_split_validation_reports_both_halves():
    # Comfortably separable, so the split cannot fail for want of a
    # threshold — this test is about the two halves being reported, and
    # test_split_validation_returns_none_when_target_unreachable covers
    # the other branch.
    docs, preds = _case(40, 1, 0, 20)
    result = split_validate(docs, preds, target_auto_accuracy=0.85)
    assert result is not None
    assert result.n_train > 0 and result.n_test > 0
    assert 0.0 <= result.test_auto_accuracy <= 1.0
    assert result.chosen_threshold in [round(0.05 * i, 2) for i in range(21)]


def test_split_validation_returns_none_when_target_unreachable():
    docs, preds = _case(10, 10, 10, 10)
    assert split_validate(docs, preds, target_auto_accuracy=0.99) is None


# ---------- guards ----------

def test_unlabeled_documents_are_rejected():
    docs = [Document(doc_id="a", text="نص")]
    preds = [Prediction(doc_id="a", institution_id="gosi", confidence=0.9, backend="t")]
    with pytest.raises(ValueError):
        sweep(docs, preds)
