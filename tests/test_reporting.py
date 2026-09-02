"""Terminal rendering.

Rendering lives apart from the CLI precisely so it can be tested without
invoking a command, so these assert on structure — row counts, column
counts, the presence of the flags that carry meaning — rather than on
formatted output, which would break on every cosmetic change.
"""

from __future__ import annotations

from rich.table import Table

from docrouter import fewshot, mockdata, reporting, taxonomy
from docrouter.evaluate import evaluate
from docrouter.labeling.store import LabelRecord, LabelStore
from docrouter.models import Alternative, Document, Prediction
from docrouter.threshold import (
    calibration,
    per_class_thresholds,
    recommend_for_target,
    recommend_min_cost,
    sweep,
)

TAX = taxonomy.load()


def _docs(n=12):
    return mockdata.generate_curated(seed=6)[:n]


def _preds(docs, confidence=0.8, wrong_every=4):
    out = []
    for i, doc in enumerate(docs):
        label = doc.true_label if i % wrong_every else "gosi"
        out.append(
            Prediction(
                doc_id=doc.doc_id,
                institution_id=label,
                confidence=confidence if i % 2 else 0.3,
                alternatives=[Alternative(institution_id="moj_courts", confidence=0.2)],
                needs_review=bool(i % 2 == 0),
                backend="t",
            )
        )
    return out


def test_institutions_table_lists_every_institution():
    table = reporting.institutions_table(TAX)
    assert isinstance(table, Table)
    assert table.row_count == len(TAX.institutions)
    assert len(table.columns) == 3


def test_routing_table_marks_held_documents():
    docs = _docs(6)
    table = reporting.routing_table(_preds(docs), TAX)
    assert table.row_count == 6
    flags = list(table.columns[-1].cells)
    assert any("مراجعة" in cell for cell in flags)


def test_routing_table_truncates_to_the_limit():
    docs = _docs(12)
    assert reporting.routing_table(_preds(docs), TAX, limit=4).row_count == 4


def test_routing_table_handles_an_empty_run():
    assert reporting.routing_table([], TAX).row_count == 0


def test_per_class_table_skips_classes_with_no_support():
    docs = _docs(8)
    report = evaluate(docs, _preds(docs))
    table = reporting.per_class_table(report)
    supported = sum(1 for m in report.per_class if m.support)
    assert table.row_count == supported
    assert table.row_count < len(TAX.ids)


def test_reliability_table_has_one_row_per_bin():
    docs = _docs(12)
    cal = calibration(docs, _preds(docs))
    table = reporting.reliability_table(cal)
    assert table.row_count == len(cal.bins)
    assert len(table.columns) == 5


def test_sweep_table_marks_the_recommended_rows():
    docs = _docs(12)
    preds = _preds(docs)
    points = sweep(docs, preds)
    cheapest = recommend_min_cost(points)
    on_target = recommend_for_target(points, 0.6)

    table = reporting.sweep_table(points, on_target, cheapest)
    assert table.row_count == len(points)
    picks = " ".join(table.columns[-1].cells)
    assert "cheapest" in picks
    if on_target:
        assert "target" in picks


def test_sweep_table_survives_no_reachable_target():
    docs = _docs(12)
    points = sweep(docs, _preds(docs))
    table = reporting.sweep_table(points, None, recommend_min_cost(points))
    assert "target" not in " ".join(table.columns[-1].cells)


def test_class_thresholds_table_flags_thin_support():
    docs = _docs(12)
    rows = per_class_thresholds(docs, _preds(docs), taxonomy=TAX)
    table = reporting.class_thresholds_table(rows, 0.95)
    assert table.row_count == len(TAX.ids)
    assert any("thin" in cell for cell in table.columns[-1].cells)


def test_coverage_table_shows_what_is_still_needed(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(LabelRecord(doc_id="a", label="gosi", model_label="gosi"))
    table = reporting.coverage_table(store.stats(), TAX, target_per_class=5)

    assert table.row_count == len(TAX.ids)
    have = list(table.columns[2].cells)
    need = list(table.columns[3].cells)
    assert "1" in have
    assert any("4" in cell for cell in need)


def test_coverage_table_marks_met_targets_green(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    for i in range(3):
        store.append(LabelRecord(doc_id=f"d{i}", label="gosi", model_label="gosi"))
    table = reporting.coverage_table(store.stats(), TAX, target_per_class=2)
    assert any("green" in cell for cell in table.columns[3].cells)


def test_examples_table_marks_corrections_and_truncation():
    docs = [
        Document(doc_id="short", text="نص قصير", true_label="gosi"),
        Document(doc_id="long", text="نص طويل جداً. " * 200, true_label="moj_courts"),
    ]
    gold = {"short": "gosi", "long": "moj_courts"}
    example_set = fewshot.select_examples(
        docs, gold, model_labels={"short": "hrsd_labor"}, max_chars=100
    )
    table = reporting.examples_table(example_set, TAX)

    assert table.row_count == len(example_set.examples)
    flags = " ".join(table.columns[-1].cells)
    assert "صُحح من" in flags
    assert "مقتطع" in flags


def test_examples_table_handles_an_empty_set():
    assert reporting.examples_table(fewshot.ExampleSet(), TAX).row_count == 0
