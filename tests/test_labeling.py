"""Labeling loop tests. All offline."""

from __future__ import annotations

import json

import pytest

from docrouter import mockdata, taxonomy
from docrouter.labeling.prioritize import Weights, build_queue
from docrouter.labeling.review import ReviewSession, load_queue, save_queue
from docrouter.labeling.store import LabelRecord, LabelStore
from docrouter.models import Alternative, Document, Prediction


def _docs(n=30):
    return mockdata.generate_curated(seed=3)[:n]


def _pred(doc_id, label, confidence, alt=None, backend="llm:test"):
    return Prediction(
        doc_id=doc_id,
        institution_id=label,
        confidence=confidence,
        alternatives=[Alternative(institution_id=a, confidence=c) for a, c in (alt or [])],
        backend=backend,
    )


# ---------- store ----------

def test_store_roundtrip_and_last_write_wins(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(LabelRecord(doc_id="d1", label="moh_health", model_label="moj_courts"))
    store.append(LabelRecord(doc_id="d1", label="moj_courts", model_label="moj_courts"))

    current = store.current()
    assert len(current) == 1
    assert current["d1"].label == "moj_courts"
    # Both are still on disk: the file is an audit trail, not a mutable table.
    assert len(store.all_records()) == 2


def test_store_never_persists_document_text(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path)
    store.append(LabelRecord(doc_id="d1", label="moh_health", path="C:/docs/d1.pdf"))
    raw = json.loads(path.read_text(encoding="utf-8").strip())
    assert "text" not in raw
    assert raw["path"] == "C:/docs/d1.pdf"


def test_agreement_and_lane_split(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(LabelRecord(doc_id="a", label="gosi", model_label="gosi", lane="random"))
    store.append(LabelRecord(doc_id="b", label="gosi", model_label="hrsd_labor", lane="random"))
    store.append(LabelRecord(doc_id="c", label="zatca", model_label="zatca", lane="priority"))

    stats = store.stats()
    assert stats.labeled == 3
    assert stats.random.n == 2 and stats.random.agreed == 1
    assert stats.random.agreement == 0.5
    assert stats.priority.n == 1 and stats.priority.agreement == 1.0
    assert stats.per_class["gosi"] == 2


def test_skips_are_not_reserved_but_do_count_as_decided(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(LabelRecord(doc_id="a", status="skipped", label=""))
    assert store.gold() == {}
    # Still "seen", so the queue won't keep re-serving it.
    assert store.labeled_ids() == {"a"}


def test_wilson_interval_widens_on_small_samples(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    for i in range(4):
        store.append(
            LabelRecord(doc_id=f"d{i}", label="gosi", model_label="gosi", lane="random")
        )
    narrow_lo, narrow_hi = store.stats().random.wilson_interval()
    for i in range(4, 60):
        store.append(
            LabelRecord(doc_id=f"d{i}", label="gosi", model_label="gosi", lane="random")
        )
    wide_lo, wide_hi = store.stats().random.wilson_interval()
    # Same 100% agreement, but far more evidence for it.
    assert wide_lo > narrow_lo


# ---------- prioritize ----------

def test_queue_reserves_a_random_lane():
    docs = _docs(40)
    items = build_queue(docs, size=20, random_ratio=0.25, seed=1)
    lanes = [i.lane for i in items]
    assert len(items) == 20
    assert lanes.count("random") == 5


def test_low_confidence_outranks_high_confidence():
    docs = _docs(12)
    preds = [
        _pred(d.doc_id, "moj_courts", 0.95 if n % 2 == 0 else 0.30)
        for n, d in enumerate(docs)
    ]
    items = build_queue(docs, predictions=preds, size=6, random_ratio=0.0, seed=1)
    priority = [i for i in items if i.lane == "priority"]
    assert priority, "expected a priority lane"
    # The uncertain half should dominate the batch.
    uncertain = sum(1 for i in priority if (i.model_confidence or 1) < 0.5)
    assert uncertain > len(priority) / 2


def test_disagreement_is_scored_and_explained():
    docs = _docs(6)
    llm = [_pred(d.doc_id, "moj_courts", 0.8) for d in docs]
    base = [_pred(d.doc_id, "gosi", 0.8, backend="baseline") for d in docs[:3]]
    items = build_queue(
        docs, predictions=llm, baseline_predictions=base,
        size=6, random_ratio=0.0, seed=1,
    )
    flagged = [i for i in items if "اختلاف بين النموذجين" in i.reasons]
    assert len(flagged) == 3
    assert all(i.baseline_label == "gosi" for i in flagged)


def test_confusion_pair_is_flagged():
    tax = taxonomy.load()
    a, b = tax.confusion_pairs[0]
    docs = _docs(4)
    preds = [_pred(d.doc_id, a, 0.55, alt=[(b, 0.45)]) for d in docs]
    items = build_queue(docs, predictions=preds, size=4, random_ratio=0.0, seed=1)
    assert any("زوج خلط معروف" in i.reasons for i in items)


def test_already_labeled_documents_are_excluded():
    docs = _docs(20)
    done = {d.doc_id for d in docs[:15]}
    items = build_queue(docs, already_labeled=done, size=20, seed=1)
    assert len(items) == 5
    assert not ({i.doc_id for i in items} & done)


def test_per_class_cap_limits_one_class_dominating():
    docs = _docs(30)
    preds = [_pred(d.doc_id, "moj_courts", 0.2) for d in docs]
    items = build_queue(
        docs, predictions=preds, size=20, random_ratio=0.0, per_class_cap=4, seed=1
    )
    assert len(items) <= 4


def test_empty_pool_returns_empty_queue():
    docs = _docs(5)
    items = build_queue(docs, already_labeled={d.doc_id for d in docs}, size=10)
    assert items == []


def test_queue_survives_a_save_load_roundtrip(tmp_path):
    items = build_queue(_docs(10), size=5, seed=1)
    path = save_queue(items, tmp_path / "q.jsonl")
    assert [i.doc_id for i in load_queue(path)] == [i.doc_id for i in items]


# ---------- review session ----------

def _session(tmp_path, lane="random", blind=True):
    doc = _docs(1)[0]
    pred = _pred(doc.doc_id, "gosi", 0.9, alt=[("hrsd_labor", 0.4)])
    items = build_queue([doc], predictions=[pred], size=1, random_ratio=1.0 if lane == "random" else 0.0)
    items[0].lane = lane
    store = LabelStore(tmp_path / "labels.jsonl")
    return ReviewSession(items, store, taxonomy.load(), "tester", blind_random=blind), items[0], store


def test_random_lane_payload_is_stripped_not_merely_hidden(tmp_path):
    session, item, _ = _session(tmp_path, lane="random", blind=True)
    payload = session.bootstrap()
    served = payload["items"][0]
    assert served["blind"] is True
    # The prediction must not be present in what reaches the browser at all.
    assert served["model_label"] is None
    assert served["alternatives"] == []
    # Scoped to the item itself: the institution list and name map must still
    # be sent, since that is what the reviewer picks from.
    served_without_text = {k: v for k, v in served.items() if k != "text"}
    assert "gosi" not in json.dumps(served_without_text, ensure_ascii=False)
    # ...but the server still knows, so agreement can be scored.
    assert session.items[item.doc_id].model_label == "gosi"


def test_priority_lane_shows_the_suggestion(tmp_path):
    session, _, _ = _session(tmp_path, lane="priority", blind=True)
    served = session.bootstrap()["items"][0]
    assert served["blind"] is False
    assert served["model_label"] == "gosi"


def test_recording_a_decision_scores_agreement(tmp_path):
    session, item, store = _session(tmp_path, lane="random")
    result = session.record(
        {"doc_id": item.doc_id, "status": "labeled", "label": "gosi", "seconds_spent": 4.2}
    )
    assert result["model_label"] == "gosi"
    assert result["stats"]["random"]["n"] == 1
    saved = store.current()[item.doc_id]
    assert saved.agreed is True
    assert saved.reviewer == "tester"
    assert saved.lane == "random"


def test_recording_an_override_is_marked_as_disagreement(tmp_path):
    session, item, store = _session(tmp_path, lane="random")
    session.record({"doc_id": item.doc_id, "status": "labeled", "label": "moj_courts"})
    assert store.current()[item.doc_id].agreed is False


def test_unknown_institution_is_rejected(tmp_path):
    session, item, _ = _session(tmp_path)
    with pytest.raises(ValueError):
        session.record({"doc_id": item.doc_id, "status": "labeled", "label": "not_a_real_id"})


def test_unknown_document_is_rejected(tmp_path):
    session, _, _ = _session(tmp_path)
    with pytest.raises(KeyError):
        session.record({"doc_id": "nope", "status": "labeled", "label": "gosi"})


def test_skip_needs_no_label(tmp_path):
    session, item, store = _session(tmp_path)
    session.record({"doc_id": item.doc_id, "status": "skipped", "label": ""})
    assert store.current()[item.doc_id].status == "skipped"
    assert store.gold() == {}


# ---------- the loop closes ----------

def test_gold_export_feeds_the_evaluator(tmp_path):
    from docrouter.evaluate import evaluate

    docs = _docs(6)
    store = LabelStore(tmp_path / "labels.jsonl")
    for doc in docs:
        store.append(
            LabelRecord(
                doc_id=doc.doc_id, label=doc.true_label,
                model_label=doc.true_label, lane="random",
            )
        )

    gold = store.gold()
    assert len(gold) == len(docs)

    labeled = [
        Document(doc_id=d.doc_id, text=d.text, true_label=gold[d.doc_id]) for d in docs
    ]
    preds = [_pred(d.doc_id, gold[d.doc_id], 0.9) for d in labeled]
    report = evaluate(labeled, preds)
    assert report.accuracy == 1.0
    assert report.total == len(docs)
