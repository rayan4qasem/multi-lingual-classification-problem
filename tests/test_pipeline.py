"""Offline tests. Nothing here calls the API."""

from __future__ import annotations

import pytest

from docrouter import mockdata, normalize, taxonomy
from docrouter.classify import BaselineClassifier
from docrouter.classify.llm import build_schema
from docrouter.evaluate import evaluate
from docrouter.models import Document


def test_taxonomy_loads_and_validates():
    tax = taxonomy.load()
    assert len(tax.institutions) >= 10
    assert len(set(tax.ids)) == len(tax.ids)
    tax.validate_refs()


def test_every_institution_has_templates():
    tax = taxonomy.load()
    missing = [i for i in tax.ids if i not in mockdata.SCENARIOS]
    assert not missing, f"institutions without mock scenarios: {missing}"


def test_prompt_block_is_byte_stable():
    # The catalogue is the cached prefix of every request; if it varies between
    # calls the cache never hits and costs quietly triple.
    tax = taxonomy.load()
    assert tax.render_for_prompt() == tax.render_for_prompt()


def test_schema_enumerates_only_known_ids():
    tax = taxonomy.load()
    schema = build_schema(tax)
    assert schema["properties"]["institution_id"]["enum"] == tax.ids
    assert schema["additionalProperties"] is False


def test_light_normalization_preserves_meaning():
    raw = "الْحَمْدُ للهِ ـــ رَبِّ العالَمين"
    out = normalize.light(raw)
    assert "ـ" not in out
    assert "َ" not in out
    assert "الحمد" in out


def test_aggressive_normalization_folds_variants():
    assert normalize.aggressive("إجازة") == normalize.aggressive("اجازه")
    assert "5" in normalize.aggressive("٥ أيام")


def test_looks_arabic():
    assert normalize.looks_arabic("هذا نص عربي واضح")
    assert not normalize.looks_arabic("this is entirely english text")
    assert not normalize.looks_arabic("")


def test_template_generation_is_deterministic():
    a = mockdata.generate_templates(n_per_class=3, seed=42)
    b = mockdata.generate_templates(n_per_class=3, seed=42)
    assert [d.doc_id for d in a] == [d.doc_id for d in b]
    assert [d.text for d in a] == [d.text for d in b]


def test_generated_docs_are_labeled_and_arabic():
    tax = taxonomy.load()
    docs = mockdata.generate_templates(n_per_class=2, seed=1)
    assert len(docs) >= len(tax.institutions) * 2
    for doc in docs:
        assert doc.true_label in tax.ids
        assert normalize.looks_arabic(doc.text)


def test_curated_corpus_loads_and_covers_every_institution():
    tax = taxonomy.load()
    docs = mockdata.generate_curated()
    assert len(docs) >= 80
    covered = {d.true_label for d in docs}
    assert covered == set(tax.ids), f"uncovered: {set(tax.ids) - covered}"


def test_curated_ids_are_unique():
    docs = mockdata.generate_curated()
    ids = [d.doc_id for d in docs]
    assert len(set(ids)) == len(ids)


def test_curated_is_deterministic():
    a = mockdata.generate_curated(seed=11)
    b = mockdata.generate_curated(seed=11)
    assert [(d.doc_id, d.text) for d in a] == [(d.doc_id, d.text) for d in b]


def test_every_confusion_pair_has_boundary_cases_both_ways():
    # A pair with cases in only one direction tests half of what it should.
    tax = taxonomy.load()
    import yaml

    entries = []
    for file in sorted(mockdata.CURATED_DIR.glob("corpus_part*.yaml")):
        payload = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        entries.extend(payload.get("documents", []))

    tagged = [e for e in entries if e.get("pair")]
    for a, b in tax.confusion_pairs:
        key = f"{a}|{b}"
        labels = {e["label"] for e in tagged if e.get("pair") == key}
        assert labels == {a, b}, f"pair {key} lacks cases both ways; has {labels}"


def test_curated_hard_only_subset_is_smaller_and_all_hard():
    everything = mockdata.generate_curated()
    hard = mockdata.generate_curated(hard_only=True)
    assert 0 < len(hard) < len(everything)
    hard_ids = {d.doc_id for d in hard}
    assert hard_ids.issubset({d.doc_id for d in everything})


def test_curated_is_harder_than_templates_for_a_keyword_model():
    # The whole reason the curated corpus exists. If this ever fails, the
    # corpus has stopped being a useful benchmark.
    tax = taxonomy.load()
    templates = mockdata.generate_templates(n_per_class=20, seed=7, ocr_noise_ratio=0.0)
    curated = mockdata.generate_curated(seed=7, ocr_noise_ratio=0.0)

    clf = BaselineClassifier()
    clf.fit(templates)

    on_templates = evaluate(templates, clf.classify_many(templates), taxonomy=tax)
    on_curated = evaluate(curated, clf.classify_many(curated), taxonomy=tax)
    assert on_curated.accuracy < on_templates.accuracy - 0.05


def test_ocr_noise_perturbs_but_keeps_length_close():
    import random

    text = "تعرضت لحادث مروري على الطريق السريع وأرجو التكرم بالتوجيه"
    noisy = mockdata.add_ocr_noise(text, random.Random(3), rate=0.1)
    assert noisy != text
    assert abs(len(noisy) - len(text)) < len(text) * 0.3


def test_baseline_trains_and_predicts():
    docs = mockdata.generate_templates(n_per_class=12, seed=5, ocr_noise_ratio=0.0)
    split = int(len(docs) * 0.75)
    clf = BaselineClassifier()
    clf.fit(docs[:split])
    preds = clf.classify_many(docs[split:])

    assert len(preds) == len(docs[split:])
    tax = taxonomy.load()
    for p in preds:
        assert p.institution_id in tax.ids
        assert 0.0 <= p.confidence <= 1.0

    report = evaluate(docs[split:], preds)
    # Templates are easy by construction; this is a smoke floor, not a target.
    assert report.accuracy > 0.5
    assert report.total == len(preds)


def test_evaluate_separates_auto_from_held():
    docs = [
        Document(doc_id="a", text="نص", true_label="moh_health"),
        Document(doc_id="b", text="نص", true_label="moj_courts"),
    ]
    from docrouter.models import Prediction

    preds = [
        Prediction(doc_id="a", institution_id="moh_health", confidence=0.9, backend="t"),
        Prediction(
            doc_id="b",
            institution_id="moh_health",
            confidence=0.2,
            needs_review=True,
            backend="t",
        ),
    ]
    report = evaluate(docs, preds)
    assert report.total == 2
    assert report.accuracy == 0.5
    assert report.auto_routed == 1
    assert report.auto_accuracy == 1.0
    assert report.held_for_review == 1
    assert report.review_would_have_been_correct == 0


def test_evaluate_rejects_unmatched_predictions():
    from docrouter.models import Prediction

    docs = [Document(doc_id="a", text="نص", true_label="moh_health")]
    preds = [Prediction(doc_id="zzz", institution_id="moh_health", confidence=0.9, backend="t")]
    with pytest.raises(ValueError):
        evaluate(docs, preds)
