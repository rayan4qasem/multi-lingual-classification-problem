"""Few-shot selection and rendering. All offline."""

from __future__ import annotations

from docrouter import fewshot, mockdata, taxonomy
from docrouter.classify.llm import SYSTEM_PREAMBLE, LLMClassifier
from docrouter.models import Document


class StubClient:
    """Stands in for anthropic.Anthropic so no key is needed."""


def _corpus(n=40):
    return mockdata.generate_curated(seed=4)[:n]


# ---------- redaction ----------


def test_redacts_saudi_identifiers():
    text = (
        "رقم الهوية: 1042779318 والجوال 0553120944 "
        "والآيبان SA4420000001234567891234 "
        "والرقم الضريبي 300442119800003 والبريد a.b@example.com"
    )
    out = fewshot.redact(text)
    for secret in (
        "1042779318",
        "0553120944",
        "SA4420000001234567891234",
        "300442119800003",
        "a.b@example.com",
    ):
        assert secret not in out
    assert "[رقم هوية]" in out and "[جوال]" in out
    assert "[آيبان]" in out and "[رقم ضريبي]" in out and "[بريد]" in out


def test_redaction_keeps_dates_and_amounts():
    # These carry routing signal and must survive.
    text = "بتاريخ 12 رجب 1446هـ بمبلغ 46,500 ريال عن 4 أشهر"
    out = fewshot.redact(text)
    assert "1446" in out and "46,500" in out and "4 أشهر" in out


def test_redaction_is_idempotent():
    text = "الهوية 1042779318"
    assert fewshot.redact(fewshot.redact(text)) == fewshot.redact(text)


# ---------- selection ----------


def test_every_class_with_gold_gets_an_example():
    tax = taxonomy.load()
    docs = mockdata.generate_curated(seed=4)
    gold = {d.doc_id: d.true_label for d in docs}
    chosen = fewshot.select_examples(docs, gold, taxonomy=tax, per_class=1, max_examples=50)
    assert set(chosen.per_class()) == set(tax.ids)


def test_overrides_are_preferred_over_confirmations():
    docs = _corpus(40)
    gold = {d.doc_id: d.true_label for d in docs}
    # Make three documents look like human corrections.
    corrected = [d.doc_id for d in docs[:3]]
    model_labels = {d.doc_id: d.true_label for d in docs}
    for doc_id in corrected:
        model_labels[doc_id] = "zatca" if gold[doc_id] != "zatca" else "gosi"

    chosen = fewshot.select_examples(
        docs, gold, model_labels=model_labels, per_class=0, max_examples=3
    )
    assert {e.doc_id for e in chosen.examples} == set(corrected)
    assert all(e.is_override for e in chosen.examples)


def test_override_records_what_the_model_got_wrong():
    docs = _corpus(5)
    gold = {docs[0].doc_id: "moj_courts"}
    chosen = fewshot.select_examples(
        docs, gold, model_labels={docs[0].doc_id: "public_prosecution"}, max_examples=1
    )
    example = chosen.examples[0]
    assert example.corrected_from == "public_prosecution"
    assert example.label == "moj_courts"


def test_max_examples_is_respected():
    docs = _corpus(40)
    gold = {d.doc_id: d.true_label for d in docs}
    chosen = fewshot.select_examples(docs, gold, max_examples=7, per_class=1)
    assert len(chosen.examples) == 7


def test_truncation_is_marked():
    docs = [Document(doc_id="d1", text="نص طويل جداً. " * 400, true_label="gosi")]
    chosen = fewshot.select_examples(docs, {"d1": "gosi"}, max_chars=200)
    example = chosen.examples[0]
    assert example.truncated is True
    assert len(example.text) < 300 and example.text.endswith("…")


def test_unknown_labels_and_missing_documents_are_dropped():
    docs = _corpus(3)
    gold = {docs[0].doc_id: "not_an_institution", "ghost": "gosi"}
    assert fewshot.select_examples(docs, gold).examples == []


def test_empty_documents_are_skipped():
    docs = [Document(doc_id="blank", text="   ", true_label="gosi")]
    assert fewshot.select_examples(docs, {"blank": "gosi"}).examples == []


# ---------- rendering and caching ----------


def test_examples_block_is_byte_stable():
    # The block sits in the cached prompt prefix. If it varies between runs
    # the cache never hits and the per-document cost roughly triples.
    docs = _corpus(30)
    gold = {d.doc_id: d.true_label for d in docs}
    a = fewshot.render(fewshot.select_examples(docs, gold, max_examples=10))
    b = fewshot.render(fewshot.select_examples(docs, gold, max_examples=10))
    assert a == b


def test_selection_order_is_sorted_not_incidental():
    docs = _corpus(30)
    gold = {d.doc_id: d.true_label for d in docs}
    chosen = fewshot.select_examples(docs, gold, max_examples=12)
    pairs = [(e.label, e.doc_id) for e in chosen.examples]
    assert pairs == sorted(pairs)


def test_render_names_the_institution_and_the_correction():
    docs = _corpus(5)
    gold = {docs[0].doc_id: "moj_courts"}
    chosen = fewshot.select_examples(
        docs, gold, model_labels={docs[0].doc_id: "public_prosecution"}, max_examples=1
    )
    block = fewshot.render(chosen)
    assert "moj_courts" in block
    assert "وزارة العدل" in block
    assert "public_prosecution" in block  # the correction is stated


def test_empty_example_set_renders_nothing():
    assert fewshot.render(fewshot.ExampleSet()) == ""


def test_classifier_appends_examples_to_the_cached_prefix():
    tax = taxonomy.load()
    docs = _corpus(20)
    gold = {d.doc_id: d.true_label for d in docs}
    chosen = fewshot.select_examples(docs, gold, taxonomy=tax, max_examples=5)

    plain = LLMClassifier(taxonomy=tax, client=StubClient())
    shot = LLMClassifier(taxonomy=tax, client=StubClient(), examples=chosen)

    assert len(shot._system) > len(plain._system)
    assert shot._system.startswith(SYSTEM_PREAMBLE)
    # Still exactly one cached block, so the prefix stays cacheable.
    blocks = shot._request_kwargs()["system"]
    assert len(blocks) == 1
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "5shot" in shot.name


def test_classifier_prefix_is_stable_across_construction():
    tax = taxonomy.load()
    docs = _corpus(20)
    gold = {d.doc_id: d.true_label for d in docs}
    chosen = fewshot.select_examples(docs, gold, taxonomy=tax, max_examples=5)
    a = LLMClassifier(taxonomy=tax, client=StubClient(), examples=chosen)._system
    b = LLMClassifier(taxonomy=tax, client=StubClient(), examples=chosen)._system
    assert a == b


# ---------- leakage ----------


def test_leakage_is_detected():
    docs = _corpus(20)
    gold = {d.doc_id: d.true_label for d in docs}
    chosen = fewshot.select_examples(docs, gold, max_examples=6)
    assert fewshot.check_leakage(chosen, docs) == chosen.doc_ids

    unseen = [Document(doc_id="fresh", text="نص", true_label="gosi")]
    assert fewshot.check_leakage(chosen, unseen) == set()


def test_roundtrip(tmp_path):
    docs = _corpus(20)
    gold = {d.doc_id: d.true_label for d in docs}
    chosen = fewshot.select_examples(docs, gold, max_examples=6)
    path = fewshot.save(chosen, tmp_path / "ex.json")
    assert fewshot.load(path).model_dump() == chosen.model_dump()
