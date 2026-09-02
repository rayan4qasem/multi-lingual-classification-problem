"""Identifier masking, and the guarantee that it reaches the API path.

The load-bearing test here is
`test_no_identifier_survives_into_a_classification_request`: every other test
checks a helper, that one checks the actual bytes that would leave the
machine.
"""

from __future__ import annotations

import json

import pytest

from docrouter import mockdata, privacy, taxonomy
from docrouter.classify import BaselineClassifier, LLMClassifier
from docrouter.classify.llm import _prepare_text, _user_content
from docrouter.evaluate import evaluate
from docrouter.models import Document

IDENTIFIERS = {
    "national id": "1042779318",
    "iqama": "2447811093",
    "phone": "0553120944",
    "phone intl": "+966553120944",
    "iban": "SA4420000001234567891234",
    "tax number": "300442119800003",
    "email": "a.b@example.com",
    "long case number": "6633120945112",
}

SAMPLE = (
    "المكرم مدير المركز\n"
    "أفيدكم بسرقة مركبتي بتاريخ 12 رجب 1446هـ بمبلغ تقديري 46,500 ريال.\n"
    "رقم الهوية: 1042779318\n"
    "الجوال: 0553120944\n"
    "الآيبان: SA4420000001234567891234\n"
    "الرقم الضريبي للمنشأة: 300442119800003\n"
    "البريد: a.b@example.com\n"
    "رقم المعاملة: 6633120945112\n"
)


class StubClient:
    """Stands in for anthropic.Anthropic so no key is needed."""


# ---------- the rules ----------


@pytest.mark.parametrize("label,value", sorted(IDENTIFIERS.items()))
def test_every_identifier_class_is_masked(label, value):
    assert value not in privacy.redact(f"البيان: {value} انتهى")


def test_dates_and_amounts_survive():
    # These carry routing signal and must not be collateral damage.
    text = "بتاريخ 12 رجب 1446هـ بمبلغ 46,500 ريال عن 4 أشهر، القضية رقم 46/1187"
    out = privacy.redact(text)
    assert "1446" in out and "46,500" in out and "4 أشهر" in out and "46/1187" in out


def test_placeholders_are_typed_not_blanked():
    """A typed placeholder keeps the routing signal a bare mask would destroy:
    'this document mentions a tax number' still points at ZATCA."""
    out = privacy.redact(SAMPLE)
    for placeholder in ("[رقم هوية]", "[جوال]", "[آيبان]", "[رقم ضريبي]", "[بريد]", "[رقم]"):
        assert placeholder in out


def test_specific_rules_win_over_the_generic_digit_run():
    # A tax number must not degrade to an untyped [رقم].
    out = privacy.redact("الرقم الضريبي 300442119800003")
    assert "[رقم ضريبي]" in out and "[رقم]" not in out


def test_redaction_is_idempotent():
    once = privacy.redact(SAMPLE)
    assert privacy.redact(once) == once


def test_scan_counts_without_modifying():
    counts = privacy.scan(SAMPLE)
    assert counts["national_id"] == 1
    assert counts["phone"] == 1
    assert counts["tax_number"] == 1
    assert counts["email"] == 1
    # Counted once under its own rule, not again as a long number.
    assert sum(counts.values()) == 6


def test_redact_with_counts_agrees_with_the_parts():
    text, counts = privacy.redact_with_counts(SAMPLE)
    assert text == privacy.redact(SAMPLE)
    assert counts == privacy.scan(SAMPLE)


def test_summarize_is_readable():
    assert privacy.summarize({}) == "no identifiers found"
    assert "national_id=1" in privacy.summarize({"national_id": 1})


def test_empty_and_clean_text_are_untouched():
    assert privacy.redact("") == ""
    clean = "خطاب بلا أرقام تعريفية على الإطلاق"
    assert privacy.redact(clean) == clean


# ---------- the guarantee that matters ----------


def test_no_identifier_survives_into_a_classification_request():
    """The bytes that would actually leave the machine."""
    doc = Document(doc_id="d1", text=SAMPLE)
    payload = _user_content(doc, redact_pii=True)
    for label, value in IDENTIFIERS.items():
        assert value not in payload, f"{label} leaked into the request"


def test_redaction_runs_before_truncation():
    """A long document is sent as head+tail; an identifier sitting in the
    middle must already be masked, not merely elided by luck."""
    doc = Document(doc_id="long", text="حشو. " * 6000 + "رقم الهوية 1042779318 " + "حشو. " * 6000)
    out = _prepare_text(doc, redact_pii=True)
    assert "1042779318" not in out


def test_opting_out_is_explicit_and_recorded():
    doc = Document(doc_id="d1", text=SAMPLE)
    assert "1042779318" in _user_content(doc, redact_pii=False)

    clf = LLMClassifier(client=StubClient(), redact_pii=False)
    # A run made without redaction must be identifiable after the fact.
    assert clf.name.endswith("+raw")
    assert not LLMClassifier(client=StubClient()).name.endswith("+raw")


def test_redaction_is_on_by_default():
    assert LLMClassifier(client=StubClient()).redact_pii is True
    doc = Document(doc_id="d1", text=SAMPLE)
    assert "1042779318" not in _user_content(doc)


def test_batch_requests_are_redacted_too():
    """The batch path is the one that would process a whole archive."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    doc = Document(doc_id="d1", text=SAMPLE)
    clf = LLMClassifier(client=StubClient())
    params = MessageCreateParamsNonStreaming(  # type: ignore[typeddict-item]
        **clf._request_kwargs(),
        messages=[{"role": "user", "content": _user_content(doc, clf.redact_pii)}],
    )
    blob = json.dumps(params, ensure_ascii=False)
    for value in IDENTIFIERS.values():
        assert value not in blob


def test_classifier_tallies_what_it_masked():
    clf = LLMClassifier(client=StubClient())
    clf._track(Document(doc_id="d1", text=SAMPLE))
    clf._track(Document(doc_id="d2", text=SAMPLE))
    assert clf.redaction_counts["national_id"] == 2
    assert clf.redaction_counts["email"] == 2


def test_no_tally_when_redaction_is_off():
    clf = LLMClassifier(client=StubClient(), redact_pii=False)
    clf._track(Document(doc_id="d1", text=SAMPLE))
    assert clf.redaction_counts == {}


# ---------- does it cost accuracy? ----------


def test_redaction_does_not_degrade_routing_signal():
    """Measured, not assumed.

    The LLM path cannot be measured without an API key, so this uses the
    offline baseline as a proxy for the question that matters: does masking
    identifiers destroy information the classifier was relying on? Typed
    placeholders should mean no.
    """
    tax = taxonomy.load()
    train = mockdata.generate_templates(n_per_class=20, seed=7, ocr_noise_ratio=0.0)
    test = mockdata.generate_curated(seed=7, ocr_noise_ratio=0.0)

    clf = BaselineClassifier(taxonomy=tax)
    clf.fit(train)

    plain = evaluate(test, clf.classify_many(test), taxonomy=tax)
    masked_docs = [d.model_copy(update={"text": privacy.redact(d.text)}) for d in test]
    masked = evaluate(test, clf.classify_many(masked_docs), taxonomy=tax)

    assert masked.accuracy >= plain.accuracy - 0.05, (
        f"redaction cost {plain.accuracy - masked.accuracy:.1%} accuracy "
        "— the placeholders are destroying routing signal"
    )
