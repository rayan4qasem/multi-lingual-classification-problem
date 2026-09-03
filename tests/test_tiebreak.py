"""Tests for confusion-pair adjudication.

The policy half (when to adjudicate) is taxonomy logic and is tested directly
against a real taxonomy. The model half is tested through a stub resolver, so
none of this needs a gateway.
"""

from __future__ import annotations

import pytest

from docrouter.classify.tiebreak import (
    LLMPairResolver,
    PairIndex,
    TiebreakClassifier,
)
from docrouter.models import Alternative, Document, Prediction
from docrouter.taxonomy import load as load_taxonomy

PAIR_A = "hrsd_labor"
PAIR_B = "gosi"
UNPAIRED = "moh_health"


@pytest.fixture
def tax():
    return load_taxonomy()


def doc(text: str = "نص الوثيقة", doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, text=text, true_label=PAIR_B)


def pred(top: str, alts: list[str], confidence: float = 0.5) -> Prediction:
    return Prediction(
        doc_id="d1",
        institution_id=top,
        confidence=confidence,
        backend="stub",
        alternatives=[Alternative(institution_id=a, confidence=0.3) for a in alts],
    )


class StubClassifier:
    def __init__(self, prediction: Prediction) -> None:
        self.prediction = prediction
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    def classify(self, document: Document) -> Prediction:
        self.calls += 1
        return self.prediction.model_copy(update={"doc_id": document.doc_id})

    def classify_many(self, documents):
        return [self.classify(d) for d in documents]


class StubResolver:
    def __init__(self, choice: str, confidence: float = 0.88, fail: bool = False):
        self.choice = choice
        self.confidence = confidence
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def resolve(self, document: Document, a: str, b: str) -> tuple[str, float, str]:
        self.calls.append((a, b))
        if self.fail:
            raise RuntimeError("gateway down")
        return self.choice, self.confidence, "لأن الإجراء المطلوب هو التسجيل"


# --- PairIndex: the policy ------------------------------------------------


def test_declared_pairs_are_order_insensitive(tax):
    index = PairIndex(tax)
    assert index.contains(PAIR_A, PAIR_B)
    assert index.contains(PAIR_B, PAIR_A)


def test_undeclared_pair_is_not_matched(tax):
    assert not PairIndex(tax).contains(PAIR_A, UNPAIRED)


def test_partner_found_beyond_first_alternative(tax):
    """A genuine rival ranked second must still be adjudicated."""
    index = PairIndex(tax)
    p = pred(PAIR_A, [UNPAIRED, PAIR_B])
    assert index.partner(p) == PAIR_B


def test_partner_ignores_a_self_referential_alternative(tax):
    index = PairIndex(tax)
    assert index.partner(pred(PAIR_A, [PAIR_A])) is None


def test_partner_none_when_no_alternatives(tax):
    assert PairIndex(tax).partner(pred(PAIR_A, [])) is None


# --- TiebreakClassifier: the wiring ---------------------------------------


def test_adjudicates_and_overturns_a_pair(tax):
    base = StubClassifier(pred(PAIR_A, [PAIR_B]))
    resolver = StubResolver(PAIR_B)
    clf = TiebreakClassifier(base, resolver, tax)

    out = clf.classify(doc())

    assert out.institution_id == PAIR_B
    assert out.confidence == pytest.approx(0.88)
    assert resolver.calls == [(PAIR_A, PAIR_B)]
    assert clf.adjudicated == 1 and clf.changed == 1
    assert out.backend.startswith("tiebreak(")


def test_adjudication_may_confirm_the_first_pass(tax):
    base = StubClassifier(pred(PAIR_A, [PAIR_B]))
    clf = TiebreakClassifier(base, StubResolver(PAIR_A), tax)

    out = clf.classify(doc())

    assert out.institution_id == PAIR_A
    assert clf.adjudicated == 1 and clf.changed == 0


def test_no_second_pass_when_pair_is_not_declared(tax):
    resolver = StubResolver(UNPAIRED)
    clf = TiebreakClassifier(StubClassifier(pred(PAIR_A, [UNPAIRED])), resolver, tax)

    out = clf.classify(doc())

    assert resolver.calls == []
    assert out.institution_id == PAIR_A
    assert clf.adjudicated == 0


def test_no_second_pass_when_first_pass_is_confident(tax):
    resolver = StubResolver(PAIR_B)
    clf = TiebreakClassifier(
        StubClassifier(pred(PAIR_A, [PAIR_B], confidence=0.97)),
        resolver,
        tax,
        max_confidence=0.90,
    )

    clf.classify(doc())

    assert resolver.calls == []


def test_a_failed_second_opinion_never_loses_the_first(tax):
    """The decorator must be safe to enable: on failure, route as before."""
    base = StubClassifier(pred(PAIR_A, [PAIR_B]))
    clf = TiebreakClassifier(base, StubResolver(PAIR_B, fail=True), tax)

    out = clf.classify(doc())

    assert out.institution_id == PAIR_A
    assert out.backend == "stub"
    assert clf.adjudicated == 1 and clf.changed == 0


def test_classify_many_preserves_doc_ids(tax):
    clf = TiebreakClassifier(StubClassifier(pred(PAIR_A, [PAIR_B])), StubResolver(PAIR_B), tax)
    out = clf.classify_many([doc(doc_id="a"), doc(doc_id="b")])
    assert [p.doc_id for p in out] == ["a", "b"]


def test_name_reports_the_wrapped_backend(tax):
    clf = TiebreakClassifier(StubClassifier(pred(PAIR_A, [PAIR_B])), StubResolver(PAIR_B), tax)
    assert clf.name == "tiebreak(stub)"


# --- LLMPairResolver: the prompt and its guard ----------------------------


class StubClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict] = []

    def chat(self, model, messages, response_format=None, **kwargs):
        self.messages = messages
        return self.reply


def test_prompt_carries_both_disambiguation_rules(tax):
    client = StubClient("{}")
    resolver = LLMPairResolver(client, "m", tax)
    text = resolver.prompt_for(doc(), PAIR_A, PAIR_B)

    assert tax.get(PAIR_A).disambiguation_ar in text
    assert tax.get(PAIR_B).disambiguation_ar in text
    assert "الإجراء المطلوب" in text
    assert "نص الوثيقة" in text


def test_resolver_returns_the_chosen_institution(tax):
    client = StubClient('{"institution_id": "gosi", "confidence": 0.8, "rationale_ar": "التسجيل"}')
    chosen, confidence, rationale = LLMPairResolver(client, "m", tax).resolve(doc(), PAIR_A, PAIR_B)
    assert (chosen, confidence, rationale) == (PAIR_B, pytest.approx(0.8), "التسجيل")


def test_resolver_rejects_an_answer_outside_the_pair(tax):
    """A binary question answered with a third option is not an answer."""
    client = StubClient(f'{{"institution_id": "{UNPAIRED}", "confidence": 0.9}}')
    with pytest.raises(ValueError, match="expected"):
        LLMPairResolver(client, "m", tax).resolve(doc(), PAIR_A, PAIR_B)


def test_resolver_clamps_an_out_of_range_confidence(tax):
    client = StubClient('{"institution_id": "gosi", "confidence": 4.2}')
    _, confidence, _ = LLMPairResolver(client, "m", tax).resolve(doc(), PAIR_A, PAIR_B)
    assert confidence == 1.0
