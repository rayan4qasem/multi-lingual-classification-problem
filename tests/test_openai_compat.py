"""The self-hosted gateway backend. All offline — no server is contacted.

The load-bearing test is
`test_an_invented_institution_is_rejected_even_without_schema_support`: the
Claude backend gets its enum guarantee from the API, but an arbitrary gateway
may enforce nothing, so the taxonomy check has to hold on our side.
"""

from __future__ import annotations

import json

import pytest

from docrouter import taxonomy
from docrouter.classify.openai_compat import (
    STRATEGIES,
    GatewayError,
    GatewayUnavailable,
    OpenAICompatClassifier,
    OpenAICompatClient,
    _extract_json,
)
from docrouter.models import Document
from docrouter.protocols import BatchClassifier, Classifier

TAX = taxonomy.load()
DOC = Document(doc_id="d1", text="أفيدكم بسرقة مركبتي وأرغب في تقديم بلاغ. رقم الهوية 1042779318")

GOOD = {
    "institution_id": "interior_public_security",
    "confidence": 0.82,
    "rationale_ar": "بلاغ عن سرقة في مرحلة الضبط",
    "alternatives": [{"institution_id": "public_prosecution", "confidence": 0.3}],
}


class FakeClient:
    """Records what was sent and replays scripted replies."""

    def __init__(self, replies=None, reject: set[str] | None = None):
        self.base_url = "http://gateway.test/v1"
        self.calls: list[dict] = []
        self.reject = reject or set()
        self._replies = replies if replies is not None else [json.dumps(GOOD)]

    def chat(self, model, messages, response_format=None, **kw):
        kind = (response_format or {}).get("type", "prompt")
        self.calls.append({"model": model, "messages": messages, "format": kind})
        if kind in self.reject:
            raise GatewayError(f"/chat/completions returned 400: {kind} unsupported")
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def list_models(self):
        return ["gpt-oss", "bge-m3"]


def _clf(client=None, **kw):
    return OpenAICompatClassifier(taxonomy=TAX, client=client or FakeClient(), **kw)


# ---------- the contract ----------


def test_satisfies_the_classifier_protocol_but_not_the_batch_one():
    clf = _clf()
    assert isinstance(clf, Classifier)
    # No asynchronous batch endpoint exists on a plain chat gateway.
    assert not isinstance(clf, BatchClassifier)


def test_a_normal_reply_becomes_a_prediction():
    clf = _clf()
    p = clf.classify(DOC)
    assert p.institution_id == "interior_public_security"
    assert p.confidence == pytest.approx(0.82)
    assert p.alternatives[0].institution_id == "public_prosecution"
    assert p.backend == clf.name


# ---------- strategy negotiation ----------


def test_strongest_strategy_is_tried_first():
    client = FakeClient()
    clf = _clf(client)
    clf.classify(DOC)
    assert client.calls[0]["format"] == "json_schema"
    assert clf.strategy == "json_schema"


def test_falls_back_when_the_server_rejects_json_schema():
    client = FakeClient(reject={"json_schema"})
    clf = _clf(client)
    clf.classify(DOC)
    assert [c["format"] for c in client.calls] == ["json_schema", "json_object"]
    assert clf.strategy == "json_object"


def test_falls_all_the_way_back_to_prompt_only():
    client = FakeClient(reject={"json_schema", "json_object"})
    clf = _clf(client)
    clf.classify(DOC)
    assert [c["format"] for c in client.calls] == list(STRATEGIES)
    assert clf.strategy == "prompt"


def test_negotiation_happens_once_not_per_document():
    client = FakeClient(reject={"json_schema"})
    clf = _clf(client)
    for _ in range(4):
        clf.classify(DOC)
    # One wasted call total, not one per document.
    assert [c["format"] for c in client.calls].count("json_schema") == 1


def test_a_weaker_strategy_is_recorded_in_the_backend_name():
    client = FakeClient(reject={"json_schema"})
    clf = _clf(client)
    p = clf.classify(DOC)
    # A run without server-side enforcement must not look like one with it.
    assert "json_object" in p.backend
    assert "json_object" not in _clf().classify(DOC).backend


def test_weaker_strategies_describe_the_shape_in_the_prompt():
    client = FakeClient(reject={"json_schema", "json_object"})
    clf = _clf(client)
    clf.classify(DOC)
    system = client.calls[-1]["messages"][0]["content"]
    assert "صيغة الإخراج" in system
    assert "institution_id" in system


def test_unreachable_server_is_not_mistaken_for_an_unsupported_strategy():
    class Down(FakeClient):
        def chat(self, *a, **kw):
            raise GatewayUnavailable("cannot reach http://gateway.test/v1")

    with pytest.raises(GatewayUnavailable):
        _clf(Down()).classify(DOC)


def test_total_failure_reports_every_strategy_tried():
    client = FakeClient(reject=set(STRATEGIES))
    with pytest.raises(GatewayError, match="no output strategy worked"):
        _clf(client).classify(DOC)


# ---------- the guarantee that must not depend on the server ----------


def test_an_invented_institution_is_rejected_even_without_schema_support():
    bad = dict(GOOD, institution_id="ministry_of_magic")
    client = FakeClient(replies=[json.dumps(bad)] * 5, reject={"json_schema", "json_object"})
    with pytest.raises(GatewayError):
        _clf(client).classify(DOC)


def test_invalid_alternatives_are_dropped_not_fatal():
    raw = dict(
        GOOD,
        alternatives=[
            {"institution_id": "not_real", "confidence": 0.4},
            {"institution_id": "moj_courts", "confidence": 0.2},
        ],
    )
    p = _clf(FakeClient(replies=[json.dumps(raw)])).classify(DOC)
    assert [a.institution_id for a in p.alternatives] == ["moj_courts"]


def test_every_returned_id_is_a_real_taxonomy_id():
    p = _clf().classify(DOC)
    assert p.institution_id in TAX.ids


# ---------- reply parsing ----------


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "```\n" + json.dumps(GOOD) + "\n```",
        "هذا هو التصنيف:\n" + json.dumps(GOOD) + "\nانتهى.",
    ],
)
def test_json_is_recovered_from_fences_and_surrounding_prose(reply):
    # A model without schema enforcement wraps its answer in whatever it likes.
    assert _extract_json(reply)["institution_id"] == GOOD["institution_id"]


def test_a_reply_with_no_json_at_all_fails_clearly():
    with pytest.raises(GatewayError, match="no JSON object"):
        _extract_json("عذراً، لا أستطيع تحديد الجهة.")


# ---------- privacy ----------


def test_identifiers_are_redacted_before_leaving_the_process():
    client = FakeClient()
    clf = _clf(client)
    clf.classify(DOC)
    sent = client.calls[0]["messages"][1]["content"]
    assert "1042779318" not in sent
    assert "[رقم هوية]" in sent


def test_redaction_can_be_disabled_and_is_recorded():
    client = FakeClient()
    clf = _clf(client, redact_pii=False)
    p = clf.classify(DOC)
    assert "1042779318" in client.calls[0]["messages"][1]["content"]
    assert p.backend.endswith("+raw")


def test_redaction_tally_is_kept():
    clf = _clf()
    clf.classify(DOC)
    assert clf.redaction_counts.get("national_id") == 1


# ---------- configuration ----------


def test_model_falls_back_through_env_names(monkeypatch):
    monkeypatch.delenv("DOCROUTER_LOCAL_MODEL", raising=False)
    monkeypatch.setenv("COMPANY_LLM_MODEL", "gpt-oss")
    assert _clf().model == "gpt-oss"
    monkeypatch.setenv("DOCROUTER_LOCAL_MODEL", "override")
    assert _clf().model == "override"


def test_a_missing_gateway_url_is_a_clear_error(monkeypatch):
    for name in ("DOCROUTER_LOCAL_URL", "COMPANY_API_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(GatewayError, match="no gateway URL"):
        OpenAICompatClassifier(taxonomy=TAX)


def test_base_url_gets_the_v1_suffix_once(monkeypatch):
    from docrouter.classify.openai_compat import OpenAICompatClient

    monkeypatch.setenv("COMPANY_API_KEY", "unused-in-this-test")
    assert OpenAICompatClient(base_url="https://host:8443").base_url == "https://host:8443/v1"
    assert OpenAICompatClient(base_url="https://host:8443/v1").base_url == "https://host:8443/v1"
    assert OpenAICompatClient(base_url="https://host:8443/").base_url == "https://host:8443/v1"


def test_preflight_reports_reachability_without_raising():
    report = _clf().preflight()
    assert report["reachable"] is True
    assert "gpt-oss" in report["available_models"]


def test_preflight_reports_an_unreachable_gateway():
    class Down(FakeClient):
        def list_models(self):
            raise GatewayUnavailable("no route to host")

    report = _clf(Down()).preflight()
    assert report["reachable"] is False
    assert "no route" in report["error"]


def test_reasoning_effort_is_omitted_unless_asked_for():
    """A gateway that does not know the parameter may reject it, not ignore it."""
    sent = {}

    class Recorder(OpenAICompatClient):
        def __init__(self):
            pass

        def _request(self, path, payload=None):
            sent.update(payload or {})
            return {"choices": [{"message": {"content": "{}"}}]}

    Recorder().chat(model="m", messages=[])
    assert "reasoning_effort" not in sent

    Recorder().chat(model="m", messages=[], reasoning_effort="low")
    assert sent["reasoning_effort"] == "low"
