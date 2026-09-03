"""Classifier for an OpenAI-compatible endpoint — LiteLLM, vLLM, LM Studio,
llama.cpp, or Ollama's `/v1` shim.

This is the backend for a self-hosted gateway, where the point is that no
document leaves the network. One implementation covers every server that
speaks the OpenAI chat API, which is nearly all of them.

**Structured output is negotiated, not assumed.** The Claude backend can rely
on `output_config.format`; an arbitrary gateway cannot. So this tries three
strategies in descending order of strength and remembers which one worked:

  json_schema   the server enforces the schema, institution_id is an enum
  json_object   the server guarantees valid JSON but not the shape
  prompt        no guarantee at all; the schema is described in words

Whatever the server does, `institution_id` is validated against the taxonomy
before a Prediction is built. A model that invents a destination fails loudly
rather than quietly routing a document somewhere that does not exist — which
is the guarantee the whole design rests on, and the one thing that must not
depend on the server being cooperative.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence

from .. import __version__
from ..fewshot import ExampleSet
from ..fewshot import render as render_examples
from ..models import Alternative, Document, LLMClassification, Prediction
from ..privacy import scan
from ..taxonomy import Taxonomy
from ..taxonomy import load as load_taxonomy
from .llm import SYSTEM_PREAMBLE, _user_content, build_schema

STRATEGIES = ("json_schema", "json_object", "prompt")

USER_AGENT = f"docrouter/{__version__}"


class GatewayError(RuntimeError):
    """Any failure talking to the gateway."""


class GatewayUnavailable(GatewayError):
    """The gateway could not be reached."""


class InvalidInstitution(GatewayError):
    """The model returned an id that is not in the taxonomy."""


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


class OpenAICompatClient:
    """Minimal chat-completions client. Injectable, so it can be stubbed."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
    ):
        url = base_url or _env("DOCROUTER_LOCAL_URL", "COMPANY_API_URL") or ""
        if not url:
            raise GatewayError(
                "no gateway URL. Set COMPANY_API_URL (or DOCROUTER_LOCAL_URL) "
                "in your environment or .env"
            )
        self.base_url = url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        # Read but never logged, never written to disk.
        self.api_key = api_key or _env("DOCROUTER_LOCAL_KEY", "COMPANY_API_KEY") or ""
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        # urllib's default User-Agent is "Python-urllib/3.x", which WAFs in
        # front of hosted gateways reject outright — Cloudflare answers it with
        # a 403 and error code 1010 before the request ever reaches the API.
        # A self-hosted gateway does not care; a proxied one does.
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise GatewayError(f"{path} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise GatewayUnavailable(
                f"cannot reach {self.base_url} ({exc.reason}). "
                "If it is on a private network, check you are connected to it."
            ) from exc

    def list_models(self) -> list[str]:
        return [m.get("id", "") for m in self._request("/models").get("data", [])]

    def chat(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        reasoning_effort: str | None = None,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # Only sent when asked for. Reasoning models bill their thinking, and
        # on a fourteen-way lookup with the rules already in the prompt, low
        # effort answers as well as high — measured at 187 reasoning tokens
        # down to 40. Omitted by default because a gateway that does not know
        # the parameter may reject the request rather than ignore it.
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        body = self._request("/chat/completions", payload)
        choices = body.get("choices") or []
        if not choices:
            raise GatewayError(f"no choices in response: {str(body)[:300]}")
        return (choices[0].get("message") or {}).get("content") or ""


def _extract_json(text: str) -> dict:
    """Pull an object out of a reply that may be wrapped in prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise GatewayError(f"no JSON object in reply: {text[:200]!r}")


class OpenAICompatClassifier:
    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        model: str | None = None,
        review_threshold: float = 0.55,
        client: OpenAICompatClient | None = None,
        examples: ExampleSet | None = None,
        redact_pii: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        strategy: str | None = None,
        detail: str = "full",
        reasoning_effort: str | None = None,
    ):
        self.reasoning_effort = reasoning_effort
        self.taxonomy = taxonomy or load_taxonomy()
        self.model = model or _env("DOCROUTER_LOCAL_MODEL", "COMPANY_LLM_MODEL") or "gpt-oss"
        self.review_threshold = review_threshold
        self.client = client or OpenAICompatClient(base_url=base_url, api_key=api_key)
        self.schema = build_schema(self.taxonomy)
        self.examples = examples
        # Kept on even though nothing leaves the network: gateways log, and
        # identifiers are not what the routing decision runs on anyway.
        self.redact_pii = redact_pii
        self.redaction_counts: dict[str, int] = {}
        # None means "not negotiated yet"; set on the first successful call.
        self.strategy: str | None = strategy
        self._valid_ids = set(self.taxonomy.ids)

        self.detail = detail
        block = render_examples(examples, self.taxonomy) if examples else ""
        self._system = SYSTEM_PREAMBLE + self.taxonomy.render_for_prompt(detail) + block

    @property
    def name(self) -> str:
        n = len(self.examples.examples) if self.examples else 0
        base = f"local:{self.model}" + (f"+{n}shot" if n else "")
        if self.strategy and self.strategy != "json_schema":
            # Visible in every Prediction: a run without schema enforcement
            # is a weaker guarantee and should not look identical to one with.
            base += f"+{self.strategy}"
        return base if self.redact_pii else base + "+raw"

    def _response_format(self, strategy: str) -> dict | None:
        if strategy == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": "routing", "strict": True, "schema": self.schema},
            }
        if strategy == "json_object":
            return {"type": "json_object"}
        return None

    def _messages(self, doc: Document, strategy: str) -> list[dict]:
        system = self._system
        if strategy != "json_schema":
            # The server will not enforce the shape, so state it in words.
            system += (
                "\n\n## صيغة الإخراج\n"
                "أعد كائن JSON فقط، بلا أي نص قبله أو بعده، بهذه الحقول:\n"
                '{"institution_id": "<معرّف من القائمة>", "confidence": <رقم بين 0 و1>, '
                '"rationale_ar": "<سبب مختصر>", "alternatives": '
                '[{"institution_id": "<معرّف>", "confidence": <رقم>}]}'
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _user_content(doc, self.redact_pii)},
        ]

    def _track(self, doc: Document) -> None:
        if not self.redact_pii:
            return
        for name, count in scan(doc.text).items():
            self.redaction_counts[name] = self.redaction_counts.get(name, 0) + count

    def _to_prediction(self, doc_id: str, raw: dict) -> Prediction:
        institution = raw.get("institution_id")
        if institution not in self._valid_ids:
            # The guarantee that must not depend on server cooperation.
            raise InvalidInstitution(
                f"{self.model} returned {institution!r}, which is not in the taxonomy"
            )
        parsed = LLMClassification.model_validate(raw)
        alternatives = [
            Alternative.model_validate(a.model_dump())
            for a in parsed.alternatives
            if a.institution_id in self._valid_ids
        ]
        return Prediction(
            doc_id=doc_id,
            institution_id=parsed.institution_id,
            confidence=parsed.confidence,
            rationale_ar=parsed.rationale_ar,
            alternatives=alternatives,
            needs_review=parsed.confidence < self.review_threshold,
            backend=self.name,
        )

    def _attempt(self, doc: Document, strategy: str) -> dict:
        """One round trip. Returns the raw object, validated but not yet built.

        Separated from `_to_prediction` so `classify` can record which
        strategy won *before* the Prediction is stamped with a backend name —
        otherwise the first prediction of a fallback run silently claims the
        stronger guarantee it did not get.
        """
        content = self.client.chat(
            model=self.model,
            messages=self._messages(doc, strategy),
            response_format=self._response_format(strategy),
            reasoning_effort=self.reasoning_effort,
        )
        raw = _extract_json(content)
        if raw.get("institution_id") not in self._valid_ids:
            raise InvalidInstitution(
                f"{self.model} returned {raw.get('institution_id')!r}, which is not in the taxonomy"
            )
        return raw

    def classify(self, doc: Document) -> Prediction:
        self._track(doc)

        if self.strategy:
            return self._to_prediction(doc.doc_id, self._attempt(doc, self.strategy))

        # First call negotiates: try the strongest guarantee, fall back only
        # when the server rejects it. Whatever succeeds is reused thereafter,
        # so this costs one extra round trip at most, once.
        errors: list[str] = []
        for strategy in STRATEGIES:
            try:
                raw = self._attempt(doc, strategy)
            except GatewayUnavailable:
                raise
            except (GatewayError, ValueError) as exc:
                errors.append(f"{strategy}: {exc}")
                continue
            self.strategy = strategy
            return self._to_prediction(doc.doc_id, raw)

        raise GatewayError(
            "no output strategy worked against this gateway:\n  " + "\n  ".join(errors)
        )

    def classify_many(self, docs: Sequence[Document]) -> list[Prediction]:
        return [self.classify(d) for d in docs]

    def preflight(self) -> dict:
        """Reachability, model presence, and which strategy the server supports."""
        report: dict = {"url": self.client.base_url, "model": self.model}
        try:
            models = self.client.list_models()
        except GatewayUnavailable as exc:
            report["reachable"] = False
            report["error"] = str(exc)
            return report
        except GatewayError as exc:
            report["reachable"] = True
            report["models_error"] = str(exc)
            models = []

        report["reachable"] = True
        report["available_models"] = models
        report["model_present"] = (not models) or self.model in models
        return report
