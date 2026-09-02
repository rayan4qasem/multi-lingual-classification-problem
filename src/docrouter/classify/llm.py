"""Claude-backed classifier.

Two paths share one prompt:

`classify` / `classify_many` are synchronous, for interactive routing and
small runs. `submit_batch` / `collect_batch` use the Batches API, which is
half the price and the right tool for a nightly archive sweep.

The system prompt is built once per taxonomy version and cached server-side,
so the per-document cost is roughly the document itself.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .. import DEFAULT_MODEL
from ..fewshot import ExampleSet
from ..fewshot import render as render_examples
from ..models import Alternative, Document, LLMClassification, Prediction
from ..privacy import redact, scan
from ..taxonomy import Taxonomy
from ..taxonomy import load as load_taxonomy

log = logging.getLogger(__name__)

# Documents longer than this are sent as head + tail with an explicit elision
# marker. Set generously — a routing decision rarely needs the middle of a
# 90-page attachment, and the marker keeps the elision visible to the model.
MAX_DOC_CHARS = 24_000
HEAD_CHARS = 16_000
TAIL_CHARS = 4_000

SYSTEM_PREAMBLE = """\
أنت نظام فرز وثائق حكومية في المملكة العربية السعودية. مهمتك تحديد الجهة \
الحكومية المختصة بمعالجة الوثيقة المعروضة عليك، وإرجاع معرّفها.

قواعد الحسم:
1. احكم بناءً على الإجراء المطلوب في الوثيقة، لا على الألفاظ المتناثرة فيها. \
ذكر «مستشفى» في وثيقة موضوعها مطالبة بأجور لا يجعلها لوزارة الصحة.
2. إن ذُكرت جهة صراحةً كمرسل إليه فذلك مرجّح قوي، لكنه ليس قاطعاً إن خالف موضوع الوثيقة.
3. راجع فقرة «تمييز» لدى الجهات المتقاربة قبل الحسم.
4. اختر جهة واحدة فقط هي الأقرب. إن تعذّر الترجيح بين جهتين فاخفض الثقة \
واذكر الأخرى في البدائل.
5. الوثائق الناقصة أو غير المقروءة أو التي لا تخص أي جهة في القائمة: اختر أقرب \
جهة ممكنة مع ثقة منخفضة (أقل من 0.4)، وسيتولى النظام تحويلها للمراجعة اليدوية.
6. اضبط الثقة بصدق: 0.9 فأعلى لوثيقة واضحة لا لبس فيها، و0.5 إلى 0.7 لوثيقة \
محتملة، وأقل من 0.4 لتخمين.
7. معرّف الجهة يُكتب بالإنجليزية حرفياً كما ورد في القائمة أدناه، ولا تخترع معرّفاً جديداً.

## قائمة الجهات
"""


# Hand-written rather than derived from the Pydantic model so that
# `additionalProperties: false` and the enum of valid ids are guaranteed.
def build_schema(taxonomy: Taxonomy) -> dict:
    return {
        "type": "object",
        "properties": {
            "institution_id": {"type": "string", "enum": taxonomy.ids},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale_ar": {"type": "string"},
            "alternatives": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "institution_id": {"type": "string", "enum": taxonomy.ids},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["institution_id", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["institution_id", "confidence", "rationale_ar", "alternatives"],
        "additionalProperties": False,
    }


def _prepare_text(doc: Document, redact_pii: bool = True) -> str:
    text = doc.text.strip()
    if not text:
        return "[الوثيقة فارغة أو تعذّرت قراءتها]"
    if redact_pii:
        # Before truncation, so identifiers cannot survive in the tail.
        text = redact(text)
    if len(text) <= MAX_DOC_CHARS:
        return text
    log.warning(
        "doc %s is %d chars; sending head+tail with an elision marker",
        doc.doc_id,
        len(text),
    )
    return (
        text[:HEAD_CHARS]
        + f"\n\n[... حُذف {len(text) - HEAD_CHARS - TAIL_CHARS} حرفاً من وسط الوثيقة ...]\n\n"
        + text[-TAIL_CHARS:]
    )


def _user_content(doc: Document, redact_pii: bool = True) -> str:
    origin = "نص رقمي" if doc.source == "digital" else "نص مستخرج ضوئياً (قد يحتوي أخطاء)"
    return (
        f"مصدر النص: {origin}\n"
        f"عدد الصفحات: {doc.page_count}\n"
        "--- بداية الوثيقة ---\n"
        f"{_prepare_text(doc, redact_pii=redact_pii)}\n"
        "--- نهاية الوثيقة ---"
    )


class LLMClassifier:
    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        model: str | None = None,
        effort: str | None = None,
        review_threshold: float = 0.55,
        client: anthropic.Anthropic | None = None,
        examples: ExampleSet | None = None,
        redact_pii: bool = True,
    ):
        self.taxonomy = taxonomy or load_taxonomy()
        self.model = model or os.environ.get("DOCROUTER_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("DOCROUTER_EFFORT", "medium")
        self.review_threshold = review_threshold
        self.client = client or anthropic.Anthropic()
        self.schema = build_schema(self.taxonomy)
        self.examples = examples
        # On by default: every document goes to the API, so this is the
        # system's largest exposure. Typed placeholders keep the routing
        # signal, so the safe default is also the cheap one.
        self.redact_pii = redact_pii
        # Tally of what has been masked, for reporting after a run.
        self.redaction_counts: dict[str, int] = {}

        # Examples join the cached prefix rather than the per-document turn.
        # They are rendered in sorted order and carry no timestamps, so the
        # prefix stays byte-stable and keeps hitting the cache.
        block = render_examples(examples, self.taxonomy) if examples else ""
        self._system = SYSTEM_PREAMBLE + self.taxonomy.render_for_prompt() + block

    @property
    def name(self) -> str:
        n = len(self.examples.examples) if self.examples else 0
        base = f"llm:{self.model}+{n}shot" if n else f"llm:{self.model}"
        # Recorded on every Prediction, so a run made without redaction is
        # identifiable after the fact rather than indistinguishable.
        return base if self.redact_pii else base + "+raw"

    def _system_blocks(self) -> list[dict]:
        # One block, one breakpoint: the whole prompt is stable across
        # documents, so everything before the user turn is cacheable.
        return [
            {
                "type": "text",
                "text": self._system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _request_kwargs(self) -> dict:
        return {
            "model": self.model,
            "max_tokens": 2000,
            "system": self._system_blocks(),
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self.schema},
            },
        }

    def _to_prediction(self, doc_id: str, raw: dict) -> Prediction:
        parsed = LLMClassification.model_validate(raw)
        return Prediction(
            doc_id=doc_id,
            institution_id=parsed.institution_id,
            confidence=parsed.confidence,
            rationale_ar=parsed.rationale_ar,
            alternatives=[Alternative.model_validate(a.model_dump()) for a in parsed.alternatives],
            needs_review=parsed.confidence < self.review_threshold,
            backend=self.name,
        )

    def _track(self, doc: Document) -> None:
        if not self.redact_pii:
            return
        for name, count in scan(doc.text).items():
            self.redaction_counts[name] = self.redaction_counts.get(name, 0) + count

    def classify(self, doc: Document) -> Prediction:
        self._track(doc)
        response = self.client.messages.create(
            **self._request_kwargs(),
            messages=[{"role": "user", "content": _user_content(doc, self.redact_pii)}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"classification refused for {doc.doc_id}: "
                f"{getattr(response.stop_details, 'category', None)}"
            )
        text = next(b.text for b in response.content if b.type == "text")
        return self._to_prediction(doc.doc_id, json.loads(text))

    def classify_many(self, docs: Iterable[Document]) -> list[Prediction]:
        """Sequential. For more than a few dozen documents use the batch path."""
        return [self.classify(d) for d in docs]

    # ---- Batches API: half price, results within the hour ----

    def submit_batch(self, docs: Iterable[Document]) -> str:
        docs = list(docs)
        if not docs:
            raise ValueError("no documents to submit")
        for doc in docs:
            self._track(doc)
        kwargs = self._request_kwargs()
        batch = self.client.messages.batches.create(
            requests=[
                Request(
                    custom_id=doc.doc_id,
                    # model and max_tokens arrive via **kwargs from
                    # _request_kwargs(); mypy cannot see through the spread
                    # into the TypedDict's required keys.
                    params=MessageCreateParamsNonStreaming(  # type: ignore[typeddict-item]
                        **kwargs,
                        messages=[{"role": "user", "content": _user_content(doc, self.redact_pii)}],
                    ),
                )
                for doc in docs
            ]
        )
        return batch.id

    def batch_status(self, batch_id: str):
        return self.client.messages.batches.retrieve(batch_id)

    def collect_batch(self, batch_id: str) -> tuple[list[Prediction], dict[str, str]]:
        """Return (predictions, errors keyed by doc_id).

        Results arrive in arbitrary order, so everything is keyed by custom_id.
        """
        predictions: list[Prediction] = []
        errors: dict[str, str] = {}

        for result in self.client.messages.batches.results(batch_id):
            doc_id = result.custom_id
            if result.result.type != "succeeded":
                errors[doc_id] = result.result.type
                continue
            message = result.result.message
            if message.stop_reason == "refusal":
                errors[doc_id] = "refusal"
                continue
            try:
                text = next(b.text for b in message.content if b.type == "text")
                predictions.append(self._to_prediction(doc_id, json.loads(text)))
            except (StopIteration, json.JSONDecodeError, ValueError) as exc:
                errors[doc_id] = f"unparseable: {exc}"

        return predictions, errors
