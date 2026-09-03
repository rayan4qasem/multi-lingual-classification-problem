"""Second-pass adjudication for documents that land on a confusion pair.

Every error the baseline makes on the adversarial subset falls inside a pair
already declared in `config/taxonomy.yaml` — labour vs social insurance, tax
vs commerce, police vs prosecution. That is not a coincidence: those cases
were authored to carry the surface vocabulary of one institution while the
competent authority is the other, so a model that routes on topic loses them
and a model that routes on the **requested action** wins them.

So instead of asking a fourteen-way question and hoping, this asks the
fourteen-way question first and then, only when the answer lands on a
declared pair, asks a much easier two-way question with both institutions'
disambiguation rules in front of the model.

Structure is a decorator, not a subclass: `TiebreakClassifier` wraps any
`Classifier` and is itself a `Classifier`, so it composes with the gateway
backend, the Claude backend or the offline baseline without any of them
knowing it exists. What to do about a pair is delegated to a `PairResolver`,
which keeps the "when to adjudicate" policy separate from the "how" — the
former is taxonomy logic and testable offline, the latter needs a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..models import Document, Prediction
from ..taxonomy import Taxonomy
from ..taxonomy import load as load_taxonomy

# Asked in Arabic, because the documents are Arabic and the disambiguation
# rules in the taxonomy are written in Arabic. Translating the question into
# English to ask about Arabic text throws away the thing being reasoned over.
TIEBREAK_PROMPT = """أمامك وثيقة رسمية، والجهة المختصة بها إما (أ) أو (ب) — لا ثالث لهما.

(أ) {a_name} — {a_id}
{a_desc}
قاعدة التمييز: {a_rule}

(ب) {b_name} — {b_id}
{b_desc}
قاعدة التمييز: {b_rule}

الوثيقة:
---
{text}
---

لا تحكم بالموضوع العام ولا بالمفردات الواردة، فكلتا الجهتين تشتركان فيها.
احكم بـ**الإجراء المطلوب** تحديداً: ما الذي يطلبه مقدم الوثيقة فعلاً، وأي
الجهتين تملك صلاحية اتخاذ ذلك الإجراء؟ وإن نفى مقدم الوثيقة صراحةً طلباً ما،
فذلك النفي يستبعد الجهة المرتبطة به ولا يرجّحها.

أجب بـ JSON فقط:
{{"institution_id": "<{a_id} أو {b_id}>", "confidence": <0.0-1.0>, "rationale_ar": "<الإجراء المطلوب ولماذا يقع في اختصاص هذه الجهة>"}}"""


@runtime_checkable
class PairResolver(Protocol):
    """Decides between exactly two institutions for one document."""

    def resolve(self, doc: Document, a: str, b: str) -> tuple[str, float, str]:
        """Return (institution_id, confidence, rationale_ar)."""


class PairIndex:
    """The declared confusion pairs, as an order-insensitive lookup."""

    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or load_taxonomy()
        self._pairs = {frozenset(pair) for pair in self.taxonomy.confusion_pairs if len(pair) == 2}

    def __len__(self) -> int:
        return len(self._pairs)

    def contains(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self._pairs

    def partner(self, prediction: Prediction) -> str | None:
        """The runner-up, if the top two form a declared pair.

        Alternatives are searched in order rather than only at position zero:
        a model that offers three alternatives may rank the genuine rival
        second, and skipping it would leave exactly the cases this exists for
        unadjudicated.
        """
        for alt in prediction.alternatives:
            if alt.institution_id != prediction.institution_id and self.contains(
                prediction.institution_id, alt.institution_id
            ):
                return alt.institution_id
        return None


class LLMPairResolver:
    """Asks a gateway model the two-way question."""

    def __init__(self, client, model: str, taxonomy: Taxonomy | None = None) -> None:
        self.client = client
        self.model = model
        self.taxonomy = taxonomy or load_taxonomy()

    def _describe(self, institution_id: str) -> dict[str, str]:
        inst = self.taxonomy.get(institution_id)
        if inst is None:  # pragma: no cover - guarded by validate_refs
            raise ValueError(f"unknown institution {institution_id!r}")
        return {
            "id": inst.id,
            "name": inst.name_ar,
            "desc": inst.description_ar,
            "rule": inst.disambiguation_ar or "—",
        }

    def prompt_for(self, doc: Document, a: str, b: str) -> str:
        da, db = self._describe(a), self._describe(b)
        return TIEBREAK_PROMPT.format(
            a_id=da["id"],
            a_name=da["name"],
            a_desc=da["desc"],
            a_rule=da["rule"],
            b_id=db["id"],
            b_name=db["name"],
            b_desc=db["desc"],
            b_rule=db["rule"],
            text=doc.text,
        )

    def resolve(self, doc: Document, a: str, b: str) -> tuple[str, float, str]:
        from .openai_compat import _extract_json

        raw = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt_for(doc, a, b)}],
            response_format={"type": "json_object"},
        )
        data = _extract_json(raw)
        chosen = data.get("institution_id")
        # The whole point is a binary choice; anything else is not an answer,
        # and silently accepting it would let the tiebreak route a document
        # somewhere neither pass proposed.
        if chosen not in (a, b):
            raise ValueError(f"tiebreak returned {chosen!r}, expected {a!r} or {b!r}")
        confidence = float(data.get("confidence", 0.0))
        return chosen, max(0.0, min(1.0, confidence)), str(data.get("rationale_ar", ""))


class TiebreakClassifier:
    """Wraps a classifier, adjudicating only documents that need it."""

    def __init__(
        self,
        base,
        resolver: PairResolver,
        taxonomy: Taxonomy | None = None,
        max_confidence: float = 0.90,
    ) -> None:
        self.base = base
        self.resolver = resolver
        self.index = PairIndex(taxonomy)
        # Above this the first pass is not in genuine doubt, and a second
        # opinion costs a request to change nothing. Set to 1.0 to adjudicate
        # every pair regardless of confidence.
        self.max_confidence = max_confidence
        self.adjudicated = 0
        self.changed = 0

    @property
    def name(self) -> str:
        return f"tiebreak({self.base.name})"

    def _needs_tiebreak(self, prediction: Prediction) -> str | None:
        if prediction.confidence > self.max_confidence:
            return None
        return self.index.partner(prediction)

    def classify(self, doc: Document) -> Prediction:
        first = self.base.classify(doc)
        partner = self._needs_tiebreak(first)
        if partner is None:
            return first

        self.adjudicated += 1
        try:
            chosen, confidence, rationale = self.resolver.resolve(
                doc, first.institution_id, partner
            )
        except Exception:
            # A failed second opinion must never lose the first one. The
            # document still routes exactly as it would have without this
            # layer, which is what makes the decorator safe to enable.
            return first

        if chosen != first.institution_id:
            self.changed += 1
        return first.model_copy(
            update={
                "institution_id": chosen,
                "confidence": confidence,
                "rationale_ar": rationale or first.rationale_ar,
                "backend": f"tiebreak({first.backend})",
            }
        )

    def classify_many(self, docs: Sequence[Document]) -> list[Prediction]:
        return [self.classify(d) for d in docs]
