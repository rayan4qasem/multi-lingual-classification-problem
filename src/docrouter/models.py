"""Shared data shapes.

`Prediction` is what every classifier backend returns, so the LLM classifier
and the scikit-learn baseline stay interchangeable behind one interface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["digital", "ocr", "mock", "plain"]


class Document(BaseModel):
    """A document after ingestion, before classification."""

    doc_id: str
    text: str
    source: Source = "plain"
    path: str | None = None
    page_count: int = 1
    # Present only for mock/eval data.
    true_label: str | None = None


class Alternative(BaseModel):
    institution_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    """The routing decision for a single document."""

    doc_id: str
    institution_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_ar: str = ""
    alternatives: list[Alternative] = Field(default_factory=list)
    # Set when confidence fell below the threshold and the document was
    # diverted to a human queue instead of being routed automatically.
    needs_review: bool = False
    backend: str = "unknown"


class LLMClassification(BaseModel):
    """Exactly what we ask Claude to return.

    Kept separate from `Prediction` because the model does not know the
    doc_id, does not decide the review threshold, and does not name itself.
    """

    institution_id: str = Field(
        description="معرّف الجهة الأنسب من القائمة المعطاة، بالإنجليزية كما هو"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="ثقة القرار من 0 إلى 1"
    )
    rationale_ar: str = Field(
        description="سبب مختصر بالعربية في جملة أو جملتين، مستنداً إلى نص الوثيقة"
    )
    alternatives: list[Alternative] = Field(
        default_factory=list,
        description="حتى جهتين بديلتين مرتبتين تنازلياً بالثقة، أو قائمة فارغة",
    )
