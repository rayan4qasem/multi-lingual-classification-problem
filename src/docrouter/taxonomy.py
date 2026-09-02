"""Loading and rendering the institution taxonomy."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "taxonomy.yaml"


class Institution(BaseModel):
    id: str
    name_ar: str
    name_en: str
    description_ar: str
    keywords_ar: list[str] = Field(default_factory=list)
    document_types_ar: list[str] = Field(default_factory=list)
    disambiguation_ar: str = ""


class Taxonomy(BaseModel):
    version: int
    language: str = "ar"
    fallback_id: str = "manual_review"
    institutions: list[Institution]
    confusion_pairs: list[list[str]] = Field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [i.id for i in self.institutions]

    def get(self, institution_id: str) -> Institution | None:
        return next((i for i in self.institutions if i.id == institution_id), None)

    def name_ar(self, institution_id: str) -> str:
        if institution_id == self.fallback_id:
            return "مراجعة يدوية"
        inst = self.get(institution_id)
        return inst.name_ar if inst else institution_id

    def validate_refs(self) -> None:
        """Fail loudly on typos in config rather than silently at inference."""
        known = set(self.ids)
        duplicates = {i for i in self.ids if self.ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate institution ids in taxonomy: {sorted(duplicates)}")
        if self.fallback_id in known:
            raise ValueError(
                f"fallback_id {self.fallback_id!r} must not also be a real institution"
            )
        for pair in self.confusion_pairs:
            unknown = [p for p in pair if p not in known]
            if unknown:
                raise ValueError(f"confusion_pairs references unknown ids: {unknown}")

    def render_for_prompt(self) -> str:
        """The institution catalogue as it appears in the system prompt.

        Deterministic ordering and no timestamps — this block is the cached
        prefix of every classification request, so it must be byte-stable.
        """
        blocks: list[str] = []
        for inst in self.institutions:
            lines = [
                f"### {inst.id}",
                f"الاسم: {inst.name_ar}",
                f"الاختصاص: {inst.description_ar.strip()}",
            ]
            if inst.document_types_ar:
                lines.append("أمثلة على الوثائق: " + "، ".join(inst.document_types_ar))
            if inst.keywords_ar:
                lines.append("ألفاظ دالة: " + "، ".join(inst.keywords_ar))
            if inst.disambiguation_ar:
                lines.append(f"تمييز: {inst.disambiguation_ar.strip()}")
            blocks.append("\n".join(lines))

        catalogue = "\n\n".join(blocks)

        if self.confusion_pairs:
            pairs = "\n".join(
                f"- {self.name_ar(a)} ({a}) مقابل {self.name_ar(b)} ({b})"
                for a, b in self.confusion_pairs
            )
            catalogue += (
                "\n\n## أزواج يكثر الخلط بينها\n"
                "راجع فقرة «تمييز» لدى كل جهة قبل الحسم في هذه الحالات:\n" + pairs
            )
        return catalogue


@lru_cache(maxsize=4)
def load(path: str | Path | None = None) -> Taxonomy:
    """Load and validate the taxonomy. Cached — it is read on every request."""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"taxonomy not found at {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    tax = Taxonomy.model_validate(data)
    tax.validate_refs()
    return tax
