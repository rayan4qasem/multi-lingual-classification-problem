"""Structural contracts for the swappable parts of the pipeline.

`typing.Protocol` rather than ABCs on purpose: implementations do not have to
import or inherit from anything here, so a caller can drop in their own OCR
backend or classifier without taking a dependency on this package's class
hierarchy. Dependency inversion without the inheritance tax.

The classifier contract is split in two deliberately. `Classifier` is the
narrow interface every backend must satisfy; `BatchClassifier` adds the
Batches API surface. The offline baseline has no notion of an asynchronous
batch, and forcing it to grow stub methods to satisfy one fat interface is
exactly the interface-segregation problem — so it simply does not implement
the second protocol, and callers that need batching ask for that type.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import Document, Prediction, Source


@runtime_checkable
class Classifier(Protocol):
    """Anything that can route documents to institutions."""

    @property
    def name(self) -> str:
        """Stable identifier recorded on every Prediction, e.g. 'llm:claude-opus-5'."""

    def classify(self, doc: Document) -> Prediction: ...

    def classify_many(self, docs: Sequence[Document]) -> list[Prediction]: ...


@runtime_checkable
class BatchClassifier(Classifier, Protocol):
    """A classifier that can also route asynchronously, in bulk."""

    def submit_batch(self, docs: Iterable[Document]) -> str: ...

    def batch_status(self, batch_id: str) -> Any: ...

    def collect_batch(self, batch_id: str) -> tuple[list[Prediction], dict[str, str]]: ...


@runtime_checkable
class OcrBackend(Protocol):
    """Turns page images into text."""

    @property
    def name(self) -> str: ...

    def transcribe(self, images: Sequence[bytes]) -> str: ...


class Extraction(Protocol):
    """The result of pulling text out of one file."""

    text: str
    source: Source
    page_count: int


@runtime_checkable
class TextExtractor(Protocol):
    """Handles one family of file types.

    Adding a format means writing an extractor and registering it — never
    editing a dispatch table in `ingest`, which is the open/closed point.
    """

    @property
    def suffixes(self) -> frozenset[str]:
        """Lower-case extensions this extractor claims, including the dot."""

    def extract(self, path: Path, ocr: OcrBackend | None = None) -> Any: ...
