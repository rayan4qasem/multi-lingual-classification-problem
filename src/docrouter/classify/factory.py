"""Constructing classifiers by name.

The CLI used to branch on the backend string and build each classifier
inline, which meant a new backend touched the CLI, the help text and the
error handling. Here backends register themselves against a name, and the
CLI asks the registry — so adding one (a fine-tuned AraBERT, say, for an
on-prem deployment) is a registration, not an edit.

Each factory takes the same keyword arguments and ignores what it does not
need, so callers do not have to know which backend they are asking for.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..protocols import Classifier
from ..taxonomy import Taxonomy
from .baseline import BaselineClassifier
from .llm import LLMClassifier
from .openai_compat import OpenAICompatClassifier

DEFAULT_BASELINE_PATH = Path("models") / "baseline.joblib"


class UnknownBackend(ValueError):
    """Raised when no classifier is registered under the requested name."""


class MissingModel(FileNotFoundError):
    """Raised when a backend needs a trained artifact that is not on disk."""


ClassifierFactory = Callable[..., Classifier]


class ClassifierRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ClassifierFactory] = {}

    def register(self, name: str, factory: ClassifierFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> Classifier:
        try:
            factory = self._factories[name]
        except KeyError:
            raise UnknownBackend(
                f"unknown backend {name!r}; registered: {sorted(self._factories)}"
            ) from None
        return factory(**kwargs)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._factories)


def _build_llm(
    taxonomy: Taxonomy | None = None,
    model: str | None = None,
    effort: str | None = None,
    review_threshold: float = 0.55,
    client=None,
    examples=None,
    redact_pii: bool = True,
    **_ignored,
) -> Classifier:
    return LLMClassifier(
        taxonomy=taxonomy,
        model=model,
        effort=effort,
        review_threshold=review_threshold,
        client=client,
        examples=examples,
        redact_pii=redact_pii,
    )


def _build_baseline(
    taxonomy: Taxonomy | None = None,
    review_threshold: float = 0.55,
    model_path: str | Path | None = None,
    **_ignored,
) -> Classifier:
    path = Path(model_path) if model_path else DEFAULT_BASELINE_PATH
    if not path.exists():
        raise MissingModel(f"no trained baseline at {path} — run `docrouter train-baseline` first")
    return BaselineClassifier(taxonomy=taxonomy, review_threshold=review_threshold).load(path)


def _build_local(
    taxonomy: Taxonomy | None = None,
    review_threshold: float = 0.55,
    examples=None,
    redact_pii: bool = True,
    local_model: str | None = None,
    base_url: str | None = None,
    **_ignored,
) -> Classifier:
    return OpenAICompatClassifier(
        taxonomy=taxonomy,
        model=local_model,
        review_threshold=review_threshold,
        examples=examples,
        redact_pii=redact_pii,
        base_url=base_url,
    )


REGISTRY = ClassifierRegistry()
REGISTRY.register("local", _build_local)
REGISTRY.register("llm", _build_llm)
REGISTRY.register("baseline", _build_baseline)


def create_classifier(backend: str, **kwargs) -> Classifier:
    """Build a classifier by registered name."""
    return REGISTRY.create(backend, **kwargs)


def available_backends() -> list[str]:
    return sorted(REGISTRY.names)
