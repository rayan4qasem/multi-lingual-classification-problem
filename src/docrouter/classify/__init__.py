"""Classifier backends. All of them satisfy `protocols.Classifier`."""

from .baseline import BaselineClassifier
from .factory import (
    REGISTRY,
    ClassifierRegistry,
    MissingModel,
    UnknownBackend,
    available_backends,
    create_classifier,
)
from .llm import LLMClassifier
from .openai_compat import (
    GatewayError,
    GatewayUnavailable,
    InvalidInstitution,
    OpenAICompatClassifier,
)

__all__ = [
    "REGISTRY",
    "BaselineClassifier",
    "ClassifierRegistry",
    "GatewayError",
    "GatewayUnavailable",
    "InvalidInstitution",
    "LLMClassifier",
    "MissingModel",
    "OpenAICompatClassifier",
    "UnknownBackend",
    "available_backends",
    "create_classifier",
]
