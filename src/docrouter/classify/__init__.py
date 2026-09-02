"""Classifier backends. All of them return `Prediction`."""

from .baseline import BaselineClassifier
from .llm import LLMClassifier

__all__ = ["LLMClassifier", "BaselineClassifier"]
