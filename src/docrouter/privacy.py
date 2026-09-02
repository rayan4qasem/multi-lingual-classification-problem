"""Masking identifiers before document text leaves the machine.

Every classification request sends a document to the API. For an archive of
citizen correspondence that is the largest single exposure in the system, so
identifiers are masked by default rather than on request.

The design point that makes this cheap: replacement is with a **typed
placeholder**, not a blank. `[رقم ضريبي]` still tells the model the document
contains a tax number — which is the part that carries routing signal toward
ZATCA — while removing the value that identifies a person. Dates and amounts
are never touched, because those genuinely inform the decision.

What this does not cover, and cannot: `ClaudeVisionOcr` uploads page *images*
to be transcribed. Redaction operates on text, so a scanned document is
exposed in full during OCR regardless of this module. Deployments that cannot
accept that must use `--ocr tesseract`, which keeps transcription local.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """One class of identifier, and what to put in its place."""

    name: str
    pattern: re.Pattern[str]
    placeholder: str


# Order matters: specific patterns must win before the generic digit-run rule,
# otherwise a tax number degrades to an untyped [رقم] and loses its signal.
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[بريد]"),
    Rule("iban", re.compile(r"\bSA\d{22}\b", re.I), "[آيبان]"),
    Rule("tax_number", re.compile(r"\b3\d{14}\b"), "[رقم ضريبي]"),
    Rule("phone", re.compile(r"(?:\+?966|00966|0)5\d{8}\b"), "[جوال]"),
    Rule("national_id", re.compile(r"\b[12]\d{9}\b"), "[رقم هوية]"),
    # Anything else long enough to identify a person or a case file.
    Rule("long_number", re.compile(r"\b\d{9,}\b"), "[رقم]"),
)


def redact(text: str, rules: tuple[Rule, ...] = DEFAULT_RULES) -> str:
    """Mask identifiers, leaving dates and amounts intact."""
    for rule in rules:
        text = rule.pattern.sub(rule.placeholder, text)
    return text


def scan(text: str, rules: tuple[Rule, ...] = DEFAULT_RULES) -> dict[str, int]:
    """Count identifiers per rule without modifying the text.

    Applied to progressively-masked text so the counts match what `redact`
    would actually replace, rather than double-counting a tax number as both
    a tax number and a long number.
    """
    counts: Counter[str] = Counter()
    working = text
    for rule in rules:
        found = rule.pattern.findall(working)
        if found:
            counts[rule.name] = len(found)
        working = rule.pattern.sub(rule.placeholder, working)
    return dict(counts)


def redact_with_counts(
    text: str, rules: tuple[Rule, ...] = DEFAULT_RULES
) -> tuple[str, dict[str, int]]:
    """Both at once, for callers that want to report what they masked."""
    return redact(text, rules), scan(text, rules)


def summarize(counts: dict[str, int]) -> str:
    """A one-line, human-readable tally for CLI output."""
    if not counts:
        return "no identifiers found"
    return ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
