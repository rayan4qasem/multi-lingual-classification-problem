"""The label store.

Append-only JSONL. Corrections are appended, never edited in place, so the
file doubles as an audit trail — which matters when the labels become the
ground truth a government system is measured against.

Deliberately stores **no document text**. A record is a decision plus a
reference to the file it was made about. That keeps citizen data out of the
artifact most likely to get copied around, and it means the store can be
shared with a vendor or auditor without shipping the documents themselves.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["labeled", "skipped", "unclear"]
Lane = Literal["priority", "random"]


class LabelRecord(BaseModel):
    doc_id: str
    # The human's decision. Empty when status is not "labeled".
    label: str = ""
    status: Status = "labeled"
    lane: Lane = "priority"

    # What the model said, kept so agreement can be measured after the fact.
    model_label: str | None = None
    model_confidence: float | None = None
    model_backend: str | None = None

    reviewer: str = "unknown"
    reviewed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    seconds_spent: float | None = None
    notes: str = ""
    # Reference, not content.
    path: str | None = None

    @property
    def agreed(self) -> bool | None:
        if self.status != "labeled" or self.model_label is None:
            return None
        return self.label == self.model_label


class LaneStats(BaseModel):
    n: int = 0
    agreed: int = 0

    @property
    def agreement(self) -> float:
        return self.agreed / self.n if self.n else 0.0

    def wilson_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for the agreement rate.

        Reported instead of a bare percentage because early in a labeling
        run n is small, and "83%" off 12 documents deserves to be shown with
        the width it actually has.
        """
        n = self.n
        if not n:
            return (0.0, 0.0)
        p = self.agreed / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))


class StoreStats(BaseModel):
    total_records: int
    labeled: int
    skipped: int
    unclear: int
    per_class: dict[str, int]
    priority: LaneStats
    random: LaneStats
    reviewers: dict[str, int]
    median_seconds: float | None = None


class LabelStore:
    def __init__(self, path: str | Path = "data/labels/labels.jsonl"):
        self.path = Path(path)

    def append(self, record: LabelRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def all_records(self) -> list[LabelRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [LabelRecord.model_validate_json(line) for line in fh if line.strip()]

    def current(self) -> dict[str, LabelRecord]:
        """Latest decision per document — last write wins, so re-labeling works."""
        latest: dict[str, LabelRecord] = {}
        for record in self.all_records():
            latest[record.doc_id] = record
        return latest

    def labeled_ids(self) -> set[str]:
        """Every document already decided on, including skips.

        Skips count as decided so the queue does not keep re-serving a
        document a human already declined to judge.
        """
        return set(self.current().keys())

    def gold(self) -> dict[str, str]:
        """doc_id -> confirmed label, for documents actually labeled."""
        return {
            doc_id: r.label
            for doc_id, r in self.current().items()
            if r.status == "labeled" and r.label
        }

    def stats(self) -> StoreStats:
        current = list(self.current().values())
        lanes = {"priority": LaneStats(), "random": LaneStats()}
        per_class: Counter[str] = Counter()
        reviewers: Counter[str] = Counter()
        durations: list[float] = []

        for r in current:
            reviewers[r.reviewer] += 1
            if r.seconds_spent:
                durations.append(r.seconds_spent)
            if r.status != "labeled":
                continue
            per_class[r.label] += 1
            if r.model_label is not None:
                lane = lanes[r.lane]
                lane.n += 1
                lane.agreed += int(r.agreed or False)

        durations.sort()
        median = durations[len(durations) // 2] if durations else None

        return StoreStats(
            total_records=len(self.all_records()),
            labeled=sum(1 for r in current if r.status == "labeled"),
            skipped=sum(1 for r in current if r.status == "skipped"),
            unclear=sum(1 for r in current if r.status == "unclear"),
            per_class=dict(per_class),
            priority=lanes["priority"],
            random=lanes["random"],
            reviewers=dict(reviewers),
            median_seconds=median,
        )
