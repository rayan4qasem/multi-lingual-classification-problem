"""Human-in-the-loop labeling for real documents.

The loop:

    ingest  ->  pre-label  ->  prioritize  ->  human adjudicates  ->  store
                   ^                                                    |
                   +------------- retrain / re-prompt -----------------+

`store` holds decisions (never document text). `prioritize` decides what is
worth a human's attention. `review` serves the local adjudication UI.
"""

from .store import LabelRecord, LabelStore
from .prioritize import QueueItem, build_queue

__all__ = ["LabelRecord", "LabelStore", "QueueItem", "build_queue"]
