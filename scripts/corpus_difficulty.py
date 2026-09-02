"""Compare how hard each mock corpus is, using the offline baseline.

The point is not the baseline's score in itself. It is that a keyword-ish
model should do very well on the template corpus (its bodies come from
per-institution pools) and much worse on the curated one. If the two scores
are close, the curated corpus is not pulling its weight as a benchmark.

    python scripts/corpus_difficulty.py
"""

from __future__ import annotations

import random

from docrouter import mockdata
from docrouter.classify import BaselineClassifier
from docrouter.evaluate import evaluate
from docrouter.taxonomy import load


def split(docs, ratio=0.25, seed=7):
    docs = list(docs)
    random.Random(seed).shuffle(docs)
    cut = int(len(docs) * (1 - ratio))
    return docs[:cut], docs[cut:]


def main() -> None:
    tax = load()
    templates = mockdata.generate_templates(n_per_class=20, seed=7)
    curated = mockdata.generate_curated(seed=7)

    print(f"template corpus : {len(templates)} docs")
    print(f"curated corpus  : {len(curated)} docs")
    print()

    # 1. Templates, held out from themselves — the optimistic number.
    train, test = split(templates)
    clf = BaselineClassifier()
    clf.fit(train)
    in_domain = evaluate(test, clf.classify_many(test), taxonomy=tax)

    # 2. Same model, applied to the curated corpus. Same 14 labels, real
    #    prose instead of pooled phrases.
    transfer = evaluate(curated, clf.classify_many(curated), taxonomy=tax)

    # 3. The boundary cases only — the adversarial subset.
    hard = mockdata.generate_curated(seed=7, hard_only=True)
    hard_report = evaluate(hard, clf.classify_many(hard), taxonomy=tax)

    rows = [
        ("templates -> templates (held out)", in_domain, len(test)),
        ("templates -> curated (all)", transfer, len(curated)),
        ("templates -> curated (hard only)", hard_report, len(hard)),
    ]
    print(f"{'evaluation':<38}{'n':>5}{'acc':>9}{'macroF1':>10}")
    print("-" * 62)
    for name, report, n in rows:
        print(f"{name:<38}{n:>5}{report.accuracy:>9.1%}{report.macro_f1:>10.3f}")

    print()
    print("Worst curated classes for the baseline:")
    for m in sorted(transfer.per_class, key=lambda m: m.f1)[:5]:
        if m.support:
            print(f"  {m.institution_id:<26} n={m.support:<3} F1={m.f1:.2f}")

    print()
    print("Errors on declared confusion pairs (curated):")
    for pair, count in sorted(transfer.confusion_pair_errors.items(), key=lambda kv: -kv[1]):
        if count:
            print(f"  {pair}: {count}")


if __name__ == "__main__":
    main()
