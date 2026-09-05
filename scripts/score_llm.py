"""Score a gateway model over the curated corpus.

Rate-limited for free tiers and check-pointed to JSON, so an interrupted run
resumes instead of re-spending quota. Results land in `runs/<model>.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from docrouter import taxonomy as T
from docrouter.classify.openai_compat import OpenAICompatClassifier
from docrouter.mockdata import generate_curated

# `cli.py` loads this at import; a standalone script has to ask for itself.
load_dotenv()

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rpm", type=int, default=28, help="requests per minute cap")
    ap.add_argument("--tag", default=None, help="output name; defaults to the model")
    ap.add_argument("--detail", default="full", choices=["full", "compact"])
    ap.add_argument("--reasoning-effort", default=None, dest="effort")
    ap.add_argument("--tiebreak", action="store_true")
    args = ap.parse_args()

    tax = T.load("config/taxonomy.yaml")
    docs = generate_curated(tax)
    hard = {d.doc_id for d in generate_curated(tax, hard_only=True)}

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{(args.tag or args.model).replace('/', '_')}.json"
    done: dict[str, dict] = {}
    if out.exists():
        done = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(done)} already scored")

    clf = OpenAICompatClassifier(
        model=args.model,
        taxonomy=tax,
        detail=args.detail,
        reasoning_effort=args.effort,
    )
    if args.tiebreak:
        from docrouter.classify.tiebreak import LLMPairResolver, TiebreakClassifier

        clf = TiebreakClassifier(
            clf, LLMPairResolver(clf.client, args.model, tax), tax
        )
    gap = 60.0 / args.rpm

    for i, d in enumerate(docs, 1):
        if d.doc_id in done:
            continue
        t0 = time.monotonic()
        # Free tiers cap tokens per minute, and this prompt carries the whole
        # taxonomy. A 429 is expected traffic here, not an error — back off
        # and retry rather than burning the document.
        for attempt in range(6):
            try:
                p = clf.classify(d)
                done[d.doc_id] = {
                    "true": d.true_label,
                    "pred": p.institution_id,
                    "conf": p.confidence,
                    "hard": d.doc_id in hard,
                }
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 5:
                    wait = 20 * (attempt + 1)
                    print(f"  429, waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                done[d.doc_id] = {
                    "true": d.true_label,
                    "pred": None,
                    "conf": 0.0,
                    "hard": d.doc_id in hard,
                    "error": str(exc)[:200],
                }
                break
        out.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        if i % 10 == 0:
            print(f"  {i}/{len(docs)}", flush=True)
        time.sleep(max(0.0, gap - (time.monotonic() - t0)))

    report(done, args.model)


def report(done: dict[str, dict], model: str) -> None:
    rows = list(done.values())
    hard = [r for r in rows if r["hard"]]
    errs = [r for r in rows if r.get("error")]

    def acc(rs: list[dict]) -> float:
        return sum(r["pred"] == r["true"] for r in rs) / len(rs) if rs else 0.0

    print(f"\n=== {model} ===")
    print(f"curated  n={len(rows):<4} acc={acc(rows):.1%}")
    print(f"hard     n={len(hard):<4} acc={acc(hard):.1%}")
    if errs:
        print(f"errors   n={len(errs)}  e.g. {errs[0].get('error')}")
    conf = Counter(f"{r['true']} -> {r['pred']}" for r in rows if r["pred"] != r["true"])
    print("\nmisroutes:")
    for k, v in conf.most_common(15):
        print(f"  {v:>2}  {k}")


if __name__ == "__main__":
    main()
