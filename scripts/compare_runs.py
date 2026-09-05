"""Side-by-side report over the scoring runs in runs/."""

import json
from collections import Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def load(tag):
    p = RUNS / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def acc(rows):
    return sum(r["pred"] == r["true"] for r in rows) / len(rows) if rows else 0.0


print(f"{'run':<28}{'n':>5}{'curated':>10}{'n_hard':>8}{'hard':>9}{'errors':>8}")
print("-" * 68)
for tag in ("openai_gpt-oss-20b", "120b_full", "120b_compact", "120b_tiebreak"):
    d = load(tag)
    if not d:
        continue
    ok = [v for v in d.values() if not v.get("error")]
    hard = [v for v in ok if v["hard"]]
    print(
        f"{tag:<28}{len(ok):>5}{acc(ok):>9.1%}{len(hard):>8}{acc(hard):>9.1%}"
        f"{len(d) - len(ok):>8}"
    )

for tag in ("120b_full", "120b_compact"):
    d = load(tag)
    if not d:
        continue
    ok = [v for v in d.values() if not v.get("error")]
    errs = Counter(f"{r['true']} -> {r['pred']}" for r in ok if r["pred"] != r["true"])
    print(f"\n{tag} misroutes ({sum(errs.values())}):")
    for k, v in errs.most_common(12):
        print(f"  {v:>2}  {k}")
