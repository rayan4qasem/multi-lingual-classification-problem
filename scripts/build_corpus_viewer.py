"""Generate a browsable viewer for the curated corpus.

    python scripts/build_corpus_viewer.py

Reads data/curated/corpus_part*.yaml and writes docs/corpus-viewer.html with
the documents embedded, so the page is self-contained and needs no server.
Regenerate it whenever the corpus changes.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import yaml

from docrouter.taxonomy import load

ROOT = pathlib.Path(__file__).resolve().parents[1]

HEAD = """<title>مجموعة وثائق الاختبار</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Noto+Naskh+Arabic:wght@400;700&family=IBM+Plex+Mono:wght@400;500&display=swap">

<style>
:root{
  --paper:#f8faf9; --card:#ffffff; --ink:#14201c; --muted:#5d6b66;
  --faint:#8b9691; --rule:#dbe4e0; --rule-strong:#c3d1cb;
  --accent:#1f5f4e; --accent-soft:#e6f0ec;
  --hard:#9a3412; --hard-soft:#f8ebe4;
  --medium:#8a6d1f; --medium-soft:#f7f1e0;
  --easy:#3f6b53; --easy-soft:#e8f1eb;
  --quote:#f2f6f4;
  --f-ui:"IBM Plex Sans Arabic","Segoe UI",Tahoma,sans-serif;
  --f-doc:"Noto Naskh Arabic","Times New Roman",serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0d1512; --card:#141d19; --ink:#e4ece8; --muted:#9aaaa4;
  --faint:#6f807a; --rule:#243029; --rule-strong:#33443c;
  --accent:#6fbfa5; --accent-soft:#16302a;
  --hard:#e08b63; --hard-soft:#31201a;
  --medium:#d4b264; --medium-soft:#2d2718;
  --easy:#7fbf9b; --easy-soft:#16281f;
  --quote:#101a17;
}}
:root[data-theme="dark"]{
  --paper:#0d1512; --card:#141d19; --ink:#e4ece8; --muted:#9aaaa4;
  --faint:#6f807a; --rule:#243029; --rule-strong:#33443c;
  --accent:#6fbfa5; --accent-soft:#16302a;
  --hard:#e08b63; --hard-soft:#31201a;
  --medium:#d4b264; --medium-soft:#2d2718;
  --easy:#7fbf9b; --easy-soft:#16281f;
  --quote:#101a17;
}
*{box-sizing:border-box}
html{direction:rtl}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--f-ui);font-size:15px;line-height:1.7;-webkit-font-smoothing:antialiased}
.wrap{max-width:1020px;margin:0 auto;padding:0 20px 80px}

header{border-bottom:2px solid var(--accent);padding:36px 0 20px;margin-bottom:22px}
.eyebrow{font-size:12px;letter-spacing:.14em;color:var(--accent);font-weight:600;margin:0 0 10px}
h1{font-size:clamp(24px,4vw,32px);margin:0 0 10px;font-weight:700;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0;max-width:62ch;font-size:15.5px}
.stats{display:flex;flex-wrap:wrap;gap:0;margin-top:22px;
  border:1px solid var(--rule);border-radius:5px;overflow:hidden;background:var(--card)}
.stat{flex:1 1 128px;padding:12px 16px;border-left:1px solid var(--rule)}
.stat:last-child{border-left:0}
.stat b{display:block;font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.3}
.stat span{font-size:12.5px;color:var(--muted)}

.controls{position:sticky;top:0;z-index:5;background:var(--paper);
  padding:14px 0;border-bottom:1px solid var(--rule);margin-bottom:20px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input[type=search]{flex:1 1 240px;font-family:var(--f-ui);font-size:14.5px;
  padding:9px 13px;border:1px solid var(--rule-strong);border-radius:4px;
  background:var(--card);color:var(--ink)}
input[type=search]:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
select{font-family:var(--f-ui);font-size:14px;padding:9px 11px;
  border:1px solid var(--rule-strong);border-radius:4px;background:var(--card);color:var(--ink)}
select:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.count{font-size:13px;color:var(--muted);margin-top:9px;font-variant-numeric:tabular-nums}

article{background:var(--card);border:1px solid var(--rule);border-radius:5px;
  padding:16px 18px;margin-bottom:12px}
article.hard{border-right:3px solid var(--hard)}
article.medium{border-right:3px solid var(--medium)}
article.easy{border-right:3px solid var(--easy)}
.top{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:9px}
.chip{font-size:11.5px;padding:2px 9px;border-radius:99px;white-space:nowrap}
.inst{background:var(--accent-soft);color:var(--accent);font-weight:600}
.d-hard{background:var(--hard-soft);color:var(--hard)}
.d-medium{background:var(--medium-soft);color:var(--medium)}
.d-easy{background:var(--easy-soft);color:var(--easy)}
.docid{font-family:var(--f-mono);font-size:11px;color:var(--faint);margin-inline-start:auto}
.subj{font-weight:600;font-size:15.5px;margin:0 0 4px}
.pair{font-size:12px;color:var(--muted);margin:0 0 10px}
.pair b{color:var(--hard);font-weight:600}
pre{font-family:var(--f-doc);font-size:15.5px;line-height:2;white-space:pre-wrap;
  word-break:break-word;margin:0;padding:14px 16px;background:var(--quote);border-radius:4px}
mark{background:var(--medium-soft);color:var(--ink);border-radius:2px;padding:0 2px}
.empty{text-align:center;padding:50px 20px;color:var(--muted)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">بيانات اختبار مُولّدة — ليست وثائق حقيقية</p>
  <h1>مجموعة وثائق الاختبار</h1>
  <p class="sub">وثائق عربية كُتبت لاختبار نظام فرز المراسلات الحكومية.
     الأسماء والأرقام والوقائع كلها متخيّلة بالكامل. بعضها حالات حدودية
     مصمّمة لتقع على الفاصل بين جهتين يكثر الخلط بينهما.</p>
  <div class="stats">__STATS__</div>
</header>

<div class="controls">
  <div class="row">
    <input type="search" id="q" placeholder="ابحث في نص الوثائق…">
    <select id="inst"><option value="">كل الجهات</option></select>
    <select id="diff">
      <option value="">كل المستويات</option>
      <option value="hard">صعبة</option>
      <option value="medium">متوسطة</option>
      <option value="easy">سهلة</option>
    </select>
    <select id="bnd">
      <option value="">الكل</option>
      <option value="1">الحالات الحدودية فقط</option>
    </select>
  </div>
  <p class="count" id="count"></p>
</div>

<div id="list"></div>
</div>

<script>
document.documentElement.lang = "ar";
document.documentElement.dir = "rtl";
const DOCS = __DATA__;
"""

TAIL = r"""
const DIFF = {hard:"صعبة", medium:"متوسطة", easy:"سهلة"};
const list = document.getElementById("list");
const q = document.getElementById("q");
const instSel = document.getElementById("inst");
const diffSel = document.getElementById("diff");
const bndSel = document.getElementById("bnd");
const count = document.getElementById("count");

const names = {};
DOCS.forEach(d => { names[d.label] = d.name; });
Object.entries(names)
  .sort((a, b) => a[1].localeCompare(b[1], "ar"))
  .forEach(([id, ar]) => {
    const n = DOCS.filter(d => d.label === id).length;
    instSel.insertAdjacentHTML("beforeend",
      '<option value="' + id + '">' + ar + " (" + n + ")</option>");
  });

function esc(s){
  return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
function highlight(text, term){
  const safe = esc(text);
  if (!term) return safe;
  try {
    const re = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g");
    return safe.replace(re, m => "<mark>" + m + "</mark>");
  } catch (e) { return safe; }
}

function render(){
  const term = q.value.trim();
  const inst = instSel.value, diff = diffSel.value, bnd = bndSel.value;
  const shown = DOCS.filter(d =>
    (!inst || d.label === inst) &&
    (!diff || d.difficulty === diff) &&
    (!bnd || d.pair) &&
    (!term || d.text.includes(term) || d.subject.includes(term) || d.name.includes(term))
  );

  count.textContent = shown.length === DOCS.length
    ? "عرض كل الوثائق (" + DOCS.length + ")"
    : "عرض " + shown.length + " من " + DOCS.length;

  if (!shown.length){
    list.innerHTML = '<p class="empty">لا توجد وثائق مطابقة.</p>';
    return;
  }
  list.innerHTML = shown.map(d => {
    let pairLine = "";
    if (d.pair){
      const parts = d.pair.split("|");
      const other = d.label === parts[0] ? parts[1] : parts[0];
      pairLine = '<p class="pair">حالة حدودية: المختص <b>' + esc(d.name) +
                 "</b> وليس " + esc(names[other] || other) + "</p>";
    }
    return '<article class="' + d.difficulty + '">' +
      '<div class="top">' +
        '<span class="chip inst">' + esc(d.name) + "</span>" +
        '<span class="chip d-' + d.difficulty + '">' + DIFF[d.difficulty] + "</span>" +
        '<span class="docid">' + esc(d.id) + "</span>" +
      "</div>" +
      '<p class="subj">' + highlight(d.subject, term) + "</p>" +
      pairLine +
      "<pre>" + highlight(d.text, term) + "</pre>" +
    "</article>";
  }).join("");
}

[q, instSel, diffSel, bndSel].forEach(el => el.addEventListener("input", render));
render();
</script>
"""


def main() -> None:
    tax = load()
    docs = []
    for f in sorted((ROOT / "data" / "curated").glob("corpus_part*.yaml")):
        payload = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for e in payload.get("documents", []):
            docs.append(
                {
                    "id": e["id"],
                    "label": e["label"],
                    "name": tax.name_ar(e["label"]),
                    "difficulty": e.get("difficulty", "medium"),
                    "subject": e["subject"],
                    "pair": e.get("pair"),
                    "text": e["text"].strip(),
                }
            )
    docs.sort(key=lambda d: (d["label"], d["id"]))

    diff = Counter(d["difficulty"] for d in docs)
    mean_len = sum(len(d["text"]) for d in docs) // max(len(docs), 1)
    stats = [
        (len(docs), "وثيقة"),
        (len({d["label"] for d in docs}), "جهة"),
        (diff["hard"], "صعبة"),
        (sum(1 for d in docs if d["pair"]), "حالة حدودية"),
        (mean_len, "متوسط الأحرف"),
    ]
    stats_html = "".join(
        f'<div class="stat"><b>{n}</b><span>{label}</span></div>' for n, label in stats
    )

    html = (
        HEAD.replace("__STATS__", stats_html).replace(
            "__DATA__", json.dumps(docs, ensure_ascii=False)
        )
        + TAIL
    )
    out = ROOT / "docs" / "corpus-viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} — {len(docs)} documents, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
