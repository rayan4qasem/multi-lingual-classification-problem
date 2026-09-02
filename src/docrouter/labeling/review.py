"""Local review UI.

A stdlib HTTP server bound to the loopback interface. No framework, no CDN,
no outbound request: real documents stay on the reviewer's machine, which is
the whole reason this is not a published web app.

Anchoring is treated as a real threat to label quality. Documents in the
random lane — the lane the honest accuracy estimate comes from — are served
**blind**: the model's prediction is stripped from the payload server-side,
so it is not merely hidden in the DOM. The server holds its own copy and
computes agreement when the decision comes back. A reviewer who never saw
the suggestion cannot have been nudged by it.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..taxonomy import Taxonomy
from ..taxonomy import load as load_taxonomy
from .prioritize import QueueItem
from .store import LabelRecord, LabelStore

PAGE = r"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>مراجعة تصنيف الوثائق</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#16191d; --muted:#6b7280; --line:#e3e6ea;
  --accent:#1f6feb; --ok:#137333; --warn:#a15c00; --danger:#b3261e;
  --chip:#eef2f7;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1216; --panel:#171b21; --ink:#e6e8eb; --muted:#9aa4b2; --line:#252b33;
  --accent:#4c8dff; --ok:#4ec27a; --warn:#e0a341; --danger:#ef5350; --chip:#1e242c;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.7 "Segoe UI","Tahoma",system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);
  border-bottom:1px solid var(--line);padding:10px 16px;
  display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.bar{flex:1;min-width:160px;height:6px;background:var(--chip);border-radius:99px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);width:0;transition:width .3s}
.stat{font-size:13px;color:var(--muted);white-space:nowrap}
.stat b{color:var(--ink)}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;
  grid-template-columns:1.35fr .95fr;gap:16px;align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.doc{white-space:pre-wrap;word-break:break-word;max-height:64vh;overflow:auto;
  font-size:16px;line-height:2}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.chip{background:var(--chip);border-radius:99px;padding:2px 10px;font-size:12px;color:var(--muted)}
.chip.lane-random{background:#e8f0fe;color:#1a53b0}
.chip.lane-priority{background:#fdf0e3;color:#8a5300}
@media(prefers-color-scheme:dark){
  .chip.lane-random{background:#16304f;color:#9dc2ff}
  .chip.lane-priority{background:#3a2b12;color:#e8b765}}
h2{font-size:14px;margin:0 0 10px;color:var(--muted);font-weight:600}
.sugg{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:12px}
.sugg .name{font-weight:700}
.conf{height:5px;background:var(--chip);border-radius:99px;margin-top:6px;overflow:hidden}
.conf>i{display:block;height:100%;background:var(--ok)}
.why{color:var(--muted);font-size:13px;margin-top:6px}
.blind{color:var(--warn);font-size:13px;border:1px dashed var(--line);
  border-radius:8px;padding:10px;margin-bottom:12px}
button.opt{display:flex;width:100%;gap:10px;align-items:center;text-align:right;
  background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:8px;padding:9px 10px;margin-bottom:6px;cursor:pointer;font:inherit}
button.opt:hover{border-color:var(--accent)}
button.opt.sel{border-color:var(--accent);box-shadow:0 0 0 2px rgba(31,111,235,.18)}
kbd{background:var(--chip);border:1px solid var(--line);border-radius:5px;
  padding:1px 7px;font-size:12px;min-width:24px;text-align:center;color:var(--muted)}
input[type=search],textarea{width:100%;background:var(--bg);color:var(--ink);
  border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}
.list{max-height:260px;overflow:auto;margin-top:8px}
.row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.row button{flex:1;min-width:110px;border-radius:8px;padding:9px;cursor:pointer;
  font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.primary{background:var(--accent)!important;color:#fff!important;border-color:transparent!important}
.done{text-align:center;padding:60px 20px}
.hint{color:var(--muted);font-size:12px;margin-top:10px}
.reveal{margin-top:10px;font-size:13px}
.agree{color:var(--ok)} .differ{color:var(--danger)}
</style></head><body>
<header>
  <div class="stat">تمت مراجعة <b id="n">0</b> من <b id="total">0</b></div>
  <div class="bar"><i id="prog"></i></div>
  <div class="stat">اتفاق العينة العشوائية: <b id="ragree">—</b></div>
  <div class="stat">اتفاق لوحة الأولوية: <b id="pagree">—</b></div>
</header>
<main id="app"></main>
<script>
const BOOT = __BOOTSTRAP__;
const NAMES = BOOT.names, IDS = BOOT.ids;
let queue = BOOT.items, i = 0, selected = null, startedAt = Date.now();

const el = (h)=>{const d=document.createElement('div');d.innerHTML=h;return d.firstElementChild};
const app = document.getElementById('app');

function setStats(s){
  document.getElementById('n').textContent = i;
  document.getElementById('total').textContent = queue.length;
  document.getElementById('prog').style.width = (queue.length? i/queue.length*100:0)+'%';
  if(s){
    const f=(l)=> l.n? Math.round(l.agreement*100)+'% ('+l.n+')' : '—';
    document.getElementById('ragree').textContent = f(s.random);
    document.getElementById('pagree').textContent = f(s.priority);
  }
}

function shortlist(item){
  const out=[];
  if(item.model_label) out.push(item.model_label);
  (item.alternatives||[]).forEach(a=>{if(!out.includes(a))out.push(a)});
  if(item.baseline_label && !out.includes(item.baseline_label)) out.push(item.baseline_label);
  return out;
}

function render(){
  if(i>=queue.length){
    app.innerHTML='<div class="card done"><h1>انتهت الدفعة</h1>'+
      '<p>تمت مراجعة '+queue.length+' وثيقة. يمكنك إغلاق النافذة.</p>'+
      '<p class="hint">شغّل <code>docrouter label status</code> لعرض الإحصاءات، '+
      'و<code>docrouter label export</code> لتصدير المجموعة الذهبية.</p></div>';
    setStats(null); return;
  }
  const it = queue[i];
  selected = null; startedAt = Date.now();
  const blind = it.blind;
  const list = blind? [] : shortlist(it);

  app.innerHTML='';
  const left = el('<section class="card"></section>');
  const meta = '<div class="meta"><span class="chip lane-'+it.lane+'">'+
    (it.lane==='random'?'عينة عشوائية':'لوحة أولوية')+'</span>'+
    '<span class="chip">'+it.doc_id+'</span>'+
    '<span class="chip">'+(it.source==='ocr'?'نص ضوئي':'نص رقمي')+'</span>'+
    (it.reasons||[]).map(r=>'<span class="chip">'+r+'</span>').join('')+'</div>';
  left.innerHTML = meta + '<div class="doc"></div>';
  left.querySelector('.doc').textContent = it.text;
  app.appendChild(left);

  const right = el('<section class="card"></section>');
  let html='';
  if(blind){
    html += '<div class="blind">هذه وثيقة من العينة العشوائية. اقتراح النموذج '+
            'مخفي عمداً حتى لا يؤثر على حكمك، وسيظهر بعد اختيارك.</div>';
  } else if(it.model_label){
    const c = it.model_confidence||0;
    html += '<div class="sugg"><div class="name">'+NAMES[it.model_label]+'</div>'+
      '<div class="conf"><i style="width:'+(c*100)+'%"></i></div>'+
      '<div class="why">ثقة '+c.toFixed(2)+
      (it.baseline_label && it.baseline_label!==it.model_label
        ? ' · النموذج الإحصائي يرجّح: '+NAMES[it.baseline_label] : '')+'</div>'+
      (it.model_rationale_ar? '<div class="why">'+it.model_rationale_ar+'</div>':'')+
      '</div>';
  }
  html += '<h2>الجهة المختصة</h2><div id="short"></div>';
  html += '<input type="search" id="q" placeholder="ابحث في كل الجهات… (اضغط /)">';
  html += '<div class="list" id="all"></div>';
  html += '<textarea id="notes" rows="2" placeholder="ملاحظة (اختياري)" style="margin-top:10px"></textarea>';
  html += '<div class="row">'+
    '<button class="primary" id="ok">اعتماد ‏(Enter)</button>'+
    '<button id="skip">تخطٍ ‏(S)</button>'+
    '<button id="unclear">غير واضحة ‏(U)</button></div>';
  html += '<div class="hint">الأرقام ١–٩ لاختيار سريع · / للبحث · Enter للاعتماد</div>';
  html += '<div class="reveal" id="reveal"></div>';
  right.innerHTML = html;
  app.appendChild(right);

  const short = right.querySelector('#short');
  list.forEach((id,n)=>{
    const b=el('<button class="opt" data-id="'+id+'"><kbd>'+(n+1)+'</kbd><span>'+NAMES[id]+'</span></button>');
    b.onclick=()=>pick(id); short.appendChild(b);
  });

  const all = right.querySelector('#all');
  function paint(filter){
    all.innerHTML='';
    IDS.filter(id=>!filter || NAMES[id].includes(filter) || id.includes(filter))
       .forEach(id=>{
      const b=el('<button class="opt" data-id="'+id+'"><span>'+NAMES[id]+'</span></button>');
      b.onclick=()=>pick(id); all.appendChild(b);
    });
  }
  paint('');
  right.querySelector('#q').oninput=(e)=>paint(e.target.value.trim());
  right.querySelector('#ok').onclick=()=>submit('labeled');
  right.querySelector('#skip').onclick=()=>submit('skipped');
  right.querySelector('#unclear').onclick=()=>submit('unclear');
  setStats(null);
}

function pick(id){
  selected=id;
  document.querySelectorAll('button.opt').forEach(b=>
    b.classList.toggle('sel', b.dataset.id===id));
}

async function submit(status){
  const it = queue[i];
  if(status==='labeled' && !selected){ alert('اختر الجهة أولاً'); return; }
  const body = {
    doc_id: it.doc_id, status, label: status==='labeled'? selected : '',
    lane: it.lane, path: it.path,
    seconds_spent: (Date.now()-startedAt)/1000,
    notes: (document.getElementById('notes')||{}).value || ''
  };
  const res = await fetch('/api/label',{method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();

  if(it.blind && data.model_label && status==='labeled'){
    // Reveal only after the decision is recorded.
    const box=document.getElementById('reveal');
    const same = data.model_label===selected;
    box.innerHTML = '<span class="'+(same?'agree':'differ')+'">'+
      (same? 'طابق النموذج اختيارك: ':'اختلف النموذج، وقد رجّح: ')+
      NAMES[data.model_label]+'</span>';
    setStats(data.stats);
    setTimeout(()=>{ i++; render(); }, 1400);
    return;
  }
  i++; setStats(data.stats); render();
}

document.addEventListener('keydown',(e)=>{
  if(e.target.tagName==='TEXTAREA') return;
  if(e.key==='/'){e.preventDefault();const q=document.getElementById('q');if(q)q.focus();return}
  if(e.target.tagName==='INPUT'){ if(e.key==='Enter'){e.preventDefault();submit('labeled')} return }
  if(e.key>='1'&&e.key<='9'){
    const b=document.querySelectorAll('#short button.opt')[+e.key-1];
    if(b){b.click()} return}
  if(e.key==='Enter'){e.preventDefault();submit('labeled')}
  if(e.key.toLowerCase()==='s'){submit('skipped')}
  if(e.key.toLowerCase()==='u'){submit('unclear')}
});

render();
</script></body></html>
"""


class ReviewSession:
    """Holds the batch in memory and mediates every decision."""

    def __init__(
        self,
        items: list[QueueItem],
        store: LabelStore,
        taxonomy: Taxonomy,
        reviewer: str,
        blind_random: bool = True,
    ):
        self.items = {item.doc_id: item for item in items}
        self.order = [item.doc_id for item in items]
        self.store = store
        self.taxonomy = taxonomy
        self.reviewer = reviewer
        self.blind_random = blind_random
        self.completed = 0

    def is_blind(self, item: QueueItem) -> bool:
        return self.blind_random and item.lane == "random"

    def bootstrap(self) -> dict:
        payload = []
        for doc_id in self.order:
            item = self.items[doc_id]
            data = item.model_dump()
            if self.is_blind(item):
                # Stripped, not hidden. The prediction never reaches the page.
                for field in (
                    "model_label",
                    "model_confidence",
                    "model_backend",
                    "model_rationale_ar",
                    "alternatives",
                    "baseline_label",
                    "score",
                    "reasons",
                ):
                    data[field] = None if field.startswith("model_") else []
                data["reasons"] = ["عينة عشوائية"]
                data["blind"] = True
            else:
                data["blind"] = False
            payload.append(data)

        names = {i.id: i.name_ar for i in self.taxonomy.institutions}
        return {"items": payload, "names": names, "ids": self.taxonomy.ids}

    def record(self, body: dict) -> dict:
        doc_id = body["doc_id"]
        item = self.items.get(doc_id)
        if item is None:
            raise KeyError(doc_id)

        label = (body.get("label") or "").strip()
        status = body.get("status", "labeled")
        # The body arrives from the browser, so both fields are validated here
        # rather than trusted — the review UI is local, but it is still input.
        if status not in ("labeled", "skipped", "unclear"):
            raise ValueError(f"unknown status: {status!r}")
        if status == "labeled" and label not in set(self.taxonomy.ids):
            raise ValueError(f"unknown institution id: {label!r}")

        record = LabelRecord(
            doc_id=doc_id,
            label=label,
            status=status,
            lane=item.lane,
            model_label=item.model_label,
            model_confidence=item.model_confidence,
            model_backend=item.model_backend,
            reviewer=self.reviewer,
            seconds_spent=body.get("seconds_spent"),
            notes=(body.get("notes") or "")[:2000],
            path=item.path,
        )
        self.store.append(record)
        self.completed += 1

        stats = self.store.stats()
        return {
            "ok": True,
            "model_label": item.model_label,
            "stats": {
                "random": {"n": stats.random.n, "agreement": stats.random.agreement},
                "priority": {"n": stats.priority.n, "agreement": stats.priority.agreement},
            },
        }


def _handler_factory(session: ReviewSession):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the console clean
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                boot = json.dumps(session.bootstrap(), ensure_ascii=False)
                page = PAGE.replace("__BOOTSTRAP__", boot)
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/label":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                result = session.record(body)
            except (KeyError, ValueError) as exc:
                self._send(
                    400,
                    json.dumps({"error": str(exc)}).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return
            self._send(
                200,
                json.dumps(result, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

    return Handler


def serve(
    items: list[QueueItem],
    store: LabelStore,
    reviewer: str,
    taxonomy: Taxonomy | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    blind_random: bool = True,
    open_browser: bool = True,
) -> ReviewSession:
    """Run the review UI until the reviewer stops it. Loopback only."""
    tax = taxonomy or load_taxonomy()
    session = ReviewSession(items, store, tax, reviewer, blind_random=blind_random)
    server = ThreadingHTTPServer((host, port), _handler_factory(session))

    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return session


def save_queue(items: list[QueueItem], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    return path


def load_queue(path: str | Path) -> list[QueueItem]:
    with Path(path).open(encoding="utf-8") as fh:
        return [QueueItem.model_validate_json(line) for line in fh if line.strip()]
