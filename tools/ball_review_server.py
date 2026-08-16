"""Local ball review web UI. Serves contact sheets with per-cell label buttons.

Start:  python tools/ball_review_server.py [--port 8100]
Open:   http://127.0.0.1:8100
"""
import argparse
import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval" / "ball_crops"

SOURCES = ["demo4", "demo1", "demo3", "demo2", "raw1", "raw2"]
LABELS = {"ball", "not_ball", "null"}
REASONS = ["shoe", "sock", "line", "light", "head", "hand", "other"]

app = Flask(__name__)
_cache = {}
_lock = threading.Lock()


def _manifest(src):
    path = EVAL / src / "candidates.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows


def _sheets(src):
    path = EVAL / src / "sheets" / "sheets.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows


def _review_order(src):
    path = EVAL / src / "review_order.txt"
    if not path.is_file():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(src):
    with _lock:
        item = _cache.get(src)
        if item and time.time() - item["t"] < 60:
            return item
        rows = _manifest(src)
        item = {"rows": rows, "by_id": {r["id"]: r for r in rows}, "t": time.time()}
        _cache[src] = item
        return item


def _save_manifest(src, item):
    path = EVAL / src / "candidates.jsonl"
    payload = "\n".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in item["rows"]
    ) + "\n"
    path.write_text(payload, encoding="utf-8")


def _progress(src):
    item = _load(src)
    labeled = sum(1 for r in item["rows"] if r.get("manual_label") in LABELS)
    return {"total": len(item["rows"]), "labeled": labeled}


def _model_fields(row):
    out = {}
    for f in ("kimi_label", "luna_label", "qwen_label"):
        if row.get(f) is not None:
            out[f] = row[f]
    return out


@app.get("/")
def index():
    rows = []
    for s in SOURCES:
        if not (EVAL / s / "candidates.jsonl").is_file():
            continue
        p = _progress(s)
        rows.append((s, p["total"], p["labeled"]))
    return render_template_string(INDEX_HTML, rows=rows)


@app.get("/api/status/<src>")
def status(src):
    if not (EVAL / src / "candidates.jsonl").is_file():
        return jsonify({"error": "no such source"}), 404
    return jsonify(_progress(src))


@app.get("/api/sheets/<src>")
def api_sheets(src):
    if not (EVAL / src / "sheets" / "sheets.jsonl").is_file():
        return jsonify({"error": "no sheets"}), 404
    sheets = _sheets(src)
    item = _load(src)
    labeled = {}
    categories = {}
    ml = {}
    for r in item["rows"]:
        if r.get("manual_label") in LABELS:
            labeled[r["id"]] = r["manual_label"]
        categories[r["id"]] = r["category"]
        models = _model_fields(r)
        if models:
            ml[r["id"]] = ";".join(f"{k}~{v}" for k, v in models.items())
    order = _review_order(src)
    if order is None:
        order = [iid for s in sheets for iid in s["ids"]]
    return jsonify({
        "sheets": sheets,
        "labeled": labeled,
        "categories": categories,
        "ml": ml,
        "order": order,
        "total": len(item["rows"]),
        "labeled_count": len(labeled),
    })


@app.get("/review/<src>")
def review(src):
    return render_template_string(REVIEW_HTML, src=src)


@app.post("/api/save/<src>")
def save(src):
    body = request.get_json(force=True)
    item = _load(src)
    for entry in body:
        iid = entry["id"]
        label = entry.get("label")
        row = item["by_id"].get(iid)
        if row is None:
            return jsonify({"error": "unknown id " + iid}), 400
        if label is None:
            row.pop("manual_label", None)
            row.pop("manual_reason", None)
        elif label in LABELS:
            row["manual_label"] = label
            reason = entry.get("reason")
            if reason:
                row["manual_reason"] = reason
            elif label != "null":
                row.pop("manual_reason", None)
        else:
            return jsonify({"error": "invalid label " + label}), 400
    _save_manifest(src, item)
    return jsonify(_progress(src))


@app.get("/api/crop/<src>/<iid>")
def crop_info(src, iid):
    item = _load(src)
    row = item["by_id"].get(iid)
    if row is None:
        return jsonify({"error": "unknown id"}), 404
    return jsonify({
        "id": iid,
        "crop": row["crop"],
        "category": row["category"],
        "frame": row["frame"],
        "conf": row["conf"],
        "models": _model_fields(row),
        "manual": row.get("manual_label"),
        "reason": row.get("manual_reason"),
    })


@app.get("/crop/<src>/<path:filepath>")
def serve_crop(src, filepath):
    root = EVAL / src
    path = (root / filepath).resolve()
    if not str(path).startswith(str(root.resolve())):
        return jsonify({"error": "bad path"}), 403
    if not path.is_file():
        return jsonify({"error": "missing"}), 404
    return send_file(str(path), conditional=True)


INDEX_HTML = """
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>球检测人工审核</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
a{color:#7cb7ff;text-decoration:none}
table{border-collapse:collapse;margin-top:12px}
td,th{border:1px solid #333;padding:8px 14px;text-align:left}
.done{color:#7dff8a}.todo{color:#ffb86c}
</style></head>
<body>
<h1>球检测人工审核</h1>
<p>规则：eval/ball_crops/README.md —— 每格判定 ball / not_ball / null（太小太糊）。
<br>先审分歧项（review_order 已把分歧排前），再按顺序全部过一遍。
<br>你标的就是唯一权威（manual_label），双模型标签只作参考。</p>
<table><tr><th>源</th><th>总数</th><th>已审</th><th>进度</th></tr>
{% for s, total, labeled in rows %}
<tr><td><a href="/review/{{s}}">{{s}}</a></td><td>{{total}}</td>
<td class="{{'done' if labeled==total else 'todo'}}">{{labeled}}</td>
<td>{{ '100%' if labeled==total else (labeled*100//total)|string + '%' }}</td></tr>
{% endfor %}
</table>
</body></html>
"""


REVIEW_HTML = """
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>审核 {{src}}</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#141414;color:#eee}
header{position:sticky;top:0;background:#1c1c1c;padding:8px 16px;display:flex;gap:12px;align-items:center;border-bottom:1px solid #333;z-index:10;flex-wrap:wrap}
header a{color:#7cb7ff;text-decoration:none}
#progress{color:#ffb86c}
#sheetwrap{position:relative;width:max-content;margin:12px auto}
#sheet{display:block;max-width:min(92vw,1300px);height:auto}
.cell-btn{position:absolute;box-sizing:border-box;cursor:pointer}
.cell-btn:hover{border:3px solid #0af !important;background:rgba(255,255,255,.18)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;flex-direction:column;align-items:center;justify-content:center;z-index:20}
#modal img{max-width:92vw;max-height:68vh}
#modal .info{color:#ccc;margin:8px 0;font-size:14px;max-width:92vw;text-align:center}
.btn{padding:8px 18px;margin:4px;border:1px solid #555;border-radius:6px;background:#2a2a2a;color:#eee;cursor:pointer;font-size:15px}
.btn.act{background:#2d6d3f;border-color:#6fdc8a}
.btn.wrong{background:#7a2f2f;border-color:#ff8585}
.btn.unk{background:#6d642d;border-color:#ffd75e}
.btn.reason{background:#3d3326;border-color:#c9a86a;font-size:13px;padding:6px 12px}
.btn:hover{filter:brightness(1.25)}
#nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sel{background:#1c1c1c;color:#eee;border:1px solid #444;padding:6px;border-radius:4px}
#legend{font-size:13px;color:#9a9a9a;margin:4px 12px 16px;text-align:center}
kbd{background:#2a2a2a;border:1px solid #555;border-radius:4px;padding:1px 6px;font-size:12px}
</style></head>
<body>
<header>
  <a href="/">全部源</a>
  <b>{{src}}</b>
  <span id="progress">-</span>
  <span id="nav">
    <button class="btn" onclick="jump('first')">第一张</button>
    <button class="btn" onclick="jump('prev')">上一张</button>
    <button class="btn" onclick="jump('next')">下一张</button>
    <button class="btn" onclick="jump('unlabeled')">跳到未审</button>
    <select id="sheetselect" class="sel" onchange="go(this.value)"></select>
    <select id="catselect" class="sel" onchange="applyCatFilter()">
      <option value="">全部类别</option>
      <option value="confirmed">confirmed</option>
      <option value="unconfirmed">unconfirmed</option>
      <option value="bridge_sweep">bridge_sweep</option>
      <option value="gap_sweep">gap_sweep</option>
      <option value="global_sweep">global_sweep</option>
    </select>
  </span>
</header>
<div id="sheetwrap">
  <img id="sheet" alt="sheet">
  <div id="cells"></div>
</div>
<div id="legend">
  格子着色：<span style="color:#6fdc8a">■人工=球</span> <span style="color:#ff8585">■人工=非球</span> <span style="color:#ffd75e">■人工=难判</span>
  <span style="color:#e8b14a">▢双模型分歧(优先审)</span> <span style="color:#3fae6e">▢双模型一致=球</span> <span style="color:#ae5a5a">▢一致=非球</span> <span style="color:#4a7dae">▢单模型</span>
  点击格子放大；弹窗内 <kbd>1</kbd>=球 <kbd>2</kbd>=非球 <kbd>3</kbd>=难判 <kbd>q</kbd>鞋 <kbd>w</kbd>袜 <kbd>e</kbd>线 <kbd>r</kbd>灯 <kbd>t</kbd>头 <kbd>y</kbd>手套 <kbd>u</kbd>其他 <kbd>Esc</kbd>关闭
</div>
<div id="modal">
  <img id="modalimg" alt="">
  <div class="info" id="modalinfo"></div>
  <div id="modalbtns">
    <button class="btn act" onclick="setModal('ball')">是球 (1)</button>
    <button class="btn wrong" onclick="setModal('not_ball')">不是球 (2)</button>
    <button class="btn unk" onclick="setModal('null')">无法判定 (3)</button>
    <span id="reasonrow" style="display:inline-block">
      <button class="btn reason" onclick="setModal('not_ball','shoe')">鞋(q)</button>
      <button class="btn reason" onclick="setModal('not_ball','sock')">袜(w)</button>
      <button class="btn reason" onclick="setModal('not_ball','line')">线(e)</button>
      <button class="btn reason" onclick="setModal('not_ball','light')">灯(r)</button>
      <button class="btn reason" onclick="setModal('not_ball','head')">头(t)</button>
      <button class="btn reason" onclick="setModal('not_ball','hand')">手套(y)</button>
      <button class="btn reason" onclick="setModal('not_ball','other')">其他(u)</button>
    </span>
    <button class="btn" onclick="closeModal()">关闭 (Esc)</button>
  </div>
</div>
<script>
const SRC = "{{src}}";
const COLS = 5, ROWS = 4;
let data = null, idx = 0, cur = null, orderSeq = [], sheetsSeq = [];

async function jget(u){const r = await fetch(u); return r.json();}
async function jpost(u, b){const r = await fetch(u, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(b)}); return r.json();}

function isLabeled(id){ return id in data.labeled; }

function parseMl(id){
  const s = data.ml[id];
  if (!s) return {};
  const out = {};
  for (const part of s.split(";")) {
    const [k, v] = part.split("~");
    if (k) out[k.replace("_label","")] = v;
  }
  return out;
}

function cellStyle(id){
  const man = data.labeled[id];
  if (man === "ball") return {border:"2px solid #6fdc8a", background:"rgba(60,190,110,.28)"};
  if (man === "not_ball") return {border:"2px solid #ff8585", background:"rgba(200,70,70,.30)"};
  if (man === "null") return {border:"2px solid #ffd75e", background:"rgba(190,170,70,.22)"};
  const m = parseMl(id);
  const vals = Object.values(m);
  const names = Object.keys(m);
  if (vals.length >= 2) {
    if (new Set(vals).size > 1) return {border:"2px dashed #e8b14a", background:"rgba(230,170,60,.16)"};
    if (vals[0] === "ball") return {border:"1px solid #3fae6e", background:"rgba(60,160,100,.14)"};
    if (vals[0] === "not_ball") return {border:"1px solid #ae5a5a", background:"rgba(170,80,80,.14)"};
    return {border:"1px solid #666", background:"rgba(120,120,120,.10)"};
  }
  if (vals.length === 1) return {border:"1px solid #4a7dae", background:"rgba(70,120,170,.14)"};
  return {border:"1px solid #444", background:"rgba(255,255,255,.03)"};
}

function applyCatFilter(){
  if (!data) return;
  const cat = document.getElementById("catselect").value;
  orderSeq = [];
  for (const id of data.order) {
    if (!cat || data.categories[id] === cat) orderSeq.push(id);
  }
  const seen = new Set();
  sheetsSeq = [];
  for (const id of orderSeq) {
    const sh = data.sheetOf[id];
    if (!seen.has(sh)) { seen.add(sh); sheetsSeq.push(sh); }
  }
  const sel = document.getElementById("sheetselect");
  sel.innerHTML = "";
  for (const sh of sheetsSeq) {
    const ids = data.sheetIds[sh];
    const done = ids.every(isLabeled), any = ids.some(isLabeled);
    const o = document.createElement("option");
    o.value = sh;
    o.textContent = sh + (done ? " ✓" : any ? " …" : "");
    sel.appendChild(o);
  }
  document.getElementById("progress").textContent = "已审 " + data.labeled_count + "/" + data.total;
  if (idx >= sheetsSeq.length) idx = 0;
  renderSheet();
}

function renderSheet(){
  if (!data || sheetsSeq.length === 0) return;
  const sheetName = sheetsSeq[idx];
  document.getElementById("sheet").src = "/crop/" + SRC + "/sheets/" + sheetName;
  document.getElementById("sheetselect").value = sheetName;
  const cells = document.getElementById("cells");
  cells.innerHTML = "";
  const wrap = document.getElementById("sheetwrap");
  data.sheetIds[sheetName].forEach((id, i) => {
    const b = document.createElement("div");
    b.className = "cell-btn";
    const r = Math.floor(i / COLS), c = i % COLS;
    b.style.left = (c * 100 / COLS) + "%";
    b.style.top = (r * 100 / ROWS) + "%";
    b.style.width = (100 / COLS) + "%";
    b.style.height = (100 / ROWS) + "%";
    const st = cellStyle(id);
    b.style.border = st.border;
    b.style.background = st.background;
    const m = parseMl(id);
    b.title = id + " " + (data.categories[id] || "") + " " + (Object.entries(m).map(([k,v])=>k+":"+v).join(" ") || "无预标") + (isLabeled(id) ? " [已标:" + data.labeled[id] + "]" : "");
    b.onclick = () => openCell(id);
    cells.appendChild(b);
  });
}

function jump(kind){
  if (!data || sheetsSeq.length === 0) return;
  if (kind === "first") idx = 0;
  else if (kind === "prev") idx = Math.max(0, idx - 1);
  else if (kind === "next") idx = Math.min(sheetsSeq.length - 1, idx + 1);
  else if (kind === "unlabeled") {
    for (let k = 1; k <= sheetsSeq.length; k++) {
      const j = (idx + k) % sheetsSeq.length;
      if (data.sheetIds[sheetsSeq[j]].some(id => !isLabeled(id))) { idx = j; break; }
    }
  }
  renderSheet();
}

function go(sheetName){
  if (!data) return;
  const i = sheetsSeq.indexOf(sheetName);
  if (i >= 0) { idx = i; renderSheet(); }
}

function openCell(id){
  fetch("/api/crop/" + SRC + "/" + id).then(r => r.json()).then(info => {
    cur = id;
    document.getElementById("modalimg").src = "/crop/" + SRC + "/" + info.crop;
    const m = info.models || {};
    const mtxt = Object.entries(m).map(([k,v]) => k.replace("_label","") + ":" + v).join(" · ") || "无预标";
    const pos = orderSeq.indexOf(id);
    document.getElementById("modalinfo").innerHTML =
      "<b>" + info.id + "</b> · " + info.category + " · frame " + info.frame + " · conf " + (info.conf === null ? "-" : Number(info.conf).toFixed(2)) +
      "<br>模型: " + mtxt +
      (info.manual ? " · 已标: " + info.manual + (info.reason ? "(" + info.reason + ")" : "") : "") +
      " · 顺序 " + (pos + 1) + "/" + orderSeq.length;
    document.getElementById("modal").style.display = "flex";
  });
}

function closeModal(){ document.getElementById("modal").style.display = "none"; cur = null; }

function nextUnlabeled(){
  if (!data || !orderSeq.length) return null;
  const start = cur ? orderSeq.indexOf(cur) : -1;
  for (let k = 1; k <= orderSeq.length; k++) {
    const id = orderSeq[(start + k) % orderSeq.length];
    if (!isLabeled(id)) return id;
  }
  return null;
}

function setModal(label, reason){
  if (!cur) return;
  const id = cur;
  const payload = [{id, label}];
  if (reason) payload[0].reason = reason;
  jpost("/api/save/" + SRC, payload).then(p => {
    data.labeled[id] = label;
    data.labeled_count = p.labeled;
    document.getElementById("progress").textContent = "已审 " + p.labeled + "/" + p.total;
    const nxt = nextUnlabeled();
    if (nxt) { openCell(nxt); }
    else { closeModal(); document.getElementById("modalinfo").innerHTML = "全部审完"; }
  });
}

document.addEventListener("keydown", (e) => {
  if (document.getElementById("modal").style.display !== "flex") return;
  const k = e.key.toLowerCase();
  if (k === "1") setModal("ball");
  else if (k === "2") setModal("not_ball");
  else if (k === "3") setModal("null");
  else if (k === "q") setModal("not_ball", "shoe");
  else if (k === "w") setModal("not_ball", "sock");
  else if (k === "e") setModal("not_ball", "line");
  else if (k === "r") setModal("not_ball", "light");
  else if (k === "t") setModal("not_ball", "head");
  else if (k === "y") setModal("not_ball", "hand");
  else if (k === "u") setModal("not_ball", "other");
  else if (k === "escape") closeModal();
  else if (k === "enter") { const n = nextUnlabeled(); if (n) openCell(n); }
});

fetch("/api/sheets/" + SRC).then(r => r.json()).then(d => {
  data = d;
  data.sheetOf = {};
  data.sheetIds = {};
  for (const s of d.sheets) {
    data.sheetIds[s.sheet] = s.ids;
    for (const id of s.ids) data.sheetOf[id] = s.sheet;
  }
  applyCatFilter();
});
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
