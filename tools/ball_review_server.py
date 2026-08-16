"""Local ball review web UI. Serves contact sheets with per-cell label buttons
and the original video frame (0.5s window) next to the zoomed crop.

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
VIDEO_DIR = Path("C:/Personal Profile/Profile/Video")

SOURCES = ["demo4", "demo1", "demo3", "demo2", "raw1", "raw2"]
LABELS = {"ball", "not_ball", "null"}
REASONS = ["shoe", "sock", "line", "light", "head", "hand", "penalty_spot",
           "debris", "goal", "corner_flag", "referee", "other"]

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
    reasons = {}
    for r in item["rows"]:
        if r.get("manual_label") in LABELS:
            labeled[r["id"]] = r["manual_label"]
            if r.get("manual_reason"):
                reasons[r["id"]] = r["manual_reason"]
        categories[r["id"]] = r["category"]
    order = _review_order(src)
    if order is None:
        order = [iid for s in sheets for iid in s["ids"]]
    return jsonify({
        "sheets": sheets,
        "labeled": labeled,
        "reasons": reasons,
        "categories": categories,
        "order": order,
        "total": len(item["rows"]),
        "labeled_count": len(labeled),
    })


@app.get("/video/<src>")
def video(src):
    if src not in SOURCES:
        return jsonify({"error": "bad src"}), 404
    path = VIDEO_DIR / f"{src}.mp4"
    if not path.is_file():
        return jsonify({"error": "no video"}), 404
    return send_file(str(path), conditional=True, mimetype="video/mp4")


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
        "t": row["t"],
        "bbox": row["bbox"],
        "conf": row["conf"],
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
<br>纯人工审核：全部格子你逐一判定，标注即时保存，随时可回退修改。</p>
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
#mrow{display:flex;gap:14px;align-items:flex-start;justify-content:center;flex-wrap:wrap;max-width:97vw}
#vidwrap{position:relative;display:inline-block;background:#000}
#vid{display:block;max-width:62vw;max-height:64vh;width:auto;height:auto}
#bboxov{position:absolute;border:2px solid #ff3b3b;box-shadow:0 0 8px rgba(255,60,60,.9);pointer-events:none;display:none;z-index:2}
#modalimg{display:block;max-width:30vw;max-height:64vh;border:1px solid #333}
#vidctl{margin-top:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
#modal .info{color:#ccc;margin:8px 0;font-size:14px;max-width:92vw;text-align:center}
.btn{padding:8px 18px;margin:4px;border:1px solid #555;border-radius:6px;background:#2a2a2a;color:#eee;cursor:pointer;font-size:15px}
.btn.act{background:#2d6d3f;border-color:#6fdc8a}
.btn.wrong{background:#7a2f2f;border-color:#ff8585}
.btn.unk{background:#6d642d;border-color:#ffd75e}
.btn.reason{background:#3d3326;border-color:#c9a86a;font-size:13px;padding:6px 12px}
.btn:hover{filter:brightness(1.25)}
#nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sel{background:#1c1c1c;color:#eee;border:1px solid #444;padding:6px;border-radius:4px}
#legend{font-size:13px;color:#b9b9b9;margin:4px 12px 16px;text-align:center;line-height:1.9}
.chip{display:inline-block;padding:1px 10px;margin:0 3px;border-radius:4px;background:#1c1c1c;white-space:nowrap}
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
  <span class="chip" style="border:2px solid #6fdc8a;background:rgba(60,190,110,.28)">已标:球</span>
  <span class="chip" style="border:2px solid #ff8585;background:rgba(200,70,70,.30)">已标:非球</span>
  <span class="chip" style="border:2px solid #c084fc;background:rgba(160,110,240,.28)">已标:点球点</span>
  <span class="chip" style="border:2px solid #fb923c;background:rgba(240,140,70,.26)">已标:杂物</span>
  <span class="chip" style="border:2px solid #ffd75e;background:rgba(190,170,70,.22)">已标:难判</span>
  <span class="chip" style="border:1px solid #444">未标</span>
  <br>粗框=你已经标过的；未标格子为深灰细框；鼠标悬停格子可看 id+类别；审核按视频时间顺序
  <br>两段式标注：<kbd>1</kbd>=是球 <kbd>2</kbd>=不是 → 再选原因 <kbd>q</kbd>鞋 <kbd>w</kbd>袜 <kbd>e</kbd>场地线 <kbd>r</kbd>灯 <kbd>t</kbd>头 <kbd>y</kbd>手/手套 <kbd>u</kbd>点球点 <kbd>i</kbd>场上杂物 <kbd>o</kbd>球门/球网 <kbd>p</kbd>角旗 <kbd>a</kbd>裁判/装备 <kbd>s</kbd>其他 <kbd>Enter</kbd>跳过原因 <kbd>Esc</kbd>返回 <kbd>3</kbd>=无法判断
  <br>其他：<kbd>v</kbd>重播视频 <kbd>l</kbd>循环 <kbd>[</kbd><kbd>]</kbd>已标间切换 <kbd>g</kbd>输入id/序号跳转 <kbd>Backspace</kbd>回退上一标 <kbd>0</kbd>清除 <kbd>Esc</kbd>关闭
</div>
<div id="modal">
  <div id="mrow">
    <div id="vidwrap">
      <video id="vid" muted playsinline preload="auto"></video>
      <div id="bboxov"></div>
    </div>
    <img id="modalimg" alt="">
  </div>
  <div id="vidctl">
    <button class="btn" onclick="replay()">重播半秒 (v)</button>
    <label class="btn" style="display:inline-flex;align-items:center;gap:6px"><input type="checkbox" id="loopchk"> 循环 (l)</label>
    <span style="color:#9a9a9a;font-size:13px">拖动进度条可看任意帧</span>
  </div>
  <div class="info" id="modalinfo"></div>
  <div id="modalbtns">
    <span id="stage1">
      <button class="btn act" onclick="setModal('ball')">是球 (1)</button>
      <button class="btn wrong" onclick="enterReason()">不是 (2)</button>
      <button class="btn unk" onclick="setModal('null')">无法判断 (3)</button>
    </span>
    <span id="stage2" style="display:none">
      <span style="color:#ffb86c;font-size:14px;margin:0 6px">不是球，选它是什么：</span>
      <button class="btn reason" onclick="setModal('not_ball','shoe')">鞋(q)</button>
      <button class="btn reason" onclick="setModal('not_ball','sock')">袜(w)</button>
      <button class="btn reason" onclick="setModal('not_ball','line')">场地线(e)</button>
      <button class="btn reason" onclick="setModal('not_ball','light')">灯(r)</button>
      <button class="btn reason" onclick="setModal('not_ball','head')">头(t)</button>
      <button class="btn reason" onclick="setModal('not_ball','hand')">手/手套(y)</button>
      <button class="btn reason" style="background:#3d2a52;border-color:#c084fc" onclick="setModal('not_ball','penalty_spot')">点球点(u)</button>
      <button class="btn reason" style="background:#4a3523;border-color:#fb923c" onclick="setModal('not_ball','debris')">场上杂物(i)</button>
      <button class="btn reason" onclick="setModal('not_ball','goal')">球门/球网(o)</button>
      <button class="btn reason" onclick="setModal('not_ball','corner_flag')">角旗(p)</button>
      <button class="btn reason" onclick="setModal('not_ball','referee')">裁判/装备(a)</button>
      <button class="btn reason" onclick="setModal('not_ball','other')">其他(s)</button>
      <button class="btn" onclick="setModal('not_ball')">跳过原因 (Enter)</button>
      <button class="btn" onclick="exitReason()">返回 (Esc)</button>
    </span>
    <button class="btn" onclick="stepLabeled(-1)">上一张已标 ([)</button>
    <button class="btn" onclick="stepLabeled(1)">下一张已标 (])</button>
    <input id="jumpin" class="sel" placeholder="跳: id 或序号 (g)" onkeydown="jumpKey(event)">
    <button class="btn" onclick="undo()">回退 (Backspace)</button>
    <button class="btn" onclick="clearLabel()">清除 (0)</button>
    <button class="btn" onclick="closeModal()">关闭 (Esc)</button>
  </div>
</div>
<script>
const SRC = "{{src}}";
const COLS = 5, ROWS = 4;
const RZ = {shoe:"鞋", sock:"袜", line:"场地线", light:"灯", head:"头", hand:"手/手套", penalty_spot:"点球点", debris:"场上杂物", goal:"球门/球网", corner_flag:"角旗", referee:"裁判/装备", other:"其他"};
const CLIP = 0.25;
let data = null, idx = 0, cur = null, orderSeq = [], sheetsSeq = [], undoStack = [], curInfo = null;
let curT = 0, curBbox = null, stage = 1;

async function jget(u){const r = await fetch(u); return r.json();}
async function jpost(u, b){const r = await fetch(u, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(b)}); return r.json();}

function isLabeled(id){ return id in data.labeled; }

function cellStyle(id){
  const man = data.labeled[id];
  if (man === "ball") return {border:"2px solid #6fdc8a", background:"rgba(60,190,110,.28)"};
  if (man === "not_ball") {
    if (data.reasons[id] === "penalty_spot") return {border:"2px solid #c084fc", background:"rgba(160,110,240,.28)"};
    if (data.reasons[id] === "debris") return {border:"2px solid #fb923c", background:"rgba(240,140,70,.26)"};
    return {border:"2px solid #ff8585", background:"rgba(200,70,70,.30)"};
  }
  if (man === "null") return {border:"2px solid #ffd75e", background:"rgba(190,170,70,.22)"};
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
    b.title = id + " " + (data.categories[id] || "") + (isLabeled(id) ? " [已标:" + data.labeled[id] + (data.reasons[id] ? "/" + (RZ[data.reasons[id]] || data.reasons[id]) : "") + "]" : "");
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

function setupVideo(info){
  const vid = document.getElementById("vid");
  const ov = document.getElementById("bboxov");
  curT = info.t;
  curBbox = info.bbox;
  const url = "/video/" + SRC;
  if (vid.getAttribute("src") !== url) {
    vid.setAttribute("src", url);
  }
  const b = curBbox;
  const fullFrame = b && b[0] <= 0.5 && b[1] <= 0.5 && b[2] >= 1919.5 && b[3] >= 1079.5;
  if (b && !fullFrame) {
    ov.style.display = "block";
    ov.style.left = (b[0] / 1920 * 100) + "%";
    ov.style.top = (b[1] / 1080 * 100) + "%";
    ov.style.width = ((b[2] - b[0]) / 1920 * 100) + "%";
    ov.style.height = ((b[3] - b[1]) / 1080 * 100) + "%";
  } else {
    ov.style.display = "none";
  }
  replay();
}

function replay(){
  const vid = document.getElementById("vid");
  if (!vid) return;
  const s = Math.max(0, curT - CLIP);
  const e = curT + CLIP;
  const start = () => {
    vid.currentTime = s;
    vid.play().catch(() => {});
  };
  vid.ontimeupdate = () => {
    if (vid.currentTime >= e) {
      if (document.getElementById("loopchk").checked) { vid.currentTime = s; return; }
      vid.pause();
    }
  };
  if (vid.readyState >= 1 && isFinite(vid.duration)) {
    start();
  } else {
    vid.addEventListener("loadedmetadata", start, {once: true});
  }
}

function openCell(id){
  fetch("/api/crop/" + SRC + "/" + id).then(r => r.json()).then(info => {
    cur = id;
    curInfo = info;
    stage = 1;
    syncStage();
    document.getElementById("modalimg").src = "/crop/" + SRC + "/" + info.crop;
    const pos = orderSeq.indexOf(id);
    document.getElementById("modalinfo").innerHTML =
      "<b>" + info.id + "</b> · " + info.category + " · frame " + info.frame + " · t=" + Number(info.t).toFixed(2) + "s · conf " + (info.conf === null ? "-" : Number(info.conf).toFixed(2)) +
      (info.manual ? " · 已标: " + (info.manual === "not_ball" && info.reason ? "非球(" + (RZ[info.reason] || info.reason) + ")" : info.manual) : "") +
      " · 顺序 " + (pos + 1) + "/" + orderSeq.length;
    document.getElementById("modal").style.display = "flex";
    setupVideo(info);
  });
}

function closeModal(){
  document.getElementById("modal").style.display = "none";
  const vid = document.getElementById("vid");
  if (vid) vid.pause();
  cur = null;
  curInfo = null;
  renderSheet();
}

function nextUnlabeled(){
  if (!data || !orderSeq.length) return null;
  const start = cur ? orderSeq.indexOf(cur) : -1;
  for (let k = 1; k <= orderSeq.length; k++) {
    const id = orderSeq[(start + k) % orderSeq.length];
    if (!isLabeled(id)) return id;
  }
  return null;
}

function applyLabel(label, reason, advance){
  if (!cur || !curInfo) return;
  const id = cur;
  const prevLabel = curInfo.manual === undefined ? null : curInfo.manual;
  const prevReason = curInfo.reason === undefined ? null : curInfo.reason;
  const payload = [{id, label}];
  if (reason) payload[0].reason = reason;
  jpost("/api/save/" + SRC, payload).then(p => {
    undoStack.push({id, label: prevLabel, reason: prevReason});
    if (undoStack.length > 1000) undoStack.shift();
    if (label === null) delete data.labeled[id]; else data.labeled[id] = label;
    if (label === null) delete data.reasons[id];
    else if (reason) data.reasons[id] = reason;
    else delete data.reasons[id];
    data.labeled_count = p.labeled;
    document.getElementById("progress").textContent = "已审 " + p.labeled + "/" + p.total;
    if (advance) {
      const nxt = nextUnlabeled();
      if (nxt) { openCell(nxt); }
      else { closeModal(); }
    } else {
      openCell(id);
    }
  });
}

function setModal(label, reason){ applyLabel(label, reason || null, true); }
function clearLabel(){ applyLabel(null, null, false); }

function enterReason(){ stage = 2; syncStage(); }
function exitReason(){ stage = 1; syncStage(); }
function syncStage(){
  document.getElementById("stage1").style.display = stage === 1 ? "inline-block" : "none";
  document.getElementById("stage2").style.display = stage === 2 ? "inline-block" : "none";
}

function undo(){
  if (!undoStack.length) return;
  const last = undoStack.pop();
  const id = last.id;
  const payload = [{id, label: last.label}];
  if (last.label !== null && last.reason) payload[0].reason = last.reason;
  jpost("/api/save/" + SRC, payload).then(p => {
    if (last.label === null) delete data.labeled[id]; else data.labeled[id] = last.label;
    if (last.label === null || !last.reason) delete data.reasons[id]; else data.reasons[id] = last.reason;
    data.labeled_count = p.labeled;
    document.getElementById("progress").textContent = "已审 " + p.labeled + "/" + p.total;
    openCell(id);
  });
}

function stepLabeled(dir){
  if (!data || !orderSeq.length) return;
  const start = cur ? orderSeq.indexOf(cur) : -1;
  for (let k = 1; k <= orderSeq.length; k++) {
    const j = (start + dir * k + orderSeq.length * 2) % orderSeq.length;
    if (isLabeled(orderSeq[j])) { openCell(orderSeq[j]); return; }
  }
}

function jumpTo(t){
  if (!data || !orderSeq.length) return;
  let id = null;
  if (/^\d+$/.test(t)) {
    const n = parseInt(t, 10);
    if (n >= 1 && n <= orderSeq.length) id = orderSeq[n - 1];
  } else if (data.sheetOf[t]) {
    id = t;
  }
  if (id) {
    openCell(id);
    document.getElementById("jumpin").value = "";
  }
}

function jumpKey(e){
  if (e.key === "Enter") jumpTo(e.target.value);
}

document.addEventListener("keydown", (e) => {
  if (document.getElementById("modal").style.display !== "flex") return;
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) return;
  const k = e.key.toLowerCase();
  if (stage === 2) {
    if (k === "q") setModal("not_ball", "shoe");
    else if (k === "w") setModal("not_ball", "sock");
    else if (k === "e") setModal("not_ball", "line");
    else if (k === "r") setModal("not_ball", "light");
    else if (k === "t") setModal("not_ball", "head");
    else if (k === "y") setModal("not_ball", "hand");
    else if (k === "u") setModal("not_ball", "penalty_spot");
    else if (k === "i") setModal("not_ball", "debris");
    else if (k === "o") setModal("not_ball", "goal");
    else if (k === "p") setModal("not_ball", "corner_flag");
    else if (k === "a") setModal("not_ball", "referee");
    else if (k === "s") setModal("not_ball", "other");
    else if (k === "enter") setModal("not_ball");
    else if (k === "escape") exitReason();
    else if (k === "backspace") { e.preventDefault(); undo(); }
    else if (k === "0") clearLabel();
    return;
  }
  if (k === "1") setModal("ball");
  else if (k === "2") enterReason();
  else if (k === "3") setModal("null");
  else if (k === "v") replay();
  else if (k === "l") { const c = document.getElementById("loopchk"); c.checked = !c.checked; }
  else if (k === "escape") closeModal();
  else if (k === "backspace") { e.preventDefault(); undo(); }
  else if (k === "0") clearLabel();
  else if (k === "enter") { const n = nextUnlabeled(); if (n) openCell(n); }
  else if (k === "[") stepLabeled(-1);
  else if (k === "]") stepLabeled(1);
  else if (k === "g") { e.preventDefault(); document.getElementById("jumpin").focus(); document.getElementById("jumpin").select(); }
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
