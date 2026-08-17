"""Video ball annotation web UI. Watch the original video and box the ball
directly on the frames; keyframe annotations + interpolation.

Start:  python tools/video_annotate_server.py [--port 8101]
Open:   http://127.0.0.1:8101/annotate/demo4
"""
import argparse
import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path("C:/Personal Profile/Profile/Video")
GT_DIR = ROOT / "eval" / "ball_gt"
TRACKS_DIR = ROOT / "output_videos"

SOURCES = ["demo4", "demo1", "demo3", "demo2", "raw1", "raw2", "raw4"]
FPS = 29.97

app = Flask(__name__)
_cache = {}
_lock = threading.Lock()


def _load_annotations(src):
    path = GT_DIR / src / "annotations.jsonl"
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            rows[int(d["frame"])] = d
    return rows


def _save_annotations(src, rows):
    path = GT_DIR / src / "annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(rows[f], ensure_ascii=False, separators=(",", ":"))
        for f in sorted(rows)
    ) + "\n"
    path.write_text(payload, encoding="utf-8")


def _tracks_index(src):
    with _lock:
        item = _cache.get(src)
        if item and time.time() - item["t"] < 300:
            return item
        idx = {}
        path = TRACKS_DIR / src / "ball" / "ball_tracks.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                tr = d.get("track")
                idx[int(d["frame"])] = tr["bbox"] if tr else None
        item = {"idx": idx, "t": time.time()}
        _cache[src] = item
        return item


@app.get("/")
def index():
    rows = []
    for s in SOURCES:
        ann = _load_annotations(s)
        segs = len({a["seg"] for a in ann.values()}) if ann else 0
        rows.append((s, len(ann), segs))
    return render_template_string(INDEX_HTML, rows=rows)


@app.get("/video/<src>")
def video(src):
    if src not in SOURCES:
        return jsonify({"error": "bad src"}), 404
    path = VIDEO_DIR / f"{src}.mp4"
    if not path.is_file():
        return jsonify({"error": "no video"}), 404
    return send_file(str(path), conditional=True, mimetype="video/mp4")


@app.get("/annotate/<src>")
def annotate(src):
    return render_template_string(PAGE_HTML, src=src)


@app.get("/api/gt/<src>")
def api_gt(src):
    return jsonify({"annotations": sorted(_load_annotations(src).values(), key=lambda a: a["frame"])})


@app.post("/api/gt/save/<src>")
def api_gt_save(src):
    body = request.get_json(force=True)
    rows = {}
    for entry in body.get("annotations", []):
        frame = int(entry["frame"])
        rows[frame] = {
            "frame": frame,
            "t": round(frame / FPS, 4),
            "bbox": [float(v) for v in entry["bbox"]],
            "seg": int(entry.get("seg", 0)),
        }
    _save_annotations(src, rows)
    return jsonify({"count": len(rows)})


@app.get("/api/hint/<src>/<int:frame>")
def api_hint(src, frame):
    idx = _tracks_index(src)["idx"]
    return jsonify({"bbox": idx.get(frame)})


INDEX_HTML = """
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>视频球标注</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
a{color:#7cb7ff;text-decoration:none}
table{border-collapse:collapse;margin-top:12px}
td,th{border:1px solid #333;padding:8px 14px;text-align:left}
</style></head>
<body>
<h1>视频球标注</h1>
<p>在视频里直接框选球：暂停 → 点击/拖拽画框 → 继续播放。关键帧自动插值。
球出视野不标，球再出现自动开新段。产出 <code>eval/ball_gt/&lt;src&gt;/annotations.jsonl</code>。</p>
<table><tr><th>源</th><th>关键帧数</th><th>段数</th><th></th></tr>
{% for s, n, segs in rows %}
<tr><td>{{s}}</td><td>{{n}}</td><td>{{segs}}</td><td><a href="/annotate/{{s}}">开始标注</a></td></tr>
{% endfor %}
</table>
</body></html>
"""

PAGE_HTML = """
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>标注 {{src}}</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1c1c1c;padding:8px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333;z-index:10}
header a{color:#7cb7ff;text-decoration:none}
#main{display:flex;flex-direction:column;align-items:center;padding:12px}
#stage{position:relative;width:min(96vw,1600px);background:#000;line-height:0;overflow:hidden}
#stage video{width:100%;display:block;transform-origin:0 0}
#stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none;transform-origin:0 0}
#zbadge{position:absolute;top:8px;right:12px;background:rgba(0,0,0,.65);color:#ffd75e;font:bold 20px system-ui;padding:2px 10px;border-radius:6px;display:none;pointer-events:none;z-index:3}
#info{font-size:13px;color:#9a9a9a;margin-top:8px;text-align:center;line-height:1.7}
kbd{background:#2a2a2a;border:1px solid #555;border-radius:4px;padding:1px 6px;font-size:12px}
#stats{color:#ffb86c}
.btn{padding:6px 14px;margin:2px;border:1px solid #555;border-radius:6px;background:#2a2a2a;color:#eee;cursor:pointer}
.btn:hover{filter:brightness(1.25)}
#marks{display:flex;flex-wrap:wrap;gap:4px;max-width:96vw;margin:8px auto 4px}
.mark{font-size:11px;padding:1px 6px;border-radius:8px;background:#2d4a2d;color:#9fdf9f;cursor:pointer}
.mark.cur{background:#7a5a1a;color:#ffd75e}
#legend{font-size:13px;color:#b9b9b9;margin:6px auto 12px;text-align:center;line-height:2}
.chip{display:inline-block;padding:1px 10px;margin:0 4px;border-radius:4px;background:#1c1c1c;white-space:nowrap}
</style></head>
<body>
<header>
  <a href="/">全部源</a>
  <b>{{src}}</b>
  <span id="stats">-</span>
  <button class="btn" onclick="saveNow()">保存 (Ctrl+S)</button>
  <span id="zind">缩放:关（z 开启）</span>
  <button class="btn" onclick="zoomStep(1.3)">放大+</button>
  <button class="btn" onclick="zoomStep(1/1.3)">缩小-</button>
  <button class="btn" onclick="zoomReset()">复位</button>
  <span id="hintsw"><input type="checkbox" id="hintbox" checked onchange="hintOn=this.checked;draw()"> 检测器提示 (h)</span>
  <span id="sizesw">框大小 <button class="btn" onclick="resizeBox(-4)">-</button> <span id="sizev">16</span> <button class="btn" onclick="resizeBox(4)">+</button></span>
</header>
<div id="main">
  <div id="stage">
    <video id="vid" preload="auto"></video>
    <canvas id="cv"></canvas>
    <div id="zbadge"></div>
  </div>
  <div id="info">
    <kbd>空格</kbd>播放/暂停 <kbd>←</kbd><kbd>→</kbd>单帧(±1) <kbd>Shift</kbd>+<kbd>←→</kbd>±10帧
    <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd>倍速0.5/1/2× <kbd>Del</kbd>删当前框
    <kbd>Backspace</kbd>撤销 <kbd>h</kbd>检测器提示 <kbd>[</kbd><kbd>]</kbd>框大小
    <kbd>z</kbd>缩放模式（滚轮缩放·左键画框·右键拖动平移；再按 z 退出）
    <br>画框：左键点击=固定大小框（居中），左键拖拽=任意矩形；左键拖动框体移动、拖角缩放；右键按住拖动=平移画面
    <br>实线绿框=你打的关键帧；虚线绿框=两关键帧之间自动插值的中间帧（随播放实时显示，导出时写入GT）
  </div>
  <div id="marks"></div>
  <div id="legend">
    <span class="chip" style="border:2px solid #3fdc6a;background:rgba(63,220,106,.15)">实线绿框 = 你标注的关键帧</span>
    <span class="chip" style="border:2px dashed #8ce6a0;background:rgba(140,230,160,.10)">虚线绿框 = 关键帧之间自动插值的中间帧（播放实时显示）</span>
    <span class="chip" style="border:2px solid #7cb7ff;background:rgba(124,183,255,.12)">蓝色框 = 正在拖拽/创建的新框</span>
    <span class="chip" style="border:2px dashed rgba(255,80,80,.7);background:rgba(255,80,80,.08)">红色虚线框 = 检测器候选提示（h 键开关）</span>
    <span class="chip" style="background:#3a3a3a">右键按住拖动 = 平移画面；z = 缩放模式（滚轮缩放）</span>
  </div>
</div>
<script>
const SRC = "{{src}}";
const NAT_W = 1920, NAT_H = 1080;
const vid = document.getElementById("vid");
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
let ann = {};            // frame -> {bbox:[x1,y1,x2,y2], seg}
let segCounter = 0;
let fixedSize = 16;
let hintOn = true;
let draft = null;        // {x1,y1,x2,y2, mode:'new'|'move'|'resize'|'pan', ...}
let editBox = null;      // box being edited on current frame (exists in ann)
let zoom = 1, panX = 0, panY = 0, zoomMode = false;

vid.src = "/video/" + SRC;
vid.addEventListener("timeupdate", draw);
vid.addEventListener("play", draw);
vid.addEventListener("pause", draw);

function scale(){ return cv.width / NAT_W; }
function stageRect(){ return vid.getBoundingClientRect(); }
function toNat(clientX, clientY){
  const r = stageRect();
  const s = r.width / NAT_W;
  return [(clientX - r.left) / s, (clientY - r.top) / s];
}
function tol(){ return 8 / scale(); }

function clampPan(){
  const r = stageRect();
  const sw = r.width / zoom, sh = r.height / zoom;
  panX = Math.max(sw * (1 - zoom), Math.min(0, panX));
  panY = Math.max(sh * (1 - zoom), Math.min(0, panY));
}

function applyZoom(){
  clampPan();
  const t = "translate(" + panX + "px," + panY + "px) scale(" + zoom + ")";
  vid.style.transform = t;
  cv.style.transform = t;
  updateZoomInd();
  draw();
}

function updateZoomInd(){
  const el = document.getElementById("zind");
  if (el) el.textContent = zoomMode ? "缩放:开 " + zoom.toFixed(1) + "x（滚轮缩放·左键画框·右键拖动）" : "缩放:关（z 开启）";
  const badge = document.getElementById("zbadge");
  if (badge){
    if (zoom > 1.01){
      badge.textContent = zoom.toFixed(1) + "x";
      badge.style.display = "block";
    } else {
      badge.style.display = "none";
    }
  }
}

function zoomStep(factor){
  zoomMode = true;
  zoom = Math.min(24, Math.max(1, zoom * factor));
  applyZoom();
}

function zoomReset(){
  zoomMode = false;
  zoom = 1; panX = 0; panY = 0;
  applyZoom();
}

function sizeCanvas(){
  const r = stageRect();
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  if (cv.width !== w || cv.height !== h){
    cv.width = w;
    cv.height = h;
  }
}
if (window.ResizeObserver) new ResizeObserver(sizeCanvas).observe(vid);
window.addEventListener("resize", sizeCanvas);
vid.addEventListener("loadedmetadata", sizeCanvas);

function interpBoxAt(f){
  if (ann[f]) return null;
  const keys = Object.keys(ann).map(Number).sort((a,b)=>a-b);
  let prev = null, next = null;
  for (const k of keys){ if (k < f) prev = k; else break; }
  for (const k of keys){ if (k > f){ next = k; break; } }
  if (prev === null || next === null) return null;
  if (ann[prev].seg !== ann[next].seg) return null;
  if (next - prev > 60 || f - prev > 60 || next - f > 60) return null;
  const t = (f - prev) / (next - prev);
  const a = ann[prev].bbox, b = ann[next].bbox;
  return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t, a[3]+(b[3]-a[3])*t];
}

function draw(){
  sizeCanvas();
  const s = scale();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.lineWidth = 1.5;
  const f = Math.round(vid.currentTime * 29.97);
  if (!draft){
    const a = ann[f];
    if (a){
      const [x1,y1,x2,y2] = a.bbox;
      ctx.strokeStyle = "#3fdc6a";
      ctx.strokeRect(x1*s, y1*s, (x2-x1)*s, (y2-y1)*s);
      ctx.fillStyle = "#3fdc6a";
      ctx.fillRect(x2*s-2, y1*s-2, 4, 4);
    } else {
      const ib = interpBoxAt(f);
      if (ib){
        const [x1,y1,x2,y2] = ib;
        ctx.strokeStyle = "rgba(140,230,160,.75)";
        ctx.setLineDash([5,4]);
        ctx.strokeRect(x1*s, y1*s, (x2-x1)*s, (y2-y1)*s);
        ctx.setLineDash([]);
      }
    }
  }
  if (hintOn){
    fetch("/api/hint/" + SRC + "/" + f).then(r=>r.json()).then(d=>{
      if (d.bbox && !ann[f]){
        const [x1,y1,x2,y2] = d.bbox;
        ctx.strokeStyle = "rgba(255,80,80,.55)";
        ctx.setLineDash([4,4]);
        ctx.strokeRect(x1*s, y1*s, (x2-x1)*s, (y2-y1)*s);
        ctx.setLineDash([]);
      }
    });
  }
  if (draft && draft.mode !== "pan"){
    ctx.strokeStyle = "#7cb7ff";
    ctx.strokeRect(draft.x1*s, draft.y1*s, (draft.x2-draft.x1)*s, (draft.y2-draft.y1)*s);
  }
}

function currentFrame(){ return Math.round(vid.currentTime * 29.97); }

function addKeyframe(f, bbox){
  let seg = null;
  for (const k in ann){
    if (Math.abs(parseInt(k) - f) <= 120){ seg = ann[k].seg; break; }
  }
  if (seg === null){ seg = segCounter++; }
  ann[f] = {bbox, seg};
  saveNow();
  refreshStats();
}

function deleteKeyframe(f){
  if (ann[f]){ delete ann[f]; saveNow(); refreshStats(); }
}

function undoLast(){
  const frames = Object.keys(ann).map(Number).sort((a,b)=>b-a);
  if (frames.length){
    const last = frames[0];
    // remove last-added keyframe: prefer current frame if it has one
    const f = currentFrame();
    if (ann[f]){ delete ann[f]; }
    else { delete ann[last]; }
    saveNow(); refreshStats();
  }
}

function saveNow(){
  const payload = Object.keys(ann).map(k => {
    const f = parseInt(k);
    return {frame: f, bbox: ann[f].bbox, seg: ann[f].seg};
  });
  fetch("/api/gt/save/" + SRC, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({annotations: payload}),
  }).then(()=>{ document.getElementById("stats").textContent = "已存 " + payload.length + " 关键帧"; });
}

function refreshStats(){
  const keys = Object.keys(ann).map(Number).sort((a,b)=>a-b);
  const segs = new Set(keys.map(k=>ann[k].seg)).size;
  document.getElementById("stats").textContent =
    "关键帧 " + keys.length + " · 段 " + segs + " · 当前帧 " + currentFrame();
  const wrap = document.getElementById("marks");
  wrap.innerHTML = "";
  for (const k of keys){
    const el = document.createElement("span");
    el.className = "mark" + (k === currentFrame() ? " cur" : "");
    el.textContent = k;
    el.onclick = () => { vid.currentTime = k / 29.97; refreshStats(); draw(); };
    wrap.appendChild(el);
  }
}

function resizeBox(d){
  fixedSize = Math.max(4, Math.min(80, fixedSize + d));
  document.getElementById("sizev").textContent = fixedSize;
}

cv.addEventListener("pointerdown", e => {
  if (e.button === 1) return;
  if (e.button === 2){
    draft = {mode:"pan", sx:e.clientX, sy:e.clientY, px:panX, py:panY};
    cv.setPointerCapture(e.pointerId);
    cv.style.cursor = "grabbing";
    return;
  }
  const [nx, ny] = toNat(e.clientX, e.clientY);
  const f = currentFrame();
  const a = ann[f];
  if (a){
    const [x1,y1,x2,y2] = a.bbox;
    const t = tol();
    const near = p => Math.abs(p - nx) < t && Math.abs(p - ny) < t;
    if (near(x2,y1) || near(x1,y2) || near(x2,y2) || near(x1,y1)){
      const cx = near(x2,y1)||near(x2,y2) ? 2 : 1;
      const cy = near(x1,y1)||near(x2,y1) ? 2 : 1;
      editBox = a; draft = {mode:"resize", x1, y1, x2, y2, cx, cy};
    } else if (nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2){
      editBox = a; draft = {mode:"move", x1, y1, x2, y2, dx: nx-x1, dy: ny-y1};
    } else {
      draft = {mode:"new", x1:nx, y1:ny, x2:nx, y2:ny};
    }
  } else {
    draft = {mode:"new", x1:nx, y1:ny, x2:nx, y2:ny};
  }
  cv.setPointerCapture(e.pointerId);
});

cv.addEventListener("pointermove", e => {
  if (!draft) return;
  if (draft.mode === "pan"){
    panX = draft.px + (e.clientX - draft.sx);
    panY = draft.py + (e.clientY - draft.sy);
    clampPan();
    applyZoom();
    return;
  }
  const [nx, ny] = toNat(e.clientX, e.clientY);
  if (draft.mode === "new"){
    draft.x2 = nx; draft.y2 = ny;
  } else if (draft.mode === "move"){
    const w = draft.x2 - draft.x1, h = draft.y2 - draft.y1;
    draft.x1 = nx - draft.dx; draft.y1 = ny - draft.dy;
    draft.x2 = draft.x1 + w; draft.y2 = draft.y1 + h;
  } else if (draft.mode === "resize"){
    if (draft.cx === 2) draft.x2 = nx; else draft.x1 = nx;
    if (draft.cy === 2) draft.y2 = ny; else draft.y1 = ny;
  }
  draw();
});

cv.addEventListener("pointerup", e => {
  if (!draft) return;
  if (draft.mode === "pan"){
    draft = null;
    cv.style.cursor = "crosshair";
    return;
  }
  const f = currentFrame();
  if (draft.mode === "new"){
    const dx = Math.abs(draft.x2 - draft.x1), dy = Math.abs(draft.y2 - draft.y1);
    if (dx < 6 && dy < 6){
      // click -> fixed box centered
      const h = fixedSize / 2;
      const b = [draft.x1 - h, draft.y1 - h, draft.x1 + h, draft.y1 + h];
      addKeyframe(f, b.map(v => Math.max(0, Math.min(NAT_W, v))));
    } else {
      const b = [Math.min(draft.x1,draft.x2), Math.min(draft.y1,draft.y2),
                 Math.max(draft.x1,draft.x2), Math.max(draft.y1,draft.y2)];
      addKeyframe(f, b.map(v => Math.max(0, Math.min(NAT_W, v))));
    }
  } else if (editBox){
    const b = [Math.min(draft.x1,draft.x2), Math.min(draft.y1,draft.y2),
               Math.max(draft.x1,draft.x2), Math.max(draft.y1,draft.y2)];
    editBox.bbox = b.map(v => Math.max(0, Math.min(NAT_W, v)));
    saveNow();
  }
  draft = null; editBox = null;
  refreshStats(); draw();
});

cv.addEventListener("wheel", e => {
  if (!zoomMode) return;
  e.preventDefault();
  const stage = document.getElementById("stage").getBoundingClientRect();
  const r = stageRect();
  const s = r.width / NAT_W;
  const nx = (e.clientX - r.left) / s;
  const ny = (e.clientY - r.top) / s;
  const factor = e.deltaY < 0 ? 1.3 : 1 / 1.3;
  zoom = Math.min(24, Math.max(1, zoom * factor));
  const ns = (stage.width * zoom) / NAT_W;
  panX = (e.clientX - stage.left) - nx * ns;
  panY = (e.clientY - stage.top) - ny * ns;
  applyZoom();
}, {passive: false});

cv.addEventListener("contextmenu", e => e.preventDefault());

document.addEventListener("keydown", e => {
  const k = e.key;
  if (k === " "){ e.preventDefault(); vid.paused ? vid.play() : vid.pause(); }
  else if (k === "ArrowLeft"){ vid.currentTime = Math.max(0, vid.currentTime - (e.shiftKey ? 10 : 1)/29.97); refreshStats(); draw(); }
  else if (k === "ArrowRight"){ vid.currentTime += (e.shiftKey ? 10 : 1)/29.97; refreshStats(); draw(); }
  else if (k === "1"){ vid.playbackRate = 0.5; }
  else if (k === "2"){ vid.playbackRate = 1; }
  else if (k === "3"){ vid.playbackRate = 2; }
  else if (k === "Delete"){ deleteKeyframe(currentFrame()); }
  else if (k === "z" || k === "Z"){
    zoomMode = !zoomMode;
    if (!zoomMode){ zoom = 1; panX = 0; panY = 0; }
    applyZoom();
  }
  else if (k === "Backspace"){ e.preventDefault(); undoLast(); }
  else if (k === "h" || k === "H"){ hintOn = !hintOn; document.getElementById("hintbox").checked = hintOn; draw(); }
  else if (k === "["){ resizeBox(-4); }
  else if (k === "]"){ resizeBox(4); }
  else if ((e.ctrlKey || e.metaKey) && k === "s"){ e.preventDefault(); saveNow(); }
});

vid.addEventListener("timeupdate", refreshStats);

fetch("/api/gt/" + SRC).then(r=>r.json()).then(d=>{
  for (const a of d.annotations){
    ann[a.frame] = {bbox: a.bbox, seg: a.seg};
    segCounter = Math.max(segCounter, a.seg + 1);
  }
  refreshStats();
});
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
