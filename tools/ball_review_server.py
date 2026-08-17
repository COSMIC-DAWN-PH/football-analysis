"""Local ball review web UI. Serves contact sheets with per-cell label buttons
and the original video frame (0.5s window) next to the zoomed crop.

Start:  python tools/ball_review_server.py [--port 8100]
Open:   http://127.0.0.1:8100
"""
import argparse
import io
import json
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template_string, request, send_file

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval" / "ball_crops"
VIDEO_DIR = Path("C:/Personal Profile/Profile/Video")
TRACKS_DIR = ROOT / "output_videos"

SOURCES = ["demo4", "demo1", "demo3", "demo2", "raw1", "raw2"]
LABELS = {"ball", "not_ball", "null"}
REASONS = ["shoe", "sock", "line", "light", "head", "hand", "penalty_spot",
           "debris", "goal", "corner_flag", "referee", "player", "other"]

app = Flask(__name__)
_cache = {}
_lock = threading.Lock()
FRAME_CACHE = {}
FOLLOW_WINDOW = 120


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


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _tracks(src):
    path = TRACKS_DIR / src / "ball" / "ball_tracks.jsonl"
    if not path.is_file():
        return None
    with _lock:
        item = _cache.get("tracks:" + src)
        if item and time.time() - item["t"] < 300:
            return item
        frames = {}
        f0 = t0 = f1 = t1 = None
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            frames[d["frame"]] = {
                "t": d["t"],
                "cands": d["candidates"],
                "track": d["track"]["bbox"] if d.get("track") else None,
            }
            if f0 is None:
                f0, t0 = d["frame"], d["t"]
            elif f1 is None:
                f1, t1 = d["frame"], d["t"]
        fps = None
        if f1 is not None and f1 != f0 and t1 is not None and t0 is not None:
            fps = (f1 - f0) / max(t1 - t0, 1e-9)
        item = {"frames": frames, "fps": fps, "t": time.time()}
        _cache["tracks:" + src] = item
        return item


def _pick(frame_rec, prev, max_d):
    if frame_rec is None:
        return None
    best, best_d = None, None
    px, py = _center(prev)
    for c in frame_rec["cands"]:
        cb = [c[0], c[1], c[2], c[3]]
        cx, cy = _center(cb)
        dd = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if best_d is None or dd < best_d:
            best, best_d = cb, dd
    if best is None or best_d > max_d:
        return None
    return best


def _follow_path(row, tracks):
    frames = tracks["frames"]
    f0 = row["frame"]
    bbox = list(row["bbox"])
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw >= 1918 or bh >= 1078:
        return {}
    max_d = max(3.0 * max(bw, bh), 60.0)
    follow = {f0: bbox}
    for direction in (-1, 1):
        last = bbox
        missed = 0
        for f in range(f0 + direction, f0 + direction * (FOLLOW_WINDOW + 1), direction):
            pick = _pick(frames.get(f), last, max_d)
            if pick is None:
                if missed >= 4:
                    break
                missed += 1
                follow[f] = last
                continue
            missed = 0
            last = pick
            follow[f] = pick
    return follow


def _draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=8, gap=5):
    x = x1
    while x < x2:
        nx = min(x + dash, x2)
        cv2.line(img, (x, y1), (nx, y1), color, thickness, cv2.LINE_AA)
        x = nx + gap
    x = x1
    while x < x2:
        nx = min(x + dash, x2)
        cv2.line(img, (x, y2), (nx, y2), color, thickness, cv2.LINE_AA)
        x = nx + gap
    y = y1
    while y < y2:
        ny = min(y + dash, y2)
        cv2.line(img, (x1, y), (x1, ny), color, thickness, cv2.LINE_AA)
        y = ny + gap
    y = y1
    while y < y2:
        ny = min(y + dash, y2)
        cv2.line(img, (x2, y), (x2, ny), color, thickness, cv2.LINE_AA)
        y = ny + gap


def _draw_frame(frame, row):
    h, w = frame.shape[:2]
    bbox = row["bbox"]
    x1 = max(0, int(round(bbox[0]))); y1 = max(0, int(round(bbox[1])))
    x2 = min(w, int(round(bbox[2]))); y2 = min(h, int(round(bbox[3])))
    if x2 - x1 >= w - 2 or y2 - y1 >= h - 2:
        return frame
    m = max(24, int(max(x2 - x1, y2 - y1) * 2.5))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx1 = max(0, cx - m); ry1 = max(0, cy - m)
    rx2 = min(w, cx + m); ry2 = min(h, cy + m)
    if rx2 - rx1 >= 12 and ry2 - ry1 >= 12:
        tile = frame[ry1:ry2, rx1:rx2].copy()
        tw0, th0 = tile.shape[1], tile.shape[0]
        scale = 280 / max(th0, tw0)
        if abs(scale - 1.0) > 0.02:
            tile = cv2.resize(
                tile, (max(1, int(tw0 * scale)), max(1, int(th0 * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        th, tw = tile.shape[0], tile.shape[1]
        ox = w - tw - 14
        oy = 14
        cv2.rectangle(frame, (ox - 5, oy - 5), (ox + tw + 5, oy + th + 5), (255, 255, 255), 2)
        frame[oy:oy + th, ox:ox + tw] = tile
        bx1 = int((x1 - rx1) * scale); by1 = int((y1 - ry1) * scale)
        bx2 = int((x2 - rx1) * scale); by2 = int((y2 - ry1) * scale)
        _draw_dashed_rect(frame, ox + bx1 - 1, oy + by1 - 1,
                          ox + bx2 + 1, oy + by2 + 1, (0, 0, 255))
        cv2.putText(frame, "zoom x%.1f" % scale, (ox, oy + th + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    pad = 1
    _draw_dashed_rect(frame, max(0, x1 - pad), max(0, y1 - pad),
                      min(w - 1, x2 + pad), min(h - 1, y2 + pad), (0, 0, 255))
    cv2.putText(frame, row["id"], (max(2, x1), max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


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


@app.get("/api/vidinfo/<src>/<iid>")
def vidinfo(src, iid):
    item = _load(src)
    row = item["by_id"].get(iid)
    if row is None:
        return jsonify({"error": "unknown id"}), 404
    tracks = _tracks(src)
    if tracks is None or not tracks["fps"]:
        return jsonify({"fps": None, "follow": {}})
    return jsonify({"fps": tracks["fps"], "follow": _follow_path(row, tracks)})


@app.get("/api/frame/<src>/<iid>")
def api_frame(src, iid):
    key = (src, iid)
    data = FRAME_CACHE.get(key)
    if data is None:
        item = _load(src)
        row = item["by_id"].get(iid)
        if row is None:
            return jsonify({"error": "unknown id"}), 404
        path = VIDEO_DIR / f"{src}.mp4"
        if not path.is_file():
            return jsonify({"error": "no video"}), 404
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return jsonify({"error": "frame read failed"}), 502
        ok2, buf = cv2.imencode(".jpg", _draw_frame(frame, row),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok2:
            return jsonify({"error": "encode failed"}), 502
        data = buf.tobytes()
        if len(FRAME_CACHE) >= 96:
            FRAME_CACHE.clear()
        FRAME_CACHE[key] = data
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


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
#frameov{position:fixed;inset:0;background:rgba(0,0,0,.95);display:none;flex-direction:column;align-items:center;justify-content:center;z-index:30;overflow:auto;cursor:pointer}
#frameov img{display:block;max-width:94vw;max-height:82vh;margin:auto;border:1px solid #555;cursor:zoom-in}
#frameov img.zoomed{max-width:none;max-height:none;cursor:zoom-out}
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
  <br>两段式标注：<kbd>1</kbd>=是球 <kbd>2</kbd>=不是 → 再选原因 <kbd>q</kbd>鞋 <kbd>w</kbd>袜 <kbd>e</kbd>场地线 <kbd>r</kbd>灯 <kbd>t</kbd>头 <kbd>y</kbd>手/手套 <kbd>u</kbd>点球点 <kbd>i</kbd>场上杂物 <kbd>o</kbd>球门/球网 <kbd>p</kbd>角旗 <kbd>a</kbd>裁判/装备 <kbd>d</kbd>球员/球衣 <kbd>s</kbd>其他 <kbd>Enter</kbd>跳过原因 <kbd>Esc</kbd>返回 <kbd>3</kbd>=无法判断
  <br>其他：<kbd>v</kbd>重播视频 <kbd>l</kbd>循环 <kbd>c</kbd>查看原帧（再按 <kbd>c</kbd>/<kbd>Esc</kbd>/点空白返回；点图可放大缩回） <kbd>[</kbd><kbd>]</kbd>已标间切换 <kbd>g</kbd>输入id/序号跳转 <kbd>Backspace</kbd>回退上一标 <kbd>0</kbd>清除 <kbd>Esc</kbd>关闭
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
    <button class="btn" onclick="toggleFrameImg()">查看原帧 (c)</button>
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
      <button class="btn reason" onclick="setModal('not_ball','player')">球员/球衣(d)</button>
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
<div id="frameov" onclick="ovBack(event)">
  <img id="frameimg" alt="" onclick="this.classList.toggle('zoomed')">
  <div><button class="btn">返回标注界面 (Esc / 点击空白)</button></div>
</div>
<script>
const SRC = "{{src}}";
const COLS = 5, ROWS = 4;
const RZ = {shoe:"鞋", sock:"袜", line:"场地线", light:"灯", head:"头", hand:"手/手套", penalty_spot:"点球点", debris:"场上杂物", goal:"球门/球网", corner_flag:"角旗", referee:"裁判/装备", player:"球员/球衣", other:"其他"};
const CLIP = 0.25;
let data = null, idx = 0, cur = null, orderSeq = [], sheetsSeq = [], undoStack = [], curInfo = null;
let curT = 0, curBbox = null, stage = 1, fps = null, followMap = null;

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
  curT = info.t;
  curBbox = info.bbox;
  const url = "/video/" + SRC;
  if (vid.getAttribute("src") !== url) {
    vid.setAttribute("src", url);
  }
  fetch("/api/vidinfo/" + SRC + "/" + info.id)
    .then(r => r.json())
    .then(d => {
      fps = d.fps || null;
      followMap = d.fps ? (d.follow || {}) : null;
      updateBbox();
    })
    .catch(() => { fps = null; followMap = null; updateBbox(); });
  updateBbox();
  replay();
}

function getFollowBbox(){
  if (fps && followMap) {
    const vid = document.getElementById("vid");
    if (!vid || !isFinite(vid.currentTime)) return null;
    return followMap[Math.round(vid.currentTime * fps)] || null;
  }
  const b = curBbox;
  if (b && !(b[0] <= 0.5 && b[1] <= 0.5 && b[2] >= 1919.5 && b[3] >= 1079.5)) return b;
  return null;
}

function updateBbox(){
  const ov = document.getElementById("bboxov");
  if (!ov) return;
  const b = getFollowBbox();
  if (b) {
    ov.style.display = "block";
    ov.style.left = (b[0] / 1920 * 100) + "%";
    ov.style.top = (b[1] / 1080 * 100) + "%";
    ov.style.width = ((b[2] - b[0]) / 1920 * 100) + "%";
    ov.style.height = ((b[3] - b[1]) / 1080 * 100) + "%";
  } else {
    ov.style.display = "none";
  }
}

function toggleFrameImg(){
  const f = document.getElementById("frameov");
  if (f.style.display !== "flex") {
    f.style.display = "flex";
    document.getElementById("vid").pause();
  } else {
    f.style.display = "none";
  }
}

function ovBack(e){
  if (!e.target || e.target.tagName !== "IMG") toggleFrameImg();
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
    updateBbox();
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
    const fimg = document.getElementById("frameimg");
    fimg.src = "/api/frame/" + SRC + "/" + info.id;
    fimg.classList.remove("zoomed");
    document.getElementById("frameov").style.display = "none";
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
  document.getElementById("frameov").style.display = "none";
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
    else if (k === "d") setModal("not_ball", "player");
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
  else if (k === "c") toggleFrameImg();
  else if (k === "escape") {
    const fo = document.getElementById("frameov");
    if (fo.style.display === "flex") fo.style.display = "none";
    else closeModal();
  }
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
