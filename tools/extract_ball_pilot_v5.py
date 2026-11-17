"""Pilot v5: dynamic-box video clips + static context frames for AI labeling.

For each item: window around the candidate frame. The box FOLLOWS the target
across frames via LK optical flow (seeded at the candidate bbox, tracked
forward and backward), so the AI sees what the detector claimed actually is.
Also emits 7 full-res static frames with the tracked box drawn, plus reuses
the zoomed crop.

Usage:
  python tools/extract_ball_pilot_v5.py --source demo4 --limit 20 --window 1.25
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path("C:/Personal Profile/Profile/Video")
MIN_MARK = 60
PHOTO_OFFSETS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)


def lk_track(frames, bbox, cand_idx):
    """Return per-frame box (x1,y1,x2,y2) following the target.

    Seeds Shi-Tomasi points inside bbox at cand_idx, tracks forward and
    backward with pyramidal LK. Box = bounding rect of surviving points,
    clamped to frame and expanded to a visible minimum.
    """
    h, w = frames[cand_idx].shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(40, (max(x2 - x1, y2 - y1) * 3) // 2)
    x1s, y1s = max(0, cx - half), max(0, cy - half)
    x2s, y2s = min(w, cx + half), min(h, cy + half)
    if x2s - x1s < 16 or y2s - y1s < 16:
        x1s, y1s = max(0, cx - 32), max(0, cy - 32)
        x2s, y2s = min(w, cx + 32), min(h, cy + 32)

    def seed(f):
        gray = cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(gray[y1s:y2s, x1s:x2s], maxCorners=30,
                                      qualityLevel=0.05, minDistance=4)
        if pts is None:
            return np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        return pts.reshape(-1, 1, 2) + np.array([x1s, y1s], dtype=np.float32)

    def run_pass(order):
        boxes = [None] * len(order)
        prev_pts = seed(order[0])
        gray_prev = cv2.cvtColor(frames[order[0]], cv2.COLOR_BGR2GRAY)
        for i, f in enumerate(order):
            if i == 0:
                boxes[i] = None
                continue
            gray = cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray, prev_pts, None,
                winSize=(31, 31), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            if next_pts is None or status is None:
                break
            good = next_pts[status.flatten() == 1]
            if len(good) < 4:
                break
            px = good[:, 0, 0]
            py = good[:, 0, 1]
            bx1, by1 = float(max(0, px.min() - 8)), float(max(0, py.min() - 8))
            bx2, by2 = float(min(w, px.max() + 8)), float(min(h, py.max() + 8))
            if bx2 - bx1 < 8 or by2 - by1 < 8:
                break
            boxes[i] = (bx1, by1, bx2, by2)
            gray_prev = gray
            prev_pts = good.reshape(-1, 1, 2)
        return boxes

    boxes = [None] * len(frames)
    fwd = run_pass(list(range(cand_idx, len(frames))))
    for i, b in enumerate(fwd):
        if b is not None:
            boxes[cand_idx + i] = b
    rev = run_pass(list(range(cand_idx, -1, -1)))
    for i, b in enumerate(rev):
        if b is not None:
            boxes[cand_idx - i] = b
    boxes[cand_idx] = (x1, y1, x2, y2)
    last = (x1, y1, x2, y2)
    for i in range(len(boxes)):
        boxes[i] = boxes[i] if boxes[i] is not None else last
        last = boxes[i]
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--window", type=float, default=1.25, help="half-window seconds")
    parser.add_argument("--min-duration", type=float, default=2.5,
                        help="minimum clip duration; early frames extend the end")
    args = parser.parse_args()

    src = args.source
    evaldir = ROOT / "eval" / "ball_crops" / src
    rows = [json.loads(l) for l in (evaldir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in rows}
    order = [l.strip() for l in (evaldir / "review_order.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [by_id[i] for i in order[: args.limit] if i in by_id]

    clips_dir = evaldir / "pilot" / "clips_v5"
    frames_dir = evaldir / "pilot" / "frames_v5"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{src}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("cannot open video")

    manifest = []
    for item in picked:
        iid = item["id"]
        t = item["t"]
        bbox = item["bbox"]
        cand_f = round(t * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_f = max(0, round((t - args.window) * fps))
        end_f = min(total_frames - 1, round((t + args.window) * fps))
        min_frames = round(args.min_duration * fps)
        if end_f - start_f + 1 < min_frames:
            end_f = min(total_frames - 1, start_f + min_frames - 1)
        if end_f - start_f + 1 < min_frames:
            start_f = max(0, end_f - min_frames + 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames = []
        for k in range(end_f - start_f + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cand_idx = cand_f - start_f

        boxes = lk_track(frames, bbox, cand_idx)
        h, w = frames[0].shape[:2]

        def draw_box(frame, box, color, thick):
            x1, y1, x2, y2 = (int(v) for v in box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            mx1 = max(0, cx - MIN_MARK // 2)
            my1 = max(0, cy - MIN_MARK // 2)
            mx2 = min(w, cx + MIN_MARK // 2)
            my2 = min(h, cy + MIN_MARK // 2)
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, thick)
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 24, 2)

        drawn = []
        for i, fr in enumerate(frames):
            d = fr.copy()
            if i == cand_idx:
                draw_box(d, boxes[i], (0, 255, 0), 8)
            else:
                draw_box(d, boxes[i], (0, 0, 255), 5)
            drawn.append(d)

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.mp4"
            wr = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for d in drawn:
                wr.write(d)
            wr.release()
            out_path = clips_dir / f"{iid}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)],
                check=True,
            )

        photos = []
        for off in PHOTO_OFFSETS:
            f = max(0, round((t + off) * fps))
            if start_f <= f <= end_f:
                d = frames[f - start_f].copy()
                draw_box(d, boxes[f - start_f], (0, 255, 0) if f == cand_f else (0, 0, 255), 6)
                name = f"{iid}_t{off:+.2f}.jpg"
                cv2.imwrite(str(frames_dir / name), d, [cv2.IMWRITE_JPEG_QUALITY, 88])
                photos.append(f"pilot/frames_v5/{name}")

        manifest.append({
            "id": iid, "category": item["category"], "frame": item["frame"],
            "t": t, "conf": item["conf"], "clip": f"pilot/clips_v5/{iid}.mp4",
            "photos": photos, "crop": item["crop"],
        })
        print(f"{iid}: {len(frames)} frames ({len(frames)/fps:.1f}s) box-tracked, {len(photos)} photos")

    cap.release()
    (evaldir / "pilot" / "manifest_v5.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False, separators=(",", ":")) for m in manifest) + "\n",
        encoding="utf-8",
    )
    print("done ->", evaldir / "pilot" / "manifest_v5.jsonl")


if __name__ == "__main__":
    main()
