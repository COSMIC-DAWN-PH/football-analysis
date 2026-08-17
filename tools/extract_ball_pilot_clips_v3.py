"""Pilot v3: 5-second boxed video clips for AI labeling (aliyun video input
requires a minimum video duration).

For each item: frames spanning t-2.5 .. t+2.5 (5s, ~150 frames at 30fps),
red box on every frame at the detector bbox, candidate frame additionally
marked with a thick green box. H.264 1920x1080, no loop.

Usage:
  python tools/extract_ball_pilot_clips_v3.py --source demo4 --limit 20
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path("C:/Personal Profile/Profile/Video")
WINDOW = 2.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    src = args.source
    evaldir = ROOT / "eval" / "ball_crops" / src
    rows = [json.loads(l) for l in (evaldir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in rows}
    order = [l.strip() for l in (evaldir / "review_order.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [by_id[i] for i in order[: args.limit] if i in by_id]

    clips_dir = evaldir / "pilot" / "clips_v3"
    clips_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{src}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("cannot open video")

    for item in picked:
        iid = item["id"]
        t = item["t"]
        bbox = item["bbox"]
        start_f = max(0, round((t - WINDOW) * fps))
        end_f = round((t + WINDOW) * fps)
        candidate_f = round(t * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames = []
        for k in range(end_f - start_f + 1):
            ok, frame = cap.read()
            if not ok:
                break
            drawn = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            MIN_MARK = 60
            mx1 = max(0, cx - MIN_MARK // 2)
            my1 = max(0, cy - MIN_MARK // 2)
            mx2 = min(frame.shape[1], cx + MIN_MARK // 2)
            my2 = min(frame.shape[0], cy + MIN_MARK // 2)
            cv2.rectangle(drawn, (mx1, my1), (mx2, my2), (0, 0, 255), 5)
            cv2.drawMarker(drawn, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            if start_f + k == candidate_f:
                cv2.rectangle(drawn, (mx1, my1), (mx2, my2), (0, 255, 0), 8)
                cv2.drawMarker(drawn, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 36, 3)
            frames.append(drawn)

        if not frames:
            print(f"{iid}: no frames")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.mp4"
            w = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
            for fr in frames:
                w.write(fr)
            w.release()
            out_path = clips_dir / f"{iid}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)],
                check=True,
            )
        print(f"{iid}: {len(frames)} frames ({len(frames)/fps:.1f}s) -> {out_path.name}")

    cap.release()
    print("done")


if __name__ == "__main__":
    main()
