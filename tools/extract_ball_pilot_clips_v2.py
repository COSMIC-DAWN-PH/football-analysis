"""Pilot v3: boxed video clips for the first N chronological review items.

15 full-res frames spanning t-0.25..t+0.25, red box drawn on EVERY frame at
the detector bbox, candidate frame (f08) additionally marked with a thick
green box. Clip loops 3 times (45 frames, ~1.5s) then transcoded to H.264.

Usage:
  python tools/extract_ball_pilot_clips_v2.py --source demo4 --limit 20
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
WINDOW = 0.25
N_FRAMES = 15
LOOPS = 3


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

    clips_dir = evaldir / "pilot" / "clips_v2"
    clips_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{src}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("cannot open video")

    for item in picked:
        iid = item["id"]
        t = item["t"]
        bbox = item["bbox"]
        center = (N_FRAMES - 1) // 2

        start_f = max(0, round((t - WINDOW) * fps))
        frames = []
        for k in range(N_FRAMES):
            f = start_f + k
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            drawn = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 0, 255), 3)
            if k == center:
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 5)
            frames.append(drawn)

        if not frames:
            print(f"{iid}: no frames")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.mp4"
            w = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
            for _ in range(LOOPS):
                for fr in frames:
                    w.write(fr)
            w.release()
            out_path = clips_dir / f"{iid}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)],
                check=True,
            )
        print(f"{iid}: {len(frames)} frames x{LOOPS} -> {out_path.name}")

    cap.release()
    print("done")


if __name__ == "__main__":
    main()
