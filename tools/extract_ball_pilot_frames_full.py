"""Pilot v2: dump full-resolution (1920x1080) JPEG frame sequences for the
first N chronological review items, for multi-image AI labeling.

For each item: 15 frames spanning t-0.25 .. t+0.25, red bbox drawn on the
candidate frame (f08), no boxes on other frames so motion stays visible.

Usage:
  python tools/extract_ball_pilot_frames_full.py --source demo4 --limit 20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path("C:/Personal Profile/Profile/Video")
WINDOW = 0.25
N_FRAMES = 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args()

    src = args.source
    evaldir = ROOT / "eval" / "ball_crops" / src
    rows = [json.loads(l) for l in (evaldir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in rows}
    order = [l.strip() for l in (evaldir / "review_order.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [by_id[i] for i in order[: args.limit] if i in by_id]

    frames_dir = evaldir / "pilot" / "frames"
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
        center = (N_FRAMES - 1) // 2
        rel = []
        for k in range(N_FRAMES):
            off = -WINDOW + (2 * WINDOW) * k / (N_FRAMES - 1)
            f = max(0, round((t + off) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            if k == center:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            name = f"{iid}_f{k + 1:02d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            rel.append(f"pilot/frames/{name}")
        manifest.append({
            "id": iid,
            "category": item["category"],
            "frame": item["frame"],
            "t": t,
            "conf": item["conf"],
            "frames": rel,
            "crop": item["crop"],
        })
        print(f"{iid}: {len(rel)} frames")

    cap.release()
    (evaldir / "pilot" / "manifest_v2.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False, separators=(",", ":")) for o in manifest) + "\n",
        encoding="utf-8",
    )
    print(f"done: {len(manifest)} items -> {evaldir / 'pilot' / 'manifest_v2.jsonl'}")


if __name__ == "__main__":
    main()
