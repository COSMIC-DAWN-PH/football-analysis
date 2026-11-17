"""Pilot: build per-candidate video clips (+/-0.25s) and context frame strips
for the first N chronological review items of a source, for AI-labeling pilot.

Usage:
  python tools/extract_ball_pilot_frames.py --source demo4 --limit 20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path("C:/Personal Profile/Profile/Video")
CLIP_WINDOW = 0.25
STRIP_OFFSETS = (-0.25, -0.125, 0.0, 0.125, 0.25)
STRIP_FRAME_W = 640
STRIP_FRAME_H = 360


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    src = args.source
    evaldir = ROOT / "eval" / "ball_crops" / src
    manifest_path = evaldir / "candidates.jsonl"
    order_path = evaldir / "review_order.txt"
    video_path = VIDEO_DIR / f"{src}.mp4"

    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in rows}
    order = [l.strip() for l in order_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [by_id[i] for i in order[: args.limit] if i in by_id]

    pilot = evaldir / "pilot"
    clips_dir = pilot / "clips"
    ctx_dir = pilot / "context"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError(f"cannot open video {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = []
    for item in picked:
        iid = item["id"]
        t = item["t"]
        bbox = item["bbox"]
        start_f = max(0, round((t - CLIP_WINDOW) * fps))
        end_f = round((t + CLIP_WINDOW) * fps)
        n_frames = end_f - start_f + 1

        clip_path = clips_dir / f"{iid}.mp4"
        writer = None
        for codec in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            w = cv2.VideoWriter(str(clip_path), fourcc, fps, (frame_w, frame_h))
            if w.isOpened():
                writer = w
                break
        if writer is None:
            raise RuntimeError(f"no usable codec for {iid}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_cache = {}
        for k in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            f = start_f + k
            frame_cache[f] = frame
            drawn = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)
            writer.write(drawn)
        writer.release()

        strip = []
        for off in STRIP_OFFSETS:
            f = max(0, round((t + off) * fps))
            if f not in frame_cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    break
                frame_cache[f] = frame
            small = cv2.resize(frame_cache[f], (STRIP_FRAME_W, STRIP_FRAME_H))
            drawn = small.copy()
            x1 = int(bbox[0] * STRIP_FRAME_W / frame_w)
            y1 = int(bbox[1] * STRIP_FRAME_H / frame_h)
            x2 = int(bbox[2] * STRIP_FRAME_W / frame_w)
            y2 = int(bbox[3] * STRIP_FRAME_H / frame_h)
            color = (0, 255, 0) if off == 0.0 else (0, 0, 255)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            strip.append(drawn)
        if strip:
            strip_img = cv2.hconcat(strip)
            h, w = strip_img.shape[:2]
            strip_img = cv2.resize(strip_img, (w // 2, h // 2))
            ctx_path = ctx_dir / f"{iid}.png"
            cv2.imwrite(str(ctx_path), strip_img)

        out.append({
            "id": iid,
            "category": item["category"],
            "frame": item["frame"],
            "t": t,
            "conf": item["conf"],
            "clip": f"pilot/clips/{iid}.mp4",
            "context": f"pilot/context/{iid}.png",
            "crop": item["crop"],
        })
        print(f"{iid}: clip {n_frames} frames -> {clip_path.name}")

    cap.release()
    (pilot / "manifest.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False, separators=(",", ":")) for o in out) + "\n",
        encoding="utf-8",
    )
    print(f"done: {len(out)} items -> {pilot / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
