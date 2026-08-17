"""Generate 3x3 motion-strip grids from the boxed pilot clips (clips_v5).

For each item: sample 9 frames evenly across the 2.5s clip, resize each to
640x360, arrange as 3x3 grid (1920x1080) -> pilot/strips_v5/<id>.png
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

ROOT = Path(__file__).resolve().parent.parent
CELL_W, CELL_H = 640, 360
GRID = 3


def make_strip(args):
    iid, clip_path, out_png = args
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False
    frames = []
    for k in range(GRID * GRID):
        f = min(total - 1, round((total - 1) * k / (GRID * GRID - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, (CELL_W, CELL_H)))
    cap.release()
    if len(frames) < GRID * GRID:
        while len(frames) < GRID * GRID:
            frames.append(frames[-1])
    rows = [cv2.hconcat(frames[r * GRID:(r + 1) * GRID]) for r in range(GRID)]
    grid = cv2.vconcat(rows)
    cv2.imwrite(str(out_png), grid, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return True


def main() -> None:
    evaldir = ROOT / "eval" / "ball_crops" / "demo4"
    manifest = [json.loads(l) for l in (evaldir / "pilot" / "manifest_v5.jsonl").read_text(encoding="utf-8").splitlines()]
    clips = evaldir / "pilot" / "clips_v5"
    out_dir = evaldir / "pilot" / "strips_v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for m in manifest:
        iid = m["id"]
        out_png = out_dir / f"{iid}.png"
        if out_png.exists():
            continue
        jobs.append((iid, clips / f"{iid}.mp4", out_png))

    with Pool(8) as pool:
        done = 0
        for ok in pool.imap_unordered(make_strip, jobs, chunksize=8):
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    print("strips done:", len(manifest))


if __name__ == "__main__":
    main()
