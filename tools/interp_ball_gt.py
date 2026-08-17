"""Expand keyframe ball annotations into per-frame GT.

Reads eval/ball_gt/<src>/annotations.jsonl (keyframes with seg), linearly
interpolates bbox between consecutive keyframes of the SAME segment when the
gap <= MAX_GAP_FRAMES (default 2s), and writes per-frame GT to
eval/ball_gt/<src>/frames_gt.jsonl.

Usage:
  python tools/interp_ball_gt.py [--src demo4] [--max-gap 60]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--max-gap", type=int, default=60, help="max frames to interpolate across (default 60 = 2s)")
    args = parser.parse_args()

    gt_dir = ROOT / "eval" / "ball_gt" / args.src
    ann_path = gt_dir / "annotations.jsonl"
    if not ann_path.is_file():
        print(f"no annotations: {ann_path}")
        sys.exit(1)

    keys = {}
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        keys[int(d["frame"])] = d

    frames = sorted(keys)
    out = {}
    for f in frames:
        out[f] = {"frame": f, "t": keys[f]["t"], "bbox": keys[f]["bbox"],
                  "seg": keys[f]["seg"], "interp": False}

    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        ka, kb = keys[a], keys[b]
        if ka["seg"] != kb["seg"]:
            continue
        if b - a > args.max_gap:
            continue
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            out[k] = {
                "frame": k,
                "t": round(k / 29.97, 4),
                "bbox": [
                    ka["bbox"][0] + (kb["bbox"][0] - ka["bbox"][0]) * t,
                    ka["bbox"][1] + (kb["bbox"][1] - ka["bbox"][1]) * t,
                    ka["bbox"][2] + (kb["bbox"][2] - ka["bbox"][2]) * t,
                    ka["bbox"][3] + (kb["bbox"][3] - ka["bbox"][3]) * t,
                ],
                "seg": ka["seg"],
                "interp": True,
            }

    out_path = gt_dir / "frames_gt.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(out[f], ensure_ascii=False, separators=(",", ":")) for f in sorted(out)) + "\n",
        encoding="utf-8",
    )
    n_interp = sum(1 for v in out.values() if v["interp"])
    print(f"{args.src}: {len(out)} frames GT ({len(frames)} keyframes, {n_interp} interpolated) -> {out_path}")


if __name__ == "__main__":
    main()
