"""Sanity-check flagged-referee tracks: are they actually yellow?

Samples N flagged player tracks from a replayed tracks file, computes their
torso hue at a few frames from the source video, and reports the hue
distribution. Flagged tracks should be yellow (hue 15-45); other hues point
at false flags (e.g. goalkeepers).

Usage:
  python tools/check_flagged_hues.py --tracks output_videos/raw1/raw/object_tracks_referee_replay.jsonl --video C:/.../raw1.mp4 --sample 30
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from club_assignment.club_assigner import ClubAssigner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--frames-per-track", type=int, default=5)
    args = parser.parse_args()

    lines = args.tracks.read_text(encoding="utf-8").splitlines()
    flagged = collections.defaultdict(list)
    for fi, ln in enumerate(lines):
        d = json.loads(ln)
        for tid, tr in d.get("player", {}).items():
            if tr.get("referee"):
                flagged[tid].append(fi)
    if not flagged:
        print("no flagged tracks")
        return

    rng = random.Random(7)
    keys = rng.sample(sorted(flagged, key=lambda k: -len(flagged[k])), min(args.sample, len(flagged)))

    assigner = ClubAssigner.__new__(ClubAssigner)
    cap = cv2.VideoCapture(str(args.video))
    hues_by_track = {}
    for tid in keys:
        frames = sorted(flagged[tid])
        step = max(1, len(frames) // args.frames_per_track)
        hues = []
        for fi in frames[::step][: args.frames_per_track]:
            d = json.loads(lines[fi])
            tr = d["player"].get(tid)
            if tr is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            stats = assigner.extract_jersey_stats(frame, tr["bbox"])
            if stats is not None and stats[3] >= 30:
                hues.append(float(stats[0]))
        if hues:
            hues_by_track[tid] = {"n": len(flagged[tid]), "median_hue": round(float(np.median(hues)), 0), "min": round(min(hues), 0), "max": round(max(hues), 0)}
    cap.release()

    zone = collections.Counter()
    for tid, info in sorted(hues_by_track.items(), key=lambda kv: -kv[1]["n"]):
        mh = info["median_hue"]
        if 15 <= mh < 45:
            zone["yellow_referee"] += 1
        elif mh < 15 or mh >= 135:
            zone["red_zone"] += 1
        else:
            zone["other"] += 1
        print(f"player{tid}: {info}")
    print(f"zone summary: {dict(zone)}")


if __name__ == "__main__":
    main()
