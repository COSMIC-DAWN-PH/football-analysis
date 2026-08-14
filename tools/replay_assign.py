"""Replay club assignment on an existing tracks JSONL using the current ClubAssigner.

Keeps detection/tracking (bboxes) fixed and re-runs only the color assignment
pipeline, producing a new tracks JSONL whose 'club' fields come from the
current club_assigner implementation. Used to evaluate Phase 1-3 changes
without re-running the full video pipeline.

Usage:
  python tools/replay_assign.py \
      --video output_videos/demo2-30s-test/demo2-30s-test-input.mp4 \
      --tracks output_videos/demo2-30s-test/raw/object_tracks.jsonl \
      --out output_videos/demo2-30s-test/raw/object_tracks_replayed.jsonl \
      --club1-name Maroon --club1-player 120,37,66 --club1-goalkeeper 80,80,80 \
      --club2-name Navy --club2-player 31,72,127 --club2-goalkeeper 80,80,80
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from club_assignment import Club, ClubAssigner


def _rgb(s: str):
    r, g, b = (int(v) for v in s.split(","))
    return (r, g, b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--club1-name", default="Maroon")
    parser.add_argument("--club1-player", type=_rgb, default=(120, 37, 66))
    parser.add_argument("--club1-goalkeeper", type=_rgb, default=(30, 30, 30))
    parser.add_argument("--club2-name", default="Navy")
    parser.add_argument("--club2-player", type=_rgb, default=(31, 72, 127))
    parser.add_argument("--club2-goalkeeper", type=_rgb, default=(48, 37, 68))
    parser.add_argument("--referee-color", type=_rgb, default=None,
                        help="Referee jersey reference color R,G,B")
    args = parser.parse_args()

    club1 = Club(args.club1_name, args.club1_player, args.club1_goalkeeper)
    club2 = Club(args.club2_name, args.club2_player, args.club2_goalkeeper)
    assigner = ClubAssigner(club1, club2, referee_color=args.referee_color)

    with open(args.tracks, encoding="utf-8") as f:
        lines = f.readlines()

    cap = cv2.VideoCapture(str(args.video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(lines) < n_frames:
        n_frames = len(lines)

    start = time.perf_counter()
    out_lines = []
    for fi in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        d = json.loads(lines[fi])
        d = assigner.assign_clubs(frame, d)
        # Mirror the live pipeline: tracks with no votes carry no club field
        for track_type in ("player", "goalkeeper"):
            for player_id, tr in d.get(track_type, {}).items():
                key = (track_type, player_id)
                if key not in assigner.club_by_track and key not in assigner.votes_by_track:
                    tr.pop("club", None)
                    tr.pop("club_color", None)
        out_lines.append(json.dumps(d, ensure_ascii=False))
    cap.release()
    elapsed = time.perf_counter() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"replayed {n_frames} frames in {elapsed:.1f}s "
          f"({elapsed / max(1, n_frames) * 1000:.1f} ms/frame)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
