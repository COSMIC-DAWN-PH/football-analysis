"""Evaluate team color classification against a manually labeled crop set.

Metrics (same script, same data, before/after every phase):
  - confusion matrix: Maroon->Navy, Navy->Maroon (per labeled track)
  - balanced accuracy: mean of per-team recall
  - coverage: fraction of player track-frames with a club assigned
  - temporal stability: per-track club flip count
  - optional: timing of assign_clubs on sampled frames

Usage:
  python tools/eval_team_colors.py \
      --labels eval/labeled_crops.json \
      --tracks output_videos/demo2-30s-test/raw/object_tracks.jsonl \
      [--video output_videos/demo2-30s-test/demo2-30s-test-input.mp4] \
      [--timing-frames 10]
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--timing-frames", type=int, default=0)
    parser.add_argument(
        "--use-auto",
        action="store_true",
        help="treat auto_label (hue-bimodal suggestion) as ground truth where manual_label is unset",
    )
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    with open(args.tracks, encoding="utf-8") as f:
        track_lines = f.readlines()

    def assigned_club(frame_idx: int, track_id: int):
        try:
            d = json.loads(track_lines[frame_idx])
        except IndexError:
            return None
        player = d.get("player", {})
        tr = player.get(str(track_id)) or player.get(track_id)
        if tr is None:
            return None
        return tr.get("club")

    # ground-truth per track: use manually labeled crops, one label per track
    track_truth = {}
    ambiguous = []
    for name, meta in labels["crops"].items():
        manual = meta.get("manual_label")
        if manual is None and args.use_auto:
            manual = meta.get("auto_label")
        pid = meta["track_id"]
        if manual in ("Maroon", "Navy"):
            prev = track_truth.get(pid)
            if prev is not None and prev != manual:
                ambiguous.append((pid, prev, manual))
            track_truth[pid] = manual
        elif manual:
            raise ValueError(f"invalid manual_label {manual!r} for {name}")

    if ambiguous:
        print("WARNING: conflicting manual labels within track:", ambiguous)

    # confusion matrix: truth -> pipeline assignment
    conf = collections.Counter()
    per_track = {}
    for pid, truth in sorted(track_truth.items(), key=lambda kv: int(kv[0])):
        frame_idx = next(
            m["frame"] for m in labels["crops"].values() if m["track_id"] == pid
        )
        assigned = assigned_club(frame_idx, pid)
        per_track[pid] = {"truth": truth, "assigned": assigned,
                          "correct": truth == assigned}
        conf[(truth, assigned)] += 1

    n_correct = sum(1 for v in per_track.values() if v["correct"])
    n_total = len(per_track)
    recalls = {}
    for team in ("Maroon", "Navy"):
        team_tracks = [v for v in per_track.values() if v["truth"] == team]
        recalls[team] = (
            sum(1 for v in team_tracks if v["correct"]) / len(team_tracks)
            if team_tracks else None
        )
    balanced = (
        sum(r for r in recalls.values() if r is not None)
        / sum(1 for r in recalls.values() if r is not None)
    )

    # coverage: fraction of player track-frame entries with club assigned
    total_entries = 0
    club_entries = 0
    for line in track_lines:
        d = json.loads(line)
        for tr in d.get("player", {}).values():
            total_entries += 1
            if tr.get("club") is not None:
                club_entries += 1
    coverage = club_entries / total_entries if total_entries else 0.0

    # temporal stability: per-track club flips across frames
    track_clubs = collections.defaultdict(list)
    for line in track_lines:
        d = json.loads(line)
        for pid, tr in d.get("player", {}).items():
            c = tr.get("club")
            if c is not None:
                track_clubs[pid].append(c)
    flips = 0
    stable = 0
    for pid, seq in track_clubs.items():
        track_flips = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        flips += track_flips
        if track_flips == 0:
            stable += 1

    report = {
        "confusion": {f"{t}->{a}": c for (t, a), c in sorted(conf.items())},
        "per_track": per_track,
        "accuracy": n_correct / n_total if n_total else None,
        "balanced_accuracy": balanced,
        "recalls": recalls,
        "n_labeled_tracks": n_total,
        "coverage": coverage,
        "temporal_flips": flips,
        "tracks_never_flipping": stable,
        "tracks_with_club": len(track_clubs),
    }

    # optional timing of the live assigner
    if args.video and args.timing_frames > 0:
        import cv2
        from club_assignment import Club, ClubAssigner

        labels_meta = list(labels["crops"].values())
        # reconstruct club config from any manual labels; colors are the demo defaults
        club1 = Club("Maroon", (120, 37, 66), (80, 80, 80))
        club2 = Club("Navy", (31, 72, 127), (80, 80, 80))
        assigner = ClubAssigner(club1, club2)
        cap = cv2.VideoCapture(str(args.video))
        start = time.perf_counter()
        for fi in range(args.timing_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            d = json.loads(track_lines[fi])
            assigner.club_by_track.clear()
            assigner.assign_clubs(frame, d)
        elapsed = time.perf_counter() - start
        report["assign_clubs_ms_per_frame"] = elapsed / args.timing_frames * 1000
        cap.release()

    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
