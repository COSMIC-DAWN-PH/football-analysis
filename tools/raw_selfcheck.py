"""Self-consistency report for replayed raw tracks (no labels needed).

Sanity checks the referee/club/goalkeeper assignment across long footage:
  - player tracks flagged referee (frames, median jersey hue)
  - referee-class tracks restored to clubs
  - goalkeeper-class club distribution
  - club coverage and per-track club flips

Usage:
  python tools/raw_selfcheck.py --tracks output_videos/raw1/raw/object_tracks_referee_replay.jsonl --name raw1
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    lines = args.tracks.read_text(encoding="utf-8").splitlines()

    track_frames = collections.Counter()
    flagged = collections.Counter()
    club_frames = collections.defaultdict(collections.Counter)
    gk_clubs = collections.Counter()
    track_club_seq = collections.defaultdict(list)
    track_type_of = {}

    for ln in lines:
        d = json.loads(ln)
        for t in ("player", "referee", "goalkeeper"):
            for tid, tr in d.get(t, {}).items():
                key = (t, tid)
                track_type_of[key] = t
                track_frames[key] += 1
                if tr.get("referee"):
                    flagged[key] += 1
                club = tr.get("club")
                if club is not None:
                    club_frames[key][club] += 1
                    track_club_seq[key].append(club)
                    if t == "goalkeeper":
                        gk_clubs[club] += 1

    flagged_tracks = {
        f"{t}{tid}": {"frames": n, "total": track_frames[(t, tid)]}
        for (t, tid), n in sorted(flagged.items(), key=lambda kv: -kv[1])
        if n >= 5
    }
    restored = {
        f"{t}{tid}": {"club": max(club_frames[(t, tid)], key=club_frames[(t, tid)].get),
                      "frames": track_frames[(t, tid)]}
        for (t, tid) in sorted(club_frames.keys(), key=lambda k: -track_frames[k])
        if t == "referee" and club_frames[(t, tid)]
    }

    # Coverage: player track-frames with a club
    player_total = sum(n for (t, _), n in track_frames.items() if t == "player")
    player_club = sum(
        sum(club_frames[(t, tid)].values())
        for (t, tid) in track_frames if t == "player"
    )

    flips = sum(
        sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        for seq in track_club_seq.values()
    )
    # Tracks whose club sequence mixes more than one club
    mixed = {
        f"{t}{tid}": {k: v for k, v in club_frames[(t, tid)].items()}
        for (t, tid) in track_frames
        if len(club_frames[(t, tid)]) > 1 and track_frames[(t, tid)] >= 30
    }

    report = {
        "name": args.name,
        "frames": len(lines),
        "player_track_frames": player_total,
        "player_coverage": player_club / player_total if player_total else None,
        "n_player_tracks_flagged_referee": len(flagged_tracks),
        "flagged_referee_tracks": flagged_tracks,
        "n_referee_tracks_restored": len(restored),
        "restored_referee_tracks": restored,
        "goalkeeper_class_club_distribution": dict(gk_clubs),
        "mixed_club_tracks": mixed,
        "total_club_flips": flips,
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
