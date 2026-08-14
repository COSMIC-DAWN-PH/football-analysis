"""Evaluate referee handling against a manually labeled crop set.

The labels come from the candidate extraction manifest (tools/
extract_referee_candidates.py) whose `manual_label` field was filled by a
human with one of: referee / maroon / navy (null = ambiguous, skipped).

Pipeline decisions are read from a tracks JSONL that was replayed with the
current ClubAssigner (tools/replay_assign.py):
  - player-class track at a frame: 'referee' when track['referee'] is set,
    otherwise its 'club' value (Maroon/Navy) or None when unassigned
  - referee-class track at a frame: its 'club' value when restored, else
    'referee'

Metrics:
  - referee recall/precision on the labeled crops (referee vs club classes)
  - club preservation: labeled maroon/navy crops still assigned the right club
  - track-level majority version of the same
  - coverage of club assignment among labeled club crops

Usage:
  python tools/eval_referee.py \
      --labels eval/referee_crops/demo2/candidates.jsonl \
      --labels eval/referee_crops/demo2/verdict/candidates.jsonl \
      --tracks output_videos/demo2-30s-test/raw/object_tracks_referee_replay.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLUB_LABELS = ("maroon", "navy")


def _pipeline_decision(track: dict, track_type: str):
    if track_type == "referee":
        club = track.get("club")
        return club if club is not None else "referee"
    if track.get("referee"):
        return "referee"
    return track.get("club")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    args = parser.parse_args()

    labels = []
    for path in args.labels:
        with open(path, encoding="utf-8") as f:
            for line in f:
                meta = json.loads(line)
                if meta.get("manual_label") in ("referee", "maroon", "navy"):
                    labels.append(meta)
    if not labels:
        raise SystemExit("no labeled entries found; fill manual_label first")

    tracks_lines = args.tracks.read_text(encoding="utf-8").splitlines()

    conf = collections.Counter()
    crop_correct = 0
    crop_total = 0
    track_decisions = collections.defaultdict(list)
    label_per_track = {}

    for meta in labels:
        frame = meta["frame"]
        track_type = meta["track_type"]
        track_id = meta["track_id"]
        truth = meta["manual_label"]
        if frame >= len(tracks_lines):
            continue
        d = json.loads(tracks_lines[frame])
        tr = d.get(track_type, {}).get(str(track_id))
        if tr is None:
            tr = d.get(track_type, {}).get(track_id)
        if tr is None:
            conf[(truth, "absent")] += 1
            continue
        decision = _pipeline_decision(tr, track_type)
        if isinstance(decision, str):
            decision = decision.casefold()
        key = (track_type, track_id)
        track_decisions[key].append(decision)
        prev = label_per_track.get(key)
        if prev is not None and prev != truth:
            print(f"WARNING: conflicting manual labels within track {key}: {prev} vs {truth}")
        label_per_track[key] = truth
        conf[(truth, decision)] += 1
        crop_total += 1
        crop_correct += int(decision == truth)

    # Track-level majority decision
    track_conf = collections.Counter()
    track_correct = 0
    track_total = 0
    for key, decisions in track_decisions.items():
        truth = label_per_track[key]
        majority = collections.Counter(decisions).most_common(1)[0][0]
        track_conf[(truth, majority)] += 1
        track_total += 1
        track_correct += int(majority == truth)

    # Referee detection metrics: truth=referee vs truth=club (maroon/navy)
    def referee_metrics(confusion):
        tp = sum(n for (t, d), n in confusion.items() if t == "referee" and d == "referee")
        fn = sum(n for (t, d), n in confusion.items() if t == "referee" and d != "referee")
        fp = sum(n for (t, d), n in confusion.items() if t != "referee" and d == "referee")
        tn = sum(n for (t, d), n in confusion.items() if t != "referee" and d != "referee")
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": recall, "precision": precision}

    # Club preservation on club-labeled crops: decision == truth
    club_crops = [(t, d) for (t, d) in conf if t in CLUB_LABELS]
    club_ok = sum(conf[k] for k in club_crops if k[0] == k[1])
    club_n = sum(conf[k] for k in club_crops)
    club_track_crops = [(t, d) for (t, d) in track_conf if t in CLUB_LABELS]
    club_track_ok = sum(track_conf[k] for k in club_track_crops if k[0] == k[1])
    club_track_n = sum(track_conf[k] for k in club_track_crops)

    # Coverage: labeled club crops with any club decision
    club_assigned = sum(
        n for (t, d), n in conf.items() if t in CLUB_LABELS and d in CLUB_LABELS
    )

    report = {
        "crop_confusion": {f"{t}->{d}": n for (t, d), n in sorted(conf.items(), key=lambda kv: str(kv[0]))},
        "crop_accuracy": crop_correct / crop_total if crop_total else None,
        "crop_referee_metrics": referee_metrics(conf),
        "crop_club_preservation": club_ok / club_n if club_n else None,
        "track_confusion": {f"{t}->{d}": n for (t, d), n in sorted(track_conf.items(), key=lambda kv: str(kv[0]))},
        "track_accuracy": track_correct / track_total if track_total else None,
        "track_referee_metrics": referee_metrics(track_conf),
        "track_club_preservation": club_track_ok / club_track_n if club_track_n else None,
        "club_coverage": club_assigned / club_n if club_n else None,
        "n_labeled_crops": crop_total,
        "n_labeled_tracks": track_total,
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
