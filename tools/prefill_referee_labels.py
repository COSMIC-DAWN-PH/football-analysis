"""Prefill manual_label in referee candidate manifests with rule-based guesses.

The human reviews the crops and CORRECTS any wrong values (manual_label is
the ground truth used by tools/eval_referee.py). Rules:

  - yellow player tracks          -> referee
  - ref_cls with navy/maroon hue  -> navy / maroon
  - ref_cls with yellow hue       -> referee
  - dark / boundary               -> null (needs human eyes)
  - controls                      -> their sampled zone
  - verdict flag_player           -> referee (my claim, must be verified)
  - verdict restore_referee       -> the club the assigner restored, read from
                                     the replayed tracks file

Usage:
  python tools/prefill_referee_labels.py --manifest eval/referee_crops/demo2/candidates.jsonl
  python tools/prefill_referee_labels.py --manifest eval/referee_crops/demo2/verdict/candidates.jsonl \
      --verdict-tracks output_videos/demo2-30s-test/raw/object_tracks_referee_replay.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _zone(hue):
    if hue is None:
        return None
    if 15 <= hue < 45:
        return "yellow"
    if 45 <= hue < 95:
        return "boundary"
    if 95 <= hue < 135:
        return "navy"
    return "maroon"


def _guess(meta):
    category = meta.get("category", "")
    hue = meta.get("median_hue")
    if category == "yellow":
        return "referee"
    if category == "navy_control":
        return "navy"
    if category == "maroon_control":
        return "maroon"
    if category == "ref_cls":
        zone = _zone(hue)
        if zone == "yellow":
            return "referee"
        if zone in ("navy", "maroon"):
            return zone
        return None
    if category == "dark":
        return None
    if category == "boundary":
        return None
    if category == "flag_player":
        return "referee"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verdict-tracks", type=Path, default=None)
    args = parser.parse_args()

    entries = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    restore_lookup = {}
    if args.verdict_tracks is not None:
        lines = args.verdict_tracks.read_text(encoding="utf-8").splitlines()
        for meta in entries:
            if meta.get("category") != "restore_referee":
                continue
            frame = meta["frame"]
            if frame >= len(lines):
                continue
            d = json.loads(lines[frame])
            tr = d.get(meta["track_type"], {}).get(str(meta["track_id"]))
            if tr is not None and tr.get("club") is not None:
                restore_lookup[meta["file"]] = tr["club"]

    changed = 0
    for meta in entries:
        guess = _guess(meta)
        if guess is None and meta.get("category") == "restore_referee":
            guess = restore_lookup.get(meta["file"])
        if guess is not None:
            if meta.get("manual_label") != guess:
                meta["manual_label"] = guess
                changed += 1

    args.manifest.write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in entries) + "\n",
        encoding="utf-8",
    )
    print(f"prefilled {changed}/{len(entries)} entries in {args.manifest}")


if __name__ == "__main__":
    main()
