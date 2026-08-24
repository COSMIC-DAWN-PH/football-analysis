"""Check whether the 20 human-confirmed demo4 false targets still survive.

The pilot manifest identifies the false targets, while ``candidates.jsonl``
contains their reference boxes.  A target survives when the replayed track at
that exact frame remains close to the same reference center.

Usage:
  python tools/smoke_ball_fp.py \
      --tracks output_videos/demo4/ball/ball_tracks.jsonl
"""
import argparse
import json
from collections import Counter
from math import hypot
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PILOT = ROOT / "eval" / "ball_crops" / "demo4" / "pilot" / "manifest.jsonl"
DEFAULT_CANDIDATES = ROOT / "eval" / "ball_crops" / "demo4" / "candidates.jsonl"


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance(first: list[float], second: list[float]) -> float:
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)
    return hypot(first_x - second_x, first_y - second_y)


def _report(
    pilot: list[dict], candidates: list[dict], tracks: list[dict], radius: float
) -> dict:
    candidates_by_id = {row["id"]: row for row in candidates}
    tracks_by_frame = {int(row["frame"]): row for row in tracks}
    results = []
    for item in pilot:
        reference = candidates_by_id[item["id"]]
        track_row = tracks_by_frame.get(int(item["frame"]))
        track = track_row.get("track") if track_row else None
        distance = (
            _distance(reference["bbox"], track["bbox"])
            if track is not None
            else None
        )
        results.append(
            {
                "id": item["id"],
                "frame": int(item["frame"]),
                "category": item["category"],
                "survived": distance is not None and distance <= radius,
                "observed": bool(track.get("observed", True)) if track else False,
                "track_confirmed": bool(track.get("track_confirmed", True)) if track else False,
                "center_distance_px": round(distance, 3) if distance is not None else None,
            }
        )
    survived = [item for item in results if item["survived"]]
    survived_observed = [item for item in survived if item["observed"]]
    survived_confirmed = [item for item in survived if item["track_confirmed"]]
    by_category_total = Counter(item["category"] for item in results)
    by_category_survived = Counter(item["category"] for item in survived)
    return {
        "ground_truth": "all pilot items are not_ball",
        "match_radius_px": radius,
        "items": len(results),
        "survived_false_targets": len(survived),
        "survived_observed_false_targets": len(survived_observed),
        "survived_confirmed_false_targets": len(survived_confirmed),
        "removed_false_targets": len(results) - len(survived),
        "by_category": {
            category: {
                "items": by_category_total[category],
                "survived": by_category_survived[category],
            }
            for category in sorted(by_category_total)
        },
        "survived_ids": [item["id"] for item in survived],
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--candidates-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--radius", type=float, default=24.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    for path in (args.tracks, args.pilot_manifest, args.candidates_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = _report(
        _load(args.pilot_manifest),
        _load(args.candidates_manifest),
        _load(args.tracks),
        args.radius,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
