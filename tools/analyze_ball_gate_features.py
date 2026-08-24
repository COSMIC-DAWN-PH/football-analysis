"""Measure player-distance and size-consistency features for ball gates.

True samples are detector candidates whose centers fall inside human GT
keyframe boxes.  False samples are the 20 human-confirmed demo4 pilot targets.
The report is diagnostic only; it does not change tracks or annotations.
"""
import argparse
import json
from math import hypot, sqrt
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _center_distance(first: list[float], second: list[float]) -> float:
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)
    return hypot(first_x - second_x, first_y - second_y)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "min": round(min(values), 4) if values else None,
        "p10": round(_percentile(values, 0.10), 4) if values else None,
        "median": round(median(values), 4) if values else None,
        "mean": round(mean(values), 4) if values else None,
        "p90": round(_percentile(values, 0.90), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def _features(bbox: list[float], objects: dict) -> dict | None:
    center_x, center_y = _center(bbox)
    candidate_size = sqrt(
        max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    )
    nearest = None
    player_count = 0
    for player_type in ("player", "goalkeeper"):
        for player in objects.get(player_type, {}).values():
            player_count += 1
            x1, y1, x2, y2 = map(float, player["bbox"])
            height = max(1.0, y2 - y1)
            distance = hypot(center_x - (x1 + x2) / 2, center_y - y2)
            radius = max(48.0, 2.5 * height)
            normalized_distance = distance / radius
            if nearest is None or normalized_distance < nearest[0]:
                nearest = (normalized_distance, height, distance)
    if nearest is None:
        return None
    normalized_distance, player_height, distance = nearest
    expected_size = max(1.0, player_height / 8.0)
    size_factor = max(candidate_size / expected_size, expected_size / candidate_size)
    return {
        "player_distance_ratio": normalized_distance,
        "player_distance_px": distance,
        "candidate_size_px": candidate_size,
        "expected_size_px": expected_size,
        "size_factor": size_factor,
        "player_count": player_count,
    }


def _true_samples(gt: list[dict], ball_rows: list[dict], objects: list[dict]) -> list[dict]:
    ball_by_frame = {int(row["frame"]): row for row in ball_rows}
    samples = []
    for row in gt:
        if row.get("interp"):
            continue
        frame = int(row["frame"])
        if frame >= len(objects) or frame not in ball_by_frame:
            continue
        gt_width = row["bbox"][2] - row["bbox"][0]
        gt_height = row["bbox"][3] - row["bbox"][1]
        radius = max(4.0, hypot(gt_width, gt_height) / 2)
        candidates = ball_by_frame[frame].get("candidates") or []
        matching = [
            candidate
            for candidate in candidates
            if _center_distance(row["bbox"], candidate[:4]) <= radius
        ]
        if not matching:
            continue
        candidate = min(matching, key=lambda item: _center_distance(row["bbox"], item[:4]))
        features = _features(candidate[:4], objects[frame])
        if features is not None:
            samples.append({"frame": frame, "bbox": candidate[:4], **features})
    return samples


def _false_samples(
    pilot: list[dict], candidates: list[dict], objects: list[dict]
) -> list[dict]:
    candidates_by_id = {row["id"]: row for row in candidates}
    samples = []
    for item in pilot:
        frame = int(item["frame"])
        if frame >= len(objects):
            continue
        reference = candidates_by_id[item["id"]]
        features = _features(reference["bbox"], objects[frame])
        if features is not None:
            samples.append({"id": item["id"], "frame": frame, **features})
    return samples


def _sample_report(samples: list[dict]) -> dict:
    return {
        "samples": len(samples),
        "player_distance_ratio": _summary(
            [sample["player_distance_ratio"] for sample in samples]
        ),
        "size_factor": _summary([sample["size_factor"] for sample in samples]),
        "player_count": _summary(
            [float(sample["player_count"]) for sample in samples]
        ),
        "details": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--ball-tracks", type=Path, required=True)
    parser.add_argument("--object-tracks", type=Path, required=True)
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=ROOT / "eval" / "ball_crops" / "demo4" / "pilot" / "manifest.jsonl",
    )
    parser.add_argument(
        "--candidates-manifest",
        type=Path,
        default=ROOT / "eval" / "ball_crops" / "demo4" / "candidates.jsonl",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    paths = (
        args.gt,
        args.ball_tracks,
        args.object_tracks,
        args.pilot_manifest,
        args.candidates_manifest,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    objects = _load(args.object_tracks)
    true_samples = _true_samples(_load(args.gt), _load(args.ball_tracks), objects)
    false_samples = _false_samples(
        _load(args.pilot_manifest), _load(args.candidates_manifest), objects
    )
    report = {
        "true_candidates": _sample_report(true_samples),
        "pilot_false_targets": _sample_report(false_samples),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
