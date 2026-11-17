"""Measure L4 ROI geometry coverage without running ball inference.

ROI proposals use only player tracks and the preceding tracker output.  Ball
GT is read afterwards solely to score coverage, preventing target leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import hypot
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking import build_ball_rescue_rois
from tracking.ball_rescue import point_in_rescue_roi


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--object-tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--maximum-rois", type=int, default=6)
    parser.add_argument("--minimum-half-size", type=float, default=48.0)
    parser.add_argument("--horizontal-player-scale", type=float, default=1.75)
    parser.add_argument("--upward-player-scale", type=float, default=1.50)
    parser.add_argument("--downward-player-scale", type=float, default=0.75)
    parser.add_argument("--predicted-player-scale", type=float, default=2.0)
    parser.add_argument("--dedup-iou", type=float, default=0.70)
    parser.add_argument(
        "--prediction-mode",
        choices=("any", "confirmed", "observed", "none"),
        default="confirmed",
        help="Which previous-frame track is trusted for ROI ranking",
    )
    parser.add_argument("--details-limit", type=int, default=50)
    return parser


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _center(bbox: list[float]) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2,
        (float(bbox[1]) + float(bbox[3])) / 2,
    )


def _center_match(gt_bbox: list[float], candidates: list[list[float]]) -> bool:
    gt_center = _center(gt_bbox)
    radius = hypot(
        float(gt_bbox[2]) - float(gt_bbox[0]),
        float(gt_bbox[3]) - float(gt_bbox[1]),
    ) / 2
    return any(hypot(*(
        gt_center[index] - _center(candidate[:4])[index]
        for index in (0, 1)
    )) <= radius for candidate in candidates)


def _previous_track_center(
    tracks_by_frame: dict[int, dict], frame: int, prediction_mode: str
):
    if prediction_mode == "none":
        return None
    previous = tracks_by_frame.get(frame - 1)
    track = previous.get("track") if previous else None
    if track and prediction_mode == "confirmed" and not track.get("track_confirmed", False):
        track = None
    if track and prediction_mode == "observed" and not track.get("observed", False):
        track = None
    return _center(track["bbox"]) if track else None


def _fraction(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(mean(values), 4) if values else None,
        "median": round(median(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def _report(
    gt_rows: list[dict],
    track_rows: list[dict],
    object_rows: list[dict],
    frame_shape: tuple[int, int, int],
    roi_options: dict,
    prediction_mode: str,
    details_limit: int,
) -> dict:
    tracks_by_frame = {int(row["frame"]): row for row in track_rows}
    keyframes = [row for row in gt_rows if not row.get("interp", False)]
    evaluated = []
    for gt in keyframes:
        frame = int(gt["frame"])
        track_row = tracks_by_frame.get(frame, {})
        objects = object_rows[frame] if frame < len(object_rows) else {}
        predicted_center = _previous_track_center(
            tracks_by_frame, frame, prediction_mode
        )
        rois = build_ball_rescue_rois(
            objects,
            frame_shape,
            predicted_center=predicted_center,
            **roi_options,
        )
        gt_center = _center(gt["bbox"])
        covered = any(point_in_rescue_roi(gt_center, roi) for roi in rois)
        detected = _center_match(gt["bbox"], track_row.get("candidates", []))
        empty_primary_candidates = not track_row.get("candidates", [])
        roi_area_fraction = sum(
            (roi.bbox[2] - roi.bbox[0]) * (roi.bbox[3] - roi.bbox[1])
            for roi in rois
        ) / (frame_shape[0] * frame_shape[1])
        player_count = sum(
            len(objects.get(player_type, {}))
            for player_type in ("player", "goalkeeper")
        )
        evaluated.append(
            {
                "frame": frame,
                "segment": int(gt.get("seg", 0)),
                "detected": detected,
                "empty_primary_candidates": empty_primary_candidates,
                "covered": covered,
                "player_count": player_count,
                "has_prediction": predicted_center is not None,
                "roi_count": len(rois),
                "roi_area_fraction": roi_area_fraction,
                "roi_sources": [roi.source for roi in rois],
            }
        )

    misses = [row for row in evaluated if not row["detected"]]
    covered_misses = [row for row in misses if row["covered"]]
    uncovered_misses = [row for row in misses if not row["covered"]]
    with_prediction = [row for row in misses if row["has_prediction"]]
    without_prediction = [row for row in misses if not row["has_prediction"]]
    empty_primary_misses = [row for row in misses if row["empty_primary_candidates"]]
    segment_ids = sorted({row["segment"] for row in evaluated})
    return {
        "scope": "human keyframes only; GT is used after ROI generation only",
        "prediction_mode": prediction_mode,
        "roi_options": roi_options,
        "keyframes": len(evaluated),
        "detector_misses": len(misses),
        "covered_detector_misses": len(covered_misses),
        "detector_miss_coverage": _fraction(len(covered_misses), len(misses)),
        "empty_primary_candidate_misses": {
            "items": len(empty_primary_misses),
            "covered": sum(row["covered"] for row in empty_primary_misses),
            "coverage": _fraction(
                sum(row["covered"] for row in empty_primary_misses),
                len(empty_primary_misses),
            ),
        },
        "misses_with_previous_track": {
            "items": len(with_prediction),
            "covered": sum(row["covered"] for row in with_prediction),
            "coverage": _fraction(
                sum(row["covered"] for row in with_prediction), len(with_prediction)
            ),
        },
        "misses_without_previous_track": {
            "items": len(without_prediction),
            "covered": sum(row["covered"] for row in without_prediction),
            "coverage": _fraction(
                sum(row["covered"] for row in without_prediction), len(without_prediction)
            ),
        },
        "by_segment": {
            str(segment): {
                "misses": len(items := [row for row in misses if row["segment"] == segment]),
                "covered": sum(row["covered"] for row in items),
                "coverage": _fraction(sum(row["covered"] for row in items), len(items)),
            }
            for segment in segment_ids
        },
        "player_counts_on_misses": dict(Counter(row["player_count"] for row in misses)),
        "roi_count": _summary([float(row["roi_count"]) for row in misses]),
        "summed_roi_area_fraction": _summary(
            [row["roi_area_fraction"] for row in misses]
        ),
        "sample_covered_miss_frames": [
            row["frame"] for row in covered_misses[:details_limit]
        ],
        "sample_uncovered_miss_frames": [
            row["frame"] for row in uncovered_misses[:details_limit]
        ],
    }


def main() -> None:
    args = _parser().parse_args()
    roi_options = {
        "maximum_rois": args.maximum_rois,
        "minimum_half_size": args.minimum_half_size,
        "horizontal_player_scale": args.horizontal_player_scale,
        "upward_player_scale": args.upward_player_scale,
        "downward_player_scale": args.downward_player_scale,
        "predicted_player_scale": args.predicted_player_scale,
        "dedup_iou": args.dedup_iou,
    }
    report = _report(
        _load(args.gt),
        _load(args.tracks),
        _load(args.object_tracks),
        (args.height, args.width, 3),
        roi_options,
        args.prediction_mode,
        args.details_limit,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
