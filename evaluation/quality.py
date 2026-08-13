from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _iou(first: Iterable[float], second: Iterable[float]) -> float:
    a = list(map(float, first))
    b = list(map(float, second))
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_run(
    tracks_dir: str | Path,
    *,
    ground_truth_path: Optional[str | Path] = None,
    fps: float = 30.0,
) -> dict[str, Any]:
    root = Path(tracks_dir)
    calibrations = _read_jsonl(root / "calibration_tracks.jsonl")
    objects = _read_jsonl(root / "object_tracks.jsonl")
    frame_count = min(len(calibrations), len(objects))
    pose_usable = sum(
        1
        for pose in calibrations[:frame_count]
        if pose.get("image_to_pitch") is not None
        and pose.get("age_seconds") is not None
        and float(pose.get("age_seconds", 99.0)) <= 0.25
        and float(pose.get("quality", 0.0)) >= 0.35
        and not pose.get("zoom_changed", False)
    )
    marking_errors = [
        float(pose["marking_error_px"])
        for pose in calibrations[:frame_count]
        if pose.get("marking_error_px") is not None
    ]
    landmark_errors = [
        float(pose["median_error_px"])
        for pose in calibrations[:frame_count]
        if pose.get("median_error_px") is not None
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "frames": frame_count,
        "duration_seconds": frame_count / max(fps, 1e-6),
        "pose_usable_ratio": pose_usable / max(1, frame_count),
        "median_landmark_reprojection_error_px": (
            median(landmark_errors) if landmark_errors else None
        ),
        "median_pitch_marking_error_px": median(marking_errors) if marking_errors else None,
        "ground_truth_available": ground_truth_path is not None,
    }
    if ground_truth_path is None:
        report["promotion_eligible"] = False
        report["promotion_errors"] = [
            "Ground-truth JSONL is required for detector and speed promotion"
        ]
        return report

    truth = _read_jsonl(Path(ground_truth_path))
    evaluated = min(frame_count, len(truth))
    true_positive = false_positive = false_negative = 0
    false_track_segments: set[str] = set()
    speed_errors: dict[str, list[float]] = {"ground": [], "air": []}
    speed_relative_errors: dict[str, list[float]] = {"ground": [], "air": []}
    unsupported_reliable = 0
    for index in range(evaluated):
        ball_items = list(objects[index].get("ball", {}).items())
        predicted_item = next(
            (
                (ball_id, ball)
                for ball_id, ball in ball_items
                if ball.get("observed", False)
                and ball.get("track_confirmed", False)
            ),
            None,
        )
        predicted = predicted_item[1] if predicted_item is not None else None
        expected = truth[index]
        visible = bool(expected.get("ball_visible", False))
        if predicted is not None and visible and expected.get("bbox") is not None:
            if _iou(predicted["bbox"], expected["bbox"]) >= 0.10:
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1
                false_track_segments.add(
                    f"{predicted_item[0]}:{predicted.get('track_segment', 0)}"
                )
        elif predicted is not None:
            false_positive += 1
            false_track_segments.add(
                f"{predicted_item[0]}:{predicted.get('track_segment', 0)}"
            )
        elif visible:
            false_negative += 1

        if predicted is not None and predicted.get("speed_state") == "reliable":
            reference = expected.get("speed_3d_kmh")
            mode = str(expected.get("motion_mode", predicted.get("motion_mode", "ground")))
            supported = bool(expected.get("speed_supported", reference is not None))
            if not supported or reference is None:
                unsupported_reliable += 1
            elif mode in speed_errors:
                absolute_error = abs(float(predicted["speed"]) - float(reference))
                speed_errors[mode].append(absolute_error)
                if abs(float(reference)) > 1e-6:
                    speed_relative_errors[mode].append(
                        absolute_error / abs(float(reference))
                    )

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    duration_minutes = evaluated / max(fps, 1e-6) / 60.0
    false_tracks_per_minute = len(false_track_segments) / max(duration_minutes, 1e-6)
    report.update(
        evaluated_frames=evaluated,
        true_positive_frames=true_positive,
        false_positive_frames=false_positive,
        confirmed_false_track_segments=len(false_track_segments),
        false_negative_frames=false_negative,
        ball_precision=precision,
        visible_ball_recall=recall,
        confirmed_false_tracks_per_minute=false_tracks_per_minute,
        unsupported_reliable_speed_frames=unsupported_reliable,
        ground_speed_median_absolute_error_kmh=(
            median(speed_errors["ground"]) if speed_errors["ground"] else None
        ),
        ground_speed_median_relative_error=(
            median(speed_relative_errors["ground"])
            if speed_relative_errors["ground"]
            else None
        ),
        air_speed_median_absolute_error_kmh=(
            median(speed_errors["air"]) if speed_errors["air"] else None
        ),
        air_speed_median_relative_error=(
            median(speed_relative_errors["air"])
            if speed_relative_errors["air"]
            else None
        ),
    )
    errors = []
    if report["pose_usable_ratio"] < 0.80:
        errors.append("pose usable coverage is below 80%")
    landmark_error = report["median_landmark_reprojection_error_px"]
    if landmark_error is None:
        errors.append("landmark reprojection error is unavailable")
    elif landmark_error > 3.0:
        errors.append("median landmark reprojection error exceeds 3 px")
    if precision < 0.95:
        errors.append("ball precision is below 95%")
    if recall < 0.75:
        errors.append("visible-ball recall is below 75%")
    if false_tracks_per_minute > 0.5:
        errors.append("confirmed false tracks exceed 0.5 per minute")
    if unsupported_reliable:
        errors.append("reliable speed was emitted for unsupported frames")
    ground_error = report["ground_speed_median_absolute_error_kmh"]
    ground_relative_error = report["ground_speed_median_relative_error"]
    air_error = report["air_speed_median_absolute_error_kmh"]
    air_relative_error = report["air_speed_median_relative_error"]
    if (
        ground_error is not None
        and ground_error > 3.0
        and (ground_relative_error is None or ground_relative_error > 0.10)
    ):
        errors.append("ground speed error exceeds both 3 km/h and 10%")
    if (
        air_error is not None
        and air_error > 5.0
        and (air_relative_error is None or air_relative_error > 0.15)
    ):
        errors.append("air speed error exceeds both 5 km/h and 15%")
    report["promotion_errors"] = errors
    report["promotion_eligible"] = not errors
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Football-Analysis run")
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_run(
        args.tracks_dir,
        ground_truth_path=args.ground_truth,
        fps=args.fps,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0 if report.get("promotion_eligible", False) else 1
