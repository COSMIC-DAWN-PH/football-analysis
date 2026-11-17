"""Evaluate L4 ROI rescue detection on human ball keyframes.

Runtime trigger and ROI generation use only stored tracker/player state.  GT is
consulted afterwards to score whether rescue adds a true candidate.  This is a
measurement tool: it does not mutate production tracks or model defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import ceil, floor, hypot
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from tracking import (
    BallCandidate,
    BallDetector,
    build_ball_rescue_rois,
    player_consistency_gate,
)
from tracking.ball_tracker import non_max_suppression


DEFAULT_MODEL = Path("models/weights/ball-detection_openvino_model_1280_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/ball-detection_openvino_model_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/ball-detection.pt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--object-tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--roi-imgsz", type=int, default=640)
    parser.add_argument(
        "--trigger-mode",
        choices=("confirmed-gap", "empty-or-unobserved"),
        default="confirmed-gap",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=None,
        help="Evaluate an evenly spaced subset of triggered human keyframes",
    )
    parser.add_argument("--maximum-rois", type=int, default=6)
    parser.add_argument("--minimum-half-size", type=float, default=64.0)
    parser.add_argument("--horizontal-player-scale", type=float, default=2.25)
    parser.add_argument("--upward-player-scale", type=float, default=2.0)
    parser.add_argument("--downward-player-scale", type=float, default=1.0)
    parser.add_argument("--predicted-player-scale", type=float, default=2.5)
    parser.add_argument("--dedup-iou", type=float, default=0.60)
    parser.add_argument("--player-distance-gate", type=float, default=1.2)
    parser.add_argument("--player-size-factor-gate", type=float, default=2.2)
    parser.add_argument("--gate-min-players", type=int, default=2)
    return parser


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _center(bbox) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2,
        (float(bbox[1]) + float(bbox[3])) / 2,
    )


def _center_match(gt_bbox, candidates: list[BallCandidate]) -> bool:
    gt_center = _center(gt_bbox)
    radius = hypot(
        float(gt_bbox[2]) - float(gt_bbox[0]),
        float(gt_bbox[3]) - float(gt_bbox[1]),
    ) / 2
    return any(
        hypot(candidate.center[0] - gt_center[0], candidate.center[1] - gt_center[1])
        <= radius
        for candidate in candidates
    )


def _stored_candidates(row: dict) -> list[BallCandidate]:
    return [
        BallCandidate(tuple(map(float, candidate[:4])), float(candidate[4]))
        for candidate in row.get("candidates", [])
    ]


def _triggered(row: dict, mode: str) -> bool:
    track = row.get("track")
    if mode == "confirmed-gap":
        return bool(
            track
            and track.get("track_confirmed", False)
            and not track.get("observed", False)
        )
    return not row.get("candidates") or not track or not track.get("observed", False)


def _previous_confirmed_center(tracks_by_frame: dict[int, dict], frame: int):
    previous = tracks_by_frame.get(frame - 1, {}).get("track")
    if previous and previous.get("track_confirmed", False):
        return _center(previous["bbox"])
    return None


def _evenly_spaced(items: list[dict], maximum: int | None) -> list[dict]:
    if maximum is None or maximum >= len(items):
        return items
    if maximum <= 0:
        return []
    if maximum == 1:
        return [items[len(items) // 2]]
    indexes = {
        round(index * (len(items) - 1) / (maximum - 1)) for index in range(maximum)
    }
    return [items[index] for index in sorted(indexes)]


def _detect_rois(
    detector,
    frame,
    rois,
    image_size: int,
) -> list[BallCandidate]:
    height, width = frame.shape[:2]
    candidates: list[BallCandidate] = []
    for roi in rois:
        x1 = max(0, floor(roi.bbox[0]))
        y1 = max(0, floor(roi.bbox[1]))
        x2 = min(width, ceil(roi.bbox[2]))
        y2 = min(height, ceil(roi.bbox[3]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        candidates.extend(detector._predict(frame[y1:y2, x1:x2], x1, y1, image_size))
    return non_max_suppression(candidates)


def _read_frame(capture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not read video frame {frame_index}")
    return frame


def main() -> None:
    args = _parser().parse_args()
    if args.roi_imgsz < 32:
        raise ValueError("--roi-imgsz must be at least 32")
    gt_rows = [row for row in _load(args.gt) if not row.get("interp", False)]
    track_rows = _load(args.tracks)
    object_rows = _load(args.object_tracks)
    tracks_by_frame = {int(row["frame"]): row for row in track_rows}

    triggered_keyframes = [
        gt
        for gt in gt_rows
        if _triggered(tracks_by_frame.get(int(gt["frame"]), {}), args.trigger_mode)
    ]
    selected_keyframes = _evenly_spaced(triggered_keyframes, args.max_keyframes)
    runtime_trigger_frames = sum(
        _triggered(row, args.trigger_mode) for row in track_rows
    )

    detector = BallDetector(
        str(args.model), confidence=args.conf, device=args.device
    )
    fixed_size = BallDetector._fixed_export_size(args.model)
    inference_size = fixed_size or args.roi_imgsz
    roi_options = {
        "maximum_rois": args.maximum_rois,
        "minimum_half_size": args.minimum_half_size,
        "horizontal_player_scale": args.horizontal_player_scale,
        "upward_player_scale": args.upward_player_scale,
        "downward_player_scale": args.downward_player_scale,
        "predicted_player_scale": args.predicted_player_scale,
        "dedup_iou": args.dedup_iou,
    }

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    details = []
    total_roi_calls = 0
    started = time.perf_counter()
    try:
        for index, gt in enumerate(selected_keyframes, 1):
            frame_index = int(gt["frame"])
            row = tracks_by_frame.get(frame_index, {})
            objects = object_rows[frame_index] if frame_index < len(object_rows) else {}
            frame = _read_frame(capture, frame_index)
            rois = build_ball_rescue_rois(
                objects,
                frame.shape,
                predicted_center=_previous_confirmed_center(
                    tracks_by_frame, frame_index
                ),
                **roi_options,
            )
            raw_rescue = _detect_rois(detector, frame, rois, inference_size)
            gated_rescue = player_consistency_gate(
                raw_rescue,
                objects,
                distance_gate=args.player_distance_gate,
                size_factor_gate=args.player_size_factor_gate,
                minimum_players=args.gate_min_players,
            )
            primary = _stored_candidates(row)
            combined = non_max_suppression(primary + gated_rescue)
            primary_hit = _center_match(gt["bbox"], primary)
            rescue_hit = _center_match(gt["bbox"], gated_rescue)
            details.append(
                {
                    "frame": frame_index,
                    "roi_count": len(rois),
                    "raw_rescue_candidates": len(raw_rescue),
                    "gated_rescue_candidates": len(gated_rescue),
                    "primary_hit": primary_hit,
                    "rescue_hit": rescue_hit,
                    "added_hit": rescue_hit and not primary_hit,
                    "combined_hit": _center_match(gt["bbox"], combined),
                }
            )
            total_roi_calls += len(rois)
            print(
                f"[{index}/{len(selected_keyframes)}] frame={frame_index} "
                f"rois={len(rois)} raw={len(raw_rescue)} gated={len(gated_rescue)} "
                f"added={details[-1]['added_hit']}",
                flush=True,
            )
    finally:
        capture.release()

    elapsed = time.perf_counter() - started
    evaluated = len(details)
    mean_rois = total_roi_calls / evaluated if evaluated else 0.0
    trigger_rate = runtime_trigger_frames / len(track_rows) if track_rows else 0.0
    pixel_normalized_extra = (
        trigger_rate * mean_rois * (inference_size / 1280.0) ** 2 / 5.0
    )
    report = {
        "scope": "human keyframes selected only after runtime trigger; GT scores output only",
        "video": str(args.video),
        "tracks": str(args.tracks),
        "model": str(args.model),
        "device": detector.device,
        "fixed_export_size": fixed_size,
        "roi_inference_size": inference_size,
        "trigger_mode": args.trigger_mode,
        "runtime_frames": len(track_rows),
        "runtime_trigger_frames": runtime_trigger_frames,
        "runtime_trigger_rate": round(trigger_rate, 4),
        "triggered_human_keyframes": len(triggered_keyframes),
        "evaluated_human_keyframes": evaluated,
        "roi_options": roi_options,
        "roi_calls": total_roi_calls,
        "mean_rois_per_evaluated_frame": round(mean_rois, 4),
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_roi_call": round(elapsed / total_roi_calls, 4)
        if total_roi_calls
        else None,
        "estimated_extra_compute_vs_five_1280_calls": round(
            pixel_normalized_extra, 4
        ),
        "primary_hits": sum(item["primary_hit"] for item in details),
        "rescue_hits": sum(item["rescue_hit"] for item in details),
        "added_hits": sum(item["added_hit"] for item in details),
        "combined_hits": sum(item["combined_hit"] for item in details),
        "details": details,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
