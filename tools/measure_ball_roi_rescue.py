"""Run a bounded L4 ROI-rescue experiment on existing demo4 evidence.

Only keyframes whose already-gated primary candidate list is empty trigger
inference.  ROI generation cannot see ball GT; GT and the pilot false targets
are consulted only after candidates have been produced.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import hypot
from pathlib import Path
from statistics import mean, median
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from tracking import (
    BallCandidate,
    BallDetector,
    build_ball_rescue_rois,
    player_consistency_gate,
)


DEFAULT_MODEL = Path("models/weights/ball-detection_openvino_model_1280_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/ball-detection_openvino_model_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/ball-detection.pt")
DEFAULT_PILOT = Path("eval/ball_crops/demo4/pilot/manifest.jsonl")
DEFAULT_CANDIDATES = Path("eval/ball_crops/demo4/candidates.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--object-tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates-out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.02)
    parser.add_argument(
        "--maximum-upscale",
        type=float,
        default=1.5,
        help="Controlled ROI enlargement before padding to the fixed model canvas",
    )
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--maximum-rois", type=int, default=6)
    parser.add_argument("--minimum-half-size", type=float, default=64.0)
    parser.add_argument("--horizontal-player-scale", type=float, default=2.25)
    parser.add_argument("--upward-player-scale", type=float, default=2.0)
    parser.add_argument("--downward-player-scale", type=float, default=1.0)
    parser.add_argument("--predicted-player-scale", type=float, default=2.5)
    parser.add_argument("--dedup-iou", type=float, default=0.6)
    parser.add_argument("--player-distance-gate", type=float, default=1.2)
    parser.add_argument("--player-size-factor-gate", type=float, default=2.2)
    parser.add_argument("--gate-min-players", type=int, default=2)
    parser.add_argument(
        "--gt-target-mode",
        choices=("empty", "matched", "all"),
        default="empty",
        help="Select empty-primary rescue targets or matched positive controls",
    )
    parser.add_argument(
        "--max-gt-frames",
        type=int,
        default=None,
        help="Bounded smoke run over the earliest trigger keyframes",
    )
    parser.add_argument(
        "--gt-start-frame",
        type=int,
        default=0,
        help="Ignore earlier GT trigger frames when selecting a bounded sample",
    )
    parser.add_argument(
        "--max-pilot-items",
        type=int,
        default=None,
        help="Bounded smoke run over the earliest pilot false targets",
    )
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


def _match(reference_bbox, candidates: list[BallCandidate], radius: float | None = None) -> bool:
    reference_center = _center(reference_bbox)
    if radius is None:
        radius = hypot(
            float(reference_bbox[2]) - float(reference_bbox[0]),
            float(reference_bbox[3]) - float(reference_bbox[1]),
        ) / 2
    return any(
        hypot(
            candidate.center[0] - reference_center[0],
            candidate.center[1] - reference_center[1],
        )
        <= radius
        for candidate in candidates
    )


def _serialized_match(reference_bbox, candidates: list[list[float]]) -> bool:
    return _match(
        reference_bbox,
        [
            BallCandidate(tuple(map(float, candidate[:4])), float(candidate[4]))
            for candidate in candidates
        ],
    )


def _confirmed_previous_center(tracks_by_frame: dict[int, dict], frame: int):
    previous = tracks_by_frame.get(frame - 1, {}).get("track")
    if previous is None or not previous.get("track_confirmed", False):
        return None
    return _center(previous["bbox"])


def _serialize_candidates(candidates: list[BallCandidate]) -> list[list[float]]:
    return [
        [round(value, 3) for value in candidate.bbox]
        + [round(candidate.confidence, 5)]
        for candidate in candidates
    ]


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(mean(values), 4) if values else None,
        "median": round(median(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
    }


def main() -> None:
    args = _parser().parse_args()
    for path in (
        args.video,
        args.gt,
        args.tracks,
        args.object_tracks,
        args.model,
        args.pilot_manifest,
        args.candidate_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    gt_rows = [row for row in _load(args.gt) if not row.get("interp", False)]
    track_rows = _load(args.tracks)
    object_rows = _load(args.object_tracks)
    tracks_by_frame = {int(row["frame"]): row for row in track_rows}
    gt_by_frame = {}
    for row in gt_rows:
        frame = int(row["frame"])
        primary_candidates = tracks_by_frame.get(frame, {}).get("candidates", [])
        include = (
            args.gt_target_mode == "all"
            or (args.gt_target_mode == "empty" and not primary_candidates)
            or (
                args.gt_target_mode == "matched"
                and _serialized_match(row["bbox"], primary_candidates)
            )
        )
        if include:
            gt_by_frame[frame] = row
    selected_gt_frames = sorted(
        frame for frame in gt_by_frame if frame >= args.gt_start_frame
    )
    gt_by_frame = {frame: gt_by_frame[frame] for frame in selected_gt_frames}
    if args.max_gt_frames is not None:
        selected_gt_frames = selected_gt_frames[: args.max_gt_frames]
        gt_by_frame = {frame: gt_by_frame[frame] for frame in selected_gt_frames}

    references_by_id = {row["id"]: row for row in _load(args.candidate_manifest)}
    pilot_by_frame: dict[int, list[dict]] = {}
    pilot_rows = _load(args.pilot_manifest)
    if args.max_pilot_items is not None:
        pilot_rows = pilot_rows[: args.max_pilot_items]
    for pilot in pilot_rows:
        frame = int(pilot["frame"])
        if tracks_by_frame.get(frame, {}).get("candidates", []):
            continue
        item = dict(pilot)
        item["reference_bbox"] = references_by_id[item["id"]]["bbox"]
        pilot_by_frame.setdefault(frame, []).append(item)

    target_frames = sorted(set(gt_by_frame) | set(pilot_by_frame))
    detector = BallDetector(
        str(args.model), confidence=args.confidence, device=args.device
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    frame_shape = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        3,
    )
    roi_options = {
        "maximum_rois": args.maximum_rois,
        "minimum_half_size": args.minimum_half_size,
        "horizontal_player_scale": args.horizontal_player_scale,
        "upward_player_scale": args.upward_player_scale,
        "downward_player_scale": args.downward_player_scale,
        "predicted_player_scale": args.predicted_player_scale,
        "dedup_iou": args.dedup_iou,
    }

    records = []
    for index, frame in enumerate(target_frames, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Cannot read video frame {frame}")
        objects = object_rows[frame] if frame < len(object_rows) else {}
        predicted_center = _confirmed_previous_center(tracks_by_frame, frame)
        rois = build_ball_rescue_rois(
            objects,
            frame_shape,
            predicted_center=predicted_center,
            **roi_options,
        )
        started = perf_counter()
        raw_candidates = detector.detect_rois(
            image,
            [roi.bbox for roi in rois],
            maximum_upscale=args.maximum_upscale,
        )
        inference_seconds = perf_counter() - started
        gated_candidates = player_consistency_gate(
            raw_candidates,
            objects,
            distance_gate=args.player_distance_gate,
            size_factor_gate=args.player_size_factor_gate,
            minimum_players=args.gate_min_players,
        )
        gt = gt_by_frame.get(frame)
        pilot_items = pilot_by_frame.get(frame, [])
        records.append(
            {
                "frame": frame,
                "roi_count": len(rois),
                "rois": [
                    {
                        "bbox": [round(value, 3) for value in roi.bbox],
                        "source": roi.source,
                        "player_id": roi.player_id,
                    }
                    for roi in rois
                ],
                "inference_seconds": round(inference_seconds, 5),
                "raw_candidates": _serialize_candidates(raw_candidates),
                "gated_candidates": _serialize_candidates(gated_candidates),
                "gt_target": gt is not None,
                "gt_raw_match": _match(gt["bbox"], raw_candidates) if gt else None,
                "gt_gated_match": _match(gt["bbox"], gated_candidates) if gt else None,
                "pilot_items": [
                    {
                        "id": item["id"],
                        "raw_false_match": _match(
                            item["reference_bbox"], raw_candidates, radius=24.0
                        ),
                        "gated_false_match": _match(
                            item["reference_bbox"], gated_candidates, radius=24.0
                        ),
                    }
                    for item in pilot_items
                ],
            }
        )
        if index == 1 or index % 10 == 0 or index == len(target_frames):
            print(
                f"frame {index}/{len(target_frames)}: source={frame}, "
                f"rois={len(rois)}, candidates={len(raw_candidates)}->{len(gated_candidates)}, "
                f"inference={inference_seconds:.2f}s",
                flush=True,
            )
    cap.release()

    gt_records = [record for record in records if record["gt_target"]]
    pilot_results = [
        item for record in records for item in record["pilot_items"]
    ]
    timings = [record["inference_seconds"] for record in records]
    roi_counts = [record["roi_count"] for record in records]
    report = {
        "scope": (
            f"{args.gt_target_mode} human keyframes plus primary-empty pilot frames"
        ),
        "device": detector.device,
        "model": str(args.model),
        "maximum_upscale": args.maximum_upscale,
        "roi_options": roi_options,
        "player_gate": {
            "distance": args.player_distance_gate,
            "size_factor": args.player_size_factor_gate,
            "minimum_players": args.gate_min_players,
        },
        "processed_frames": len(records),
        "gt_trigger_frames": len(gt_records),
        "gt_raw_rescued": sum(record["gt_raw_match"] for record in gt_records),
        "gt_gated_rescued": sum(record["gt_gated_match"] for record in gt_records),
        "pilot_trigger_items": len(pilot_results),
        "pilot_raw_false_matches": sum(item["raw_false_match"] for item in pilot_results),
        "pilot_gated_false_matches": sum(
            item["gated_false_match"] for item in pilot_results
        ),
        "raw_candidates": sum(len(record["raw_candidates"]) for record in records),
        "gated_candidates": sum(
            len(record["gated_candidates"]) for record in records
        ),
        "roi_count": _summary([float(value) for value in roi_counts]),
        "inference_seconds_per_trigger_frame": _summary(timings),
        "inference_seconds_total": round(sum(timings), 3),
    }
    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates_out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {args.out} and {args.candidates_out}")


if __name__ == "__main__":
    main()
