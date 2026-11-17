"""Measure fixed dense tiling as a bounded ball-recall experiment.

Tile layouts and runtime triggers are chosen without looking at ball ground
truth. Human GT and the pilot false-target set are consulted only after model
inference to score recall and false matches. This tool never mutates production
tracker output or detector defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import hypot
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from tracking import BallCandidate, BallDetector, player_consistency_gate
from tracking.ball_tracker import non_max_suppression, overlapping_tiles


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
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.02)
    parser.add_argument(
        "--image-size",
        type=int,
        default=1280,
        help="Requested inference size (fixed exports may override it)",
    )
    parser.add_argument("--layouts", nargs="+", default=("2x3", "2x4"))
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument(
        "--trigger-mode",
        choices=("empty", "confirmed-gap"),
        default="empty",
    )
    parser.add_argument(
        "--max-gt-frames",
        type=int,
        default=20,
        help="Evenly spaced human keyframes selected after the runtime trigger",
    )
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--max-pilot-items", type=int, default=20)
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


def _parse_layout(value: str) -> tuple[int, int]:
    try:
        rows_text, columns_text = value.lower().split("x", 1)
        rows, columns = int(rows_text), int(columns_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid tile layout: {value!r}") from error
    if rows < 1 or columns < 1:
        raise ValueError(f"invalid tile layout: {value!r}")
    return rows, columns


def _triggered(row: dict, mode: str) -> bool:
    if mode == "empty":
        return not row.get("candidates")
    track = row.get("track")
    return bool(
        track
        and track.get("track_confirmed", False)
        and not track.get("observed", False)
    )


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


def _dense_candidates(
    detector,
    frame,
    rows: int,
    columns: int,
    overlap: float,
) -> list[BallCandidate]:
    candidates: list[BallCandidate] = []
    for tile, offset_x, offset_y in overlapping_tiles(
        frame, rows=rows, columns=columns, overlap=overlap
    ):
        candidates.extend(
            detector._predict(
                tile,
                offset_x,
                offset_y,
                detector.tile_imgsz,
            )
        )
    return non_max_suppression(candidates)


def _center(bbox) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2,
        (float(bbox[1]) + float(bbox[3])) / 2,
    )


def _match(
    reference_bbox,
    candidates: list[BallCandidate],
    radius: float | None = None,
) -> bool:
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


def main() -> None:
    args = _parser().parse_args()
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")
    layouts = {value: _parse_layout(value) for value in args.layouts}
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

    track_rows = _load(args.tracks)
    tracks_by_frame = {int(row["frame"]): row for row in track_rows}
    object_rows = _load(args.object_tracks)
    human_gt = [row for row in _load(args.gt) if not row.get("interp", False)]
    eligible_gt = [
        row
        for row in human_gt
        if _triggered(tracks_by_frame.get(int(row["frame"]), {}), args.trigger_mode)
    ]
    selected_gt = _evenly_spaced(eligible_gt, args.max_gt_frames)
    gt_by_frame = {int(row["frame"]): row for row in selected_gt}

    references_by_id = {row["id"]: row for row in _load(args.candidate_manifest)}
    pilot_rows = _load(args.pilot_manifest)
    if args.max_pilot_items is not None:
        pilot_rows = pilot_rows[: args.max_pilot_items]
    pilot_by_frame: dict[int, list[dict]] = {}
    for pilot in pilot_rows:
        frame_index = int(pilot["frame"])
        if not _triggered(tracks_by_frame.get(frame_index, {}), args.trigger_mode):
            continue
        item = dict(pilot)
        item["reference_bbox"] = references_by_id[item["id"]]["bbox"]
        pilot_by_frame.setdefault(frame_index, []).append(item)

    target_frames = sorted(set(gt_by_frame) | set(pilot_by_frame))
    detector = BallDetector(
        str(args.model),
        confidence=args.confidence,
        global_imgsz=args.image_size,
        tile_imgsz=args.image_size,
        device=args.device,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    layout_records: dict[str, list[dict]] = {name: [] for name in layouts}
    try:
        for index, frame_index in enumerate(target_frames, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read video frame {frame_index}")
            objects = object_rows[frame_index] if frame_index < len(object_rows) else {}
            gt = gt_by_frame.get(frame_index)
            pilot_items = pilot_by_frame.get(frame_index, [])
            for name, (rows, columns) in layouts.items():
                started = perf_counter()
                raw_candidates = _dense_candidates(
                    detector, frame, rows, columns, args.overlap
                )
                inference_seconds = perf_counter() - started
                gated_candidates = player_consistency_gate(
                    raw_candidates,
                    objects,
                    distance_gate=args.player_distance_gate,
                    size_factor_gate=args.player_size_factor_gate,
                    minimum_players=args.gate_min_players,
                )
                layout_records[name].append(
                    {
                        "frame": frame_index,
                        "gt_target": gt is not None,
                        "gt_raw_match": _match(gt["bbox"], raw_candidates)
                        if gt
                        else None,
                        "gt_gated_match": _match(gt["bbox"], gated_candidates)
                        if gt
                        else None,
                        "pilot_raw_false_matches": sum(
                            _match(
                                item["reference_bbox"], raw_candidates, radius=24.0
                            )
                            for item in pilot_items
                        ),
                        "pilot_gated_false_matches": sum(
                            _match(
                                item["reference_bbox"], gated_candidates, radius=24.0
                            )
                            for item in pilot_items
                        ),
                        "raw_candidates": len(raw_candidates),
                        "gated_candidates": len(gated_candidates),
                        "inference_seconds": round(inference_seconds, 5),
                    }
                )
            if index == 1 or index % 10 == 0 or index == len(target_frames):
                print(
                    f"frame {index}/{len(target_frames)}: source={frame_index}",
                    flush=True,
                )
    finally:
        capture.release()

    runtime_trigger_frames = sum(
        _triggered(row, args.trigger_mode) for row in track_rows
    )
    trigger_rate = runtime_trigger_frames / len(track_rows) if track_rows else 0.0
    summaries = {}
    for name, records in layout_records.items():
        rows, columns = layouts[name]
        gt_records = [record for record in records if record["gt_target"]]
        timings = [record["inference_seconds"] for record in records]
        summaries[name] = {
            "rows": rows,
            "columns": columns,
            "tile_calls_per_trigger_frame": rows * columns,
            "estimated_extra_compute_vs_five_primary_calls": round(
                trigger_rate * rows * columns / 5.0, 4
            ),
            "gt_raw_hits": sum(record["gt_raw_match"] for record in gt_records),
            "gt_gated_hits": sum(
                record["gt_gated_match"] for record in gt_records
            ),
            "pilot_raw_false_matches": sum(
                record["pilot_raw_false_matches"] for record in records
            ),
            "pilot_gated_false_matches": sum(
                record["pilot_gated_false_matches"] for record in records
            ),
            "raw_candidates": sum(record["raw_candidates"] for record in records),
            "gated_candidates": sum(
                record["gated_candidates"] for record in records
            ),
            "inference_seconds_total": round(sum(timings), 3),
            "inference_seconds_per_trigger_frame": round(
                sum(timings) / len(timings), 4
            )
            if timings
            else None,
        }

    report = {
        "scope": "fixed dense layouts selected without GT; GT scores output only",
        "device": detector.device,
        "requested_image_size": args.image_size,
        "resolved_image_size": detector.tile_imgsz,
        "model": str(args.model),
        "trigger_mode": args.trigger_mode,
        "runtime_frames": len(track_rows),
        "runtime_trigger_frames": runtime_trigger_frames,
        "runtime_trigger_rate": round(trigger_rate, 4),
        "eligible_human_keyframes": len(eligible_gt),
        "evaluated_human_keyframes": len(selected_gt),
        "pilot_trigger_items": sum(len(items) for items in pilot_by_frame.values()),
        "overlap": args.overlap,
        "player_gate": {
            "distance": args.player_distance_gate,
            "size_factor": args.player_size_factor_gate,
            "minimum_players": args.gate_min_players,
        },
        "layouts": summaries,
        "details": layout_records,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
