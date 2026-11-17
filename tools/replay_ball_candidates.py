"""Replay stored ball candidates through ``BallTracker`` without inference.

This makes tracker-only experiments cheap and reproducible.  The input is a
``ball_tracks.jsonl`` produced by ``detect_ball_tracks.py``; its saved raw
candidates are reused while the old selected track is ignored.

Usage:
  python tools/replay_ball_candidates.py \
      --input output_videos/demo4/ball/ball_tracks.jsonl \
      --out output_videos/demo4/ball_replay/ball_tracks.jsonl \
      --object-tracks output_videos/demo4/raw/object_tracks.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from position_mappers import PitchGeometry
from position_mappers.camera_calibrator import CalibrationResult
from tracking import BallCandidate, BallTracker, player_consistency_gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--object-tracks", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--maximum-prediction-seconds", type=float, default=0.5)
    parser.add_argument("--restart-gap-seconds", type=float, default=0.15)
    parser.add_argument("--confirmations-required", type=int, default=3)
    parser.add_argument(
        "--player-distance-gate",
        type=float,
        default=None,
        help="experimental max distance / max(48px, 2.5 * nearest player height)",
    )
    parser.add_argument(
        "--player-size-factor-gate",
        type=float,
        default=None,
        help="experimental max symmetric factor from nearest player height / 8",
    )
    parser.add_argument("--gate-min-players", type=int, default=2)
    return parser


def _serialize_track(track: dict) -> dict:
    result = {
        "bbox": list(map(float, track["bbox"])),
        "confidence": float(track.get("confidence", 0.0)),
        "observed": bool(track.get("observed", True)),
        "track_confidence": float(track.get("track_confidence", 0.0)),
        "track_segment": int(track.get("track_segment", 0)),
        "track_confirmed": bool(track.get("track_confirmed", False)),
    }
    if track.get("position_m") is not None:
        result["position_m"] = list(map(float, track["position_m"]))
    return result


def _object_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Missing object tracks: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gate_candidates(
    candidates: list[BallCandidate],
    player_tracks: dict,
    distance_gate: float | None,
    size_factor_gate: float | None,
    minimum_players: int,
) -> list[BallCandidate]:
    return player_consistency_gate(
        candidates,
        player_tracks,
        distance_gate=distance_gate,
        size_factor_gate=size_factor_gate,
        minimum_players=minimum_players,
    )


def main() -> None:
    args = _parser().parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing candidates input: {args.input}")

    objects = _object_rows(args.object_tracks)
    tracker_options = {
        "maximum_prediction_seconds": args.maximum_prediction_seconds,
        "restart_gap_seconds": args.restart_gap_seconds,
        "confirmations_required": args.confirmations_required,
    }
    tracker = BallTracker(PitchGeometry(105.0, 68.0), **tracker_options)
    null_calibration = CalibrationResult(image_to_pitch=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    raw_candidate_count = 0
    gated_candidate_count = 0
    with args.input.open(encoding="utf-8") as source, args.out.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            if args.max_frames is not None and written >= args.max_frames:
                break
            row = json.loads(line)
            frame = int(row["frame"])
            candidates = [
                BallCandidate(tuple(map(float, candidate[:4])), float(candidate[4]))
                for candidate in row.get("candidates", [])
            ]
            player_tracks = objects[frame] if frame < len(objects) else {}
            raw_candidate_count += len(candidates)
            candidates = _gate_candidates(
                candidates,
                player_tracks,
                args.player_distance_gate,
                args.player_size_factor_gate,
                args.gate_min_players,
            )
            gated_candidate_count += len(candidates)
            selected = tracker.update(
                candidates,
                float(row["t"]),
                null_calibration,
                player_tracks,
                (args.height, args.width, 3),
            )
            track = selected.get(1) if selected else None
            output = {
                "frame": frame,
                "t": float(row["t"]),
                "candidates": [
                    [round(value, 3) for value in candidate.bbox]
                    + [round(candidate.confidence, 4)]
                    for candidate in candidates
                ],
                "track": _serialize_track(track) if track else None,
            }
            destination.write(
                json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            written += 1
    print(
        f"replayed {written} frames; candidates "
        f"{raw_candidate_count}->{gated_candidate_count} -> {args.out}"
    )


if __name__ == "__main__":
    main()
