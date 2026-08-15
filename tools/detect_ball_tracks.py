"""Run ball detection + image-space ball tracking on a video and write ball_tracks JSONL.

No object/keypoint models, no annotated video rendering: every frame is
processed with the production BallDetector (whole-frame + 2x2 overlapping tiled
inference, conf=0.02, NMS) and the production BallTracker. Produces one JSON
line per frame containing every raw detection candidate plus the tracker's
selected segment so downstream candidate cropping (confirmed / unconfirmed /
fn_sweep) can consume the file cheaply.

The tracker runs with a null pitch calibration (metric gates degrade to
image-space). This keeps the review set high-recall: off-pitch candidates stay
visible as review items instead of being dropped by the metric containment
gate.

Usage:
  python tools/detect_ball_tracks.py --video raw1.mp4 --device intel:NPU
  python tools/detect_ball_tracks.py --video raw2.mp4 --device intel:GPU
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from position_mappers import PitchGeometry
from position_mappers.camera_calibrator import CalibrationResult
from tracking import BallDetector, BallTracker

DEFAULT_MODEL = Path("models/weights/ball-detection_openvino_model_1280_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/ball-detection_openvino_model_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/ball-detection.pt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Defaults to output_videos/<video-stem>/ball")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto",
                        help="'auto' probes OpenVINO NPU/GPU/CPU for exported models")
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Process every Nth frame (default 1 = full)")
    parser.add_argument("--progress-interval", type=int, default=30)
    return parser


def _serialize_track(track: dict) -> dict:
    return {
        "bbox": list(map(float, track["bbox"])),
        "confidence": float(track.get("confidence", 0.0)),
        "observed": bool(track.get("observed", True)),
        "track_confidence": float(track.get("track_confidence", 0.0)),
        "track_segment": int(track.get("track_segment", 0)),
        "track_confirmed": bool(track.get("track_confirmed", False)),
    }


def main() -> None:
    args = _parser().parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Missing video: {args.video}")

    out_dir = args.out_dir or (Path("output_videos") / args.video.stem / "ball")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ball_tracks.jsonl"

    detector = BallDetector(str(args.model), confidence=args.conf, device=args.device)
    tracker = BallTracker(PitchGeometry(105.0, 68.0))
    null_calibration = CalibrationResult(image_to_pitch=None)
    print(f"device resolved: {detector.device}", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    written = 0
    if out_path.exists():
        written = sum(1 for _ in out_path.open(encoding="utf-8"))
        if written:
            print(f"resuming after {written} frames", flush=True)
    if written >= total_frames:
        print(f"already complete ({written} frames)", flush=True)
        cap.release()
        return

    start_frame = max(args.start_frame, written)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    processed = 0
    started = time.perf_counter()
    last_log = time.perf_counter()
    with out_path.open("a", encoding="utf-8", newline="\n") as out:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_index - start_frame) % args.frame_stride != 0:
                frame_index += 1
                continue
            timestamp = frame_index / fps
            candidates = detector.detect([frame])[0]
            ball_tracks = tracker.update(
                candidates,
                timestamp,
                null_calibration,
                {},
                frame.shape,
            )
            track = ball_tracks.get(1) if ball_tracks else None
            record = {
                "frame": frame_index,
                "t": round(timestamp, 4),
                "candidates": [
                    [round(v, 3) for v in c.bbox] + [round(c.confidence, 4)]
                    for c in candidates
                ],
                "track": _serialize_track(track) if track else None,
            }
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed += 1
            frame_index += 1

            now = time.perf_counter()
            if now - last_log >= args.progress_interval:
                elapsed = now - started
                rate = processed / max(elapsed, 1e-9)
                remaining = max(0, total_frames - frame_index) / max(rate, 1e-9)
                print(
                    f"frame {frame_index}/{total_frames} "
                    f"({processed / max(elapsed, 1e-9):.1f} fps, "
                    f"eta {remaining / 60:.0f} min)",
                    flush=True,
                )
                out.flush()
                last_log = now
    cap.release()
    elapsed = time.perf_counter() - started
    print(
        f"done: {processed} frames in {elapsed / 60:.1f} min "
        f"({processed / max(elapsed, 1e-9):.1f} fps) -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
