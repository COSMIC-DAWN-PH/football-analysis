"""Run object detection + ByteTrack on a video and write tracks-only JSONL.

No keypoint/ball models, no annotated video rendering: this produces the same
`object_tracks.jsonl` frame format as the full pipeline so candidate cropping
and replay tools can consume raw footage cheaply.

Usage:
  python tools/detect_tracks_only.py --video raw1.mp4 --device intel:GPU
  python tools/detect_tracks_only.py --video raw2.mp4 --device intel:NPU
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from tracking import ObjectTracker

DEFAULT_MODEL = Path("models/weights/object-detection_openvino_model_1280_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/object-detection_openvino_model_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/object-detection.pt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Defaults to output_videos/<video-stem>/raw")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto",
                        help="'auto' probes NPU/GPU/CPU for OpenVINO exports")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Process every Nth frame (default 1 = full)")
    parser.add_argument("--progress-interval", type=int, default=300)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Missing video: {args.video}")

    out_dir = args.out_dir or (Path("output_videos") / args.video.stem / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "object_tracks.jsonl"

    tracker = ObjectTracker(
        model_path=str(args.model),
        conf=args.conf,
        ball_conf=0.05,
        include_ball=False,
        device=args.device,
    )
    print(f"device resolved: {tracker.device}", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    tracker.set_frame_rate(fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resume: skip frames already written.
    written = 0
    if out_path.exists():
        written = sum(1 for _ in out_path.open(encoding="utf-8"))
        if written:
            print(f"resuming after {written} frames", flush=True)
    if written >= total_frames:
        print(f"already complete ({written} frames)", flush=True)
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
            detections = tracker.detect([frame])
            tracks = tracker.track(detections[0])
            out.write(json.dumps(tracks, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed += 1
            frame_index += 1

            now = time.perf_counter()
            if now - last_log >= 30.0:
                elapsed = now - started
                rate = processed / elapsed
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
