import argparse
from pathlib import Path

import numpy as np

from annotation import FootballVideoProcessor
from ball_to_player_assignment import BallToPlayerAssigner
from club_assignment import Club, ClubAssigner
from tracking import KeypointsTracker, ObjectTracker
from utils import process_video


DEFAULT_OBJECT_MODEL = Path("models/weights/object-detection_openvino_model_fp16")
if not DEFAULT_OBJECT_MODEL.exists():
    DEFAULT_OBJECT_MODEL = Path("models/weights/object-detection_openvino_model")
if not DEFAULT_OBJECT_MODEL.exists():
    DEFAULT_OBJECT_MODEL = Path("models/weights/object-detection.pt")

DEFAULT_KEYPOINTS_MODEL = Path("models/weights/keypoints-detection_openvino_model_fp16")
if not DEFAULT_KEYPOINTS_MODEL.exists():
    DEFAULT_KEYPOINTS_MODEL = Path("models/weights/keypoints-detection_openvino_model")
if not DEFAULT_KEYPOINTS_MODEL.exists():
    DEFAULT_KEYPOINTS_MODEL = Path("models/weights/keypoints-detection.pt")
DEFAULT_FIELD_IMAGE = Path("input_videos/field_2d_v2.png")


def _rgb(value: str) -> tuple[int, int, int]:
    """Parse an RGB value written as R,G,B."""
    try:
        channels = tuple(int(channel.strip()) for channel in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB values must be integers: R,G,B") from exc

    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError("RGB values must contain three channels from 0 to 255")
    return channels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a football video with YOLO, ByteTrack, and pitch mapping."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input match video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_videos/analysis.mp4"),
        help="Annotated output video",
    )
    parser.add_argument("--object-model", type=Path, default=DEFAULT_OBJECT_MODEL)
    parser.add_argument("--keypoints-model", type=Path, default=DEFAULT_KEYPOINTS_MODEL)
    parser.add_argument("--field-image", type=Path, default=DEFAULT_FIELD_IMAGE)
    parser.add_argument("--tracks-dir", type=Path, default=Path("output_videos"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-seconds", type=int, default=0)
    parser.add_argument(
        "--estimate-speed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Estimate and draw player speed (disabled by default for sampled video)",
    )
    parser.add_argument(
        "--annotate-possession",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw model-estimated possession (disabled by default)",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the live OpenCV preview; use --no-preview on a server",
    )
    parser.add_argument("--club1-name", default="Club1")
    parser.add_argument("--club1-player", type=_rgb, default=(232, 247, 248), metavar="R,G,B")
    parser.add_argument("--club1-goalkeeper", type=_rgb, default=(6, 25, 21), metavar="R,G,B")
    parser.add_argument("--club2-name", default="Club2")
    parser.add_argument("--club2-player", type=_rgb, default=(172, 251, 145), metavar="R,G,B")
    parser.add_argument("--club2-goalkeeper", type=_rgb, default=(239, 156, 132), metavar="R,G,B")
    return parser


def _require_paths(parser: argparse.ArgumentParser, paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        parser.error(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Run check_setup.py for details."
        )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.skip_seconds < 0:
        parser.error("--skip-seconds cannot be negative")

    _require_paths(
        parser,
        [args.input, args.object_model, args.keypoints_model, args.field_image],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tracks_dir.mkdir(parents=True, exist_ok=True)

    obj_tracker = ObjectTracker(
        model_path=str(args.object_model),
        conf=0.25,
        ball_conf=0.05,
    )
    kp_tracker = KeypointsTracker(
        model_path=str(args.keypoints_model),
        conf=0.2,
        kp_conf=0.5,
    )

    club1 = Club(args.club1_name, args.club1_player, args.club1_goalkeeper)
    club2 = Club(args.club2_name, args.club2_player, args.club2_goalkeeper)
    club_assigner = ClubAssigner(club1, club2)
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    top_down_keypoints = np.array(
        [
            [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],
            [32, 122], [32, 229], [64, 176],
            [96, 57], [96, 122], [96, 229], [96, 293],
            [263, 0], [263, 122], [263, 229], [263, 351],
            [431, 57], [431, 122], [431, 229], [431, 293],
            [463, 176], [495, 122], [495, 229],
            [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351],
            [210, 176], [317, 176],
        ],
        dtype=np.float32,
    )

    processor = FootballVideoProcessor(
        obj_tracker,
        kp_tracker,
        club_assigner,
        ball_player_assigner,
        top_down_keypoints,
        field_img_path=str(args.field_image),
        save_tracks_dir=str(args.tracks_dir),
        draw_frame_num=True,
        estimate_speed=args.estimate_speed,
        annotate_possession=args.annotate_possession,
    )

    process_video(
        processor,
        video_source=str(args.input),
        output_video=str(args.output),
        batch_size=args.batch_size,
        skip_seconds=args.skip_seconds,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
