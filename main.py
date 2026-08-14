import argparse
from pathlib import Path

from annotation import FootballVideoProcessor
from ball_to_player_assignment import BallToPlayerAssigner
from club_assignment import Club, ClubAssigner
from tracking import BallDetector, BallTracker, KeypointsTracker, ObjectTracker
from utils import process_video
from position_mappers import PitchGeometry


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
DEFAULT_BALL_MODEL = Path("models/weights/ball-detection_openvino_model_fp16")
if not DEFAULT_BALL_MODEL.exists():
    DEFAULT_BALL_MODEL = Path("models/weights/ball-detection_openvino_model")
if not DEFAULT_BALL_MODEL.exists():
    DEFAULT_BALL_MODEL = Path("models/weights/ball-detection.pt")
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


def _run_name(input_path: Path) -> str:
    """Return a stable output folder name for an input video."""
    name = input_path.stem
    for suffix in ("-input", "_input"):
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _resolve_output_layout(
    input_path: Path,
    run_dir: Path | None = None,
    output: Path | None = None,
    tracks_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve the run root, annotated video, and raw-track directory."""
    name = _run_name(input_path)
    resolved_run_dir = run_dir or Path("output_videos") / name
    resolved_output = output or resolved_run_dir / f"{name}-analysis.mp4"
    resolved_tracks_dir = tracks_dir or resolved_run_dir / "raw"
    return resolved_run_dir, resolved_output, resolved_tracks_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a football video with YOLO, ByteTrack, and pitch mapping."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input match video")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated output video",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run folder; defaults to output_videos/<input-name>",
    )
    parser.add_argument("--object-model", type=Path, default=DEFAULT_OBJECT_MODEL)
    parser.add_argument("--keypoints-model", type=Path, default=DEFAULT_KEYPOINTS_MODEL)
    parser.add_argument("--ball-model", type=Path, default=DEFAULT_BALL_MODEL)
    parser.add_argument("--field-image", type=Path, default=DEFAULT_FIELD_IMAGE)
    parser.add_argument(
        "--pitch-length-m",
        type=float,
        required=True,
        help="Measured touchline length of this pitch in metres",
    )
    parser.add_argument(
        "--pitch-width-m",
        type=float,
        required=True,
        help="Measured goal-line width of this pitch in metres",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=None,
        help="Raw JSONL directory; defaults to <run-dir>/raw",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Inference device: 'auto' probes OpenVINO iGPU/NPU for exported "
            "models ('intel:NPU', 'intel:GPU', 'intel:CPU' or a torch device "
            "like 'cpu')"
        ),
    )
    parser.add_argument("--skip-seconds", type=int, default=0)
    parser.add_argument(
        "--estimate-speed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Estimate and draw player and ball speed (disabled by default for sampled video)",
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
    parser.add_argument(
        "--referee-color",
        type=_rgb,
        default=None,
        metavar="R,G,B",
        help=(
            "Referee jersey reference color; when given, samples closest to "
            "this color are classified as referees instead of relying only on "
            "the distance to both club colors"
        ),
    )
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

    args.run_dir, args.output, args.tracks_dir = _resolve_output_layout(
        args.input,
        run_dir=args.run_dir,
        output=args.output,
        tracks_dir=args.tracks_dir,
    )

    _require_paths(
        parser,
        [
            args.input,
            args.object_model,
            args.keypoints_model,
            args.ball_model,
            args.field_image,
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tracks_dir.mkdir(parents=True, exist_ok=True)

    obj_tracker = ObjectTracker(
        model_path=str(args.object_model),
        conf=0.25,
        ball_conf=0.05,
        include_ball=False,
        device=args.device,
    )
    kp_tracker = KeypointsTracker(
        model_path=str(args.keypoints_model),
        conf=0.2,
        kp_conf=0.5,
        device=args.device,
    )

    club1 = Club(args.club1_name, args.club1_player, args.club1_goalkeeper)
    club2 = Club(args.club2_name, args.club2_player, args.club2_goalkeeper)
    club_assigner = ClubAssigner(club1, club2, referee_color=args.referee_color)
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    try:
        pitch_geometry = PitchGeometry(args.pitch_length_m, args.pitch_width_m)
    except ValueError as exc:
        parser.error(str(exc))

    ball_detector = BallDetector(str(args.ball_model), device=args.device)
    ball_tracker = BallTracker(pitch_geometry)

    processor = FootballVideoProcessor(
        obj_tracker,
        kp_tracker,
        ball_detector,
        ball_tracker,
        club_assigner,
        ball_player_assigner,
        pitch_geometry,
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
