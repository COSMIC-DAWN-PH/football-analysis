from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from position_mappers import CameraProfile, PitchAnchorSet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Football-Analysis camera and pitch calibration artifacts"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    camera = commands.add_parser("camera", help="Calibrate one locked camera/focal setting")
    camera.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Checkerboard paths or glob patterns; quote globs in PowerShell",
    )
    camera.add_argument("--columns", type=int, required=True, help="Checkerboard inner corners across")
    camera.add_argument("--rows", type=int, required=True, help="Checkerboard inner corners down")
    camera.add_argument("--square-size-m", type=float, required=True)
    camera.add_argument("--profile-id", required=True)
    camera.add_argument("--focal-setting", default="locked")
    camera.add_argument("--output", type=Path, required=True)

    pitch = commands.add_parser("pitch", help="Create Pitch Anchor Set for a continuous shot")
    source = pitch.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--image", type=Path)
    pitch.add_argument("--frame", type=int, default=0)
    pitch.add_argument(
        "--pitch-points",
        required=True,
        help='Ordered metric points, for example "0,0;52.5,0;105,0;105,68;52.5,68;0,68"',
    )
    pitch.add_argument(
        "--image-points",
        help="Optional ordered image points in the same format; omit for interactive clicking",
    )
    pitch.add_argument("--output", type=Path, required=True)
    return parser


def _parse_points(value: str) -> np.ndarray:
    points = []
    for item in value.split(";"):
        channels = [float(channel.strip()) for channel in item.split(",")]
        if len(channels) != 2:
            raise ValueError(f"Invalid point '{item}'; expected x,y")
        points.append(channels)
    return np.asarray(points, dtype=np.float64)


def _calibrate_camera(args: argparse.Namespace) -> int:
    if args.columns < 2 or args.rows < 2 or args.square_size_m <= 0:
        raise ValueError("Checkerboard dimensions and square size must be positive")
    object_template = np.zeros((args.columns * args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
    object_template *= float(args.square_size_m)
    object_points = []
    image_points = []
    image_size = None
    image_paths: list[Path] = []
    for value in args.images:
        matches = [Path(path) for path in glob.glob(value)]
        image_paths.extend(matches or [Path(value)])
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read checkerboard image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError("All checkerboard images must have the same resolution")
        found, corners = cv2.findChessboardCornersSB(
            gray, (args.columns, args.rows), flags=cv2.CALIB_CB_EXHAUSTIVE
        )
        if not found:
            continue
        object_points.append(object_template.copy())
        image_points.append(corners.reshape(-1, 2).astype(np.float32))
    if image_size is None or len(object_points) < 8:
        raise ValueError(
            f"Camera calibration needs at least 8 usable checkerboard images; found {len(object_points)}"
        )
    rms, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    profile = CameraProfile(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        image_width=image_size[0],
        image_height=image_size[1],
        profile_id=args.profile_id,
        focal_setting=args.focal_setting,
        rms_error_px=float(rms),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.save(args.output)
    print(args.output)
    return 0


def _read_reference_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(f"Could not read reference image: {args.image}")
        return image
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not read reference video: {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {args.frame} from {args.video}")
    return frame


def _interactive_points(frame: np.ndarray, count: int) -> np.ndarray:
    points: list[tuple[float, float]] = []
    window = "Pitch anchors: click in listed order; Backspace undo; Esc cancel"

    def click(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, click)
    while len(points) < count:
        canvas = frame.copy()
        for index, point in enumerate(points):
            pixel = tuple(map(int, point))
            cv2.circle(canvas, pixel, 6, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(index + 1),
                (pixel[0] + 8, pixel[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                str(index + 1),
                (pixel[0] + 8, pixel[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Pitch anchor selection cancelled")
        if key in (8, 127) and points:
            points.pop()
    cv2.destroyWindow(window)
    return np.asarray(points, dtype=np.float64)


def _calibrate_pitch(args: argparse.Namespace) -> int:
    pitch_points = _parse_points(args.pitch_points)
    if not 4 <= len(pitch_points) <= 8:
        raise ValueError("Pitch Anchor Set requires 4 to 8 points")
    frame = _read_reference_frame(args)
    image_points = (
        _parse_points(args.image_points)
        if args.image_points
        else _interactive_points(frame, len(pitch_points))
    )
    anchors = PitchAnchorSet(
        image_points=image_points,
        pitch_points=pitch_points,
        reference_frame=int(args.frame),
        source="manual",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anchors.save(args.output)
    print(args.output)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "camera":
        return _calibrate_camera(args)
    return _calibrate_pitch(args)
