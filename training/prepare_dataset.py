from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PITCH_FLIP_INDEX = [
    24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15,
    16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30,
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and validate XbotGo YOLO datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract diverse annotation candidates")
    extract.add_argument("--task", choices=("ball", "pitch"), required=True)
    extract.add_argument("--videos", type=Path, nargs="+", required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--total-frames", type=int)
    extract.add_argument("--model", type=Path, help="Optional model for low-confidence prelabels")

    unpack = subparsers.add_parser("unpack", help="Unpack a Roboflow YOLO export")
    unpack.add_argument("--zip", type=Path, required=True)
    unpack.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate a local/Roboflow YOLO dataset")
    validate.add_argument("--task", choices=("ball", "pitch"), required=True)
    validate.add_argument("--data", type=Path, required=True, help="Dataset data.yaml")
    validate.add_argument("--report", type=Path)

    mine = subparsers.add_parser(
        "mine-hard-negatives",
        help="Export likely white-line/low-confidence false positives for manual review",
    )
    mine.add_argument("--video", type=Path, required=True)
    mine.add_argument("--object-tracks", type=Path, required=True)
    mine.add_argument("--output-dir", type=Path, required=True)
    mine.add_argument("--max-frames", type=int, default=800)
    return parser


def _average_hash(frame: np.ndarray) -> int:
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (8, 8))
    return int("".join("1" if value >= gray.mean() else "0" for value in gray.flat), 2)


def _is_near_duplicate(value: int, previous: Iterable[int]) -> bool:
    return any((value ^ existing).bit_count() <= 3 for existing in previous)


def extract_frames(args: argparse.Namespace) -> None:
    videos = [path.resolve() for path in args.videos]
    missing = [str(path) for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing video(s): {', '.join(missing)}")
    total_frames = args.total_frames or (800 if args.task == "ball" else 400)
    per_video = max(1, int(np.ceil(total_frames / len(videos))))
    image_dir = args.output_dir / "images"
    label_dir = args.output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.model:
        from ultralytics import YOLO

        model = YOLO(str(args.model), task="detect" if args.task == "ball" else "pose")

    manifest_rows = []
    extracted = 0
    for video_index, video_path in enumerate(videos):
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        indexes = np.linspace(0, max(0, frame_count - 1), per_video * 2, dtype=int)
        hashes: list[int] = []
        accepted_for_video = 0
        for frame_index in indexes:
            if extracted >= total_frames or accepted_for_video >= per_video:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            frame_hash = _average_hash(frame)
            if _is_near_duplicate(frame_hash, hashes):
                continue
            hashes.append(frame_hash)
            stem = f"{args.task}-{video_index:02d}-{video_path.stem}-{frame_index:08d}"
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            labels = _prelabel(frame, model, args.task) if model is not None else []
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            manifest_rows.append(
                {
                    "image": image_path.name,
                    "source_video": str(video_path),
                    "source_frame": int(frame_index),
                    "timestamp_seconds": frame_index / fps,
                    "task": args.task,
                    "prelabels": len(labels),
                }
            )
            accepted_for_video += 1
            extracted += 1
        capture.release()

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)
    print(f"Extracted {extracted} {args.task} annotation candidates to {args.output_dir}")


def _prelabel(frame: np.ndarray, model, task: str) -> list[str]:
    result = model.predict(frame, conf=0.01 if task == "ball" else 0.10, imgsz=1280, verbose=False)[0]
    if task == "ball":
        labels = []
        if result.boxes is not None:
            for box, class_id in zip(result.boxes.xywhn.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(class_id) == 0:
                    labels.append("0 " + " ".join(f"{float(value):.7f}" for value in box))
        return labels
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return []
    box = result.boxes.xywhn[0].cpu().numpy()
    xy = result.keypoints.xyn[0].cpu().numpy()
    confidence = result.keypoints.conf[0].cpu().numpy()
    values = ["0", *(f"{float(value):.7f}" for value in box)]
    for point, score in zip(xy, confidence):
        if float(score) >= 0.20:
            values.extend((f"{float(point[0]):.7f}", f"{float(point[1]):.7f}", "2"))
        else:
            values.extend(("0", "0", "0"))
    return [" ".join(values)]


def unpack_export(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        output_root = args.output_dir.resolve()
        for member in archive.infolist():
            destination = (output_root / member.filename).resolve()
            if output_root not in destination.parents and destination != output_root:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(args.output_dir)
    data_files = list(args.output_dir.rglob("data.yaml"))
    if len(data_files) != 1:
        raise ValueError(f"Expected one data.yaml in export, found {len(data_files)}")
    print(data_files[0])


def mine_hard_negatives(args: argparse.Namespace) -> None:
    if not args.video.is_file() or not args.object_tracks.is_file():
        raise FileNotFoundError("Video and object_tracks.jsonl must both exist")
    image_dir = args.output_dir / "images"
    label_dir = args.output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Could not open {args.video}")
    rows = []
    with args.object_tracks.open(encoding="utf-8") as handle:
        for frame_index, line in enumerate(handle):
            ok, frame = capture.read()
            if not ok or len(rows) >= args.max_frames:
                break
            payload = json.loads(line)
            balls = payload.get("ball", {})
            selected = next(iter(balls.values()), None)
            if selected is None or not selected.get("observed", False):
                continue
            reasons = set(selected.get("rejection_reasons", []))
            likely_hard_negative = (
                "pitch_marking_overlap" in reasons
                or "low_ball_confidence" in reasons
                or float(selected.get("line_score", 0.0)) >= 0.50
            )
            if not likely_hard_negative:
                continue
            stem = f"hard-negative-{args.video.stem}-{frame_index:08d}"
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_path.write_text("", encoding="utf-8")
            rows.append(
                {
                    "image": image_path.name,
                    "source_video": str(args.video.resolve()),
                    "source_frame": frame_index,
                    "reasons": "|".join(sorted(reasons)),
                    "candidate_bbox": json.dumps(selected.get("bbox")),
                    "review_required": "true",
                }
            )
    capture.release()
    manifest_path = args.output_dir / "hard_negative_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0]) if rows else [
            "image",
            "source_video",
            "source_frame",
            "reasons",
            "candidate_bbox",
            "review_required",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Exported {len(rows)} review-required hard-negative candidates to {args.output_dir}"
    )


def validate_dataset(data_path: Path, task: str) -> dict:
    data_path = data_path.resolve()
    with data_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    errors: list[str] = []
    warnings: list[str] = []
    if task == "pitch" and config.get("flip_idx") != PITCH_FLIP_INDEX:
        errors.append(f"data.yaml flip_idx must equal {PITCH_FLIP_INDEX}")

    configured_root = Path(config.get("path", data_path.parent))
    dataset_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (data_path.parent / configured_root).resolve()
    )
    split_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}
    positive_images = 0
    hashes: dict[str, tuple[str, Path]] = {}
    for split in ("train", "val", "test"):
        entries = config.get(split)
        if entries is None:
            errors.append(f"Missing '{split}' split")
            continue
        if not isinstance(entries, list):
            entries = [entries]
        images: list[Path] = []
        for entry in entries:
            path = (dataset_root / entry).resolve()
            if path.is_file() and path.suffix.lower() == ".txt":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        listed = Path(line.strip())
                        images.append(
                            listed.resolve()
                            if listed.is_absolute()
                            else (path.parent / listed).resolve()
                        )
            elif path.is_dir():
                images.extend(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_EXTENSIONS)
            else:
                errors.append(f"Invalid {split} path: {path}")
        split_counts[split] = len(images)
        split_annotations = 0
        for image in images:
            if not image.is_file():
                errors.append(f"Missing image: {image}")
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest][0] != split:
                errors.append(f"Duplicate image across splits: {image} and {hashes[digest][1]}")
            else:
                hashes[digest] = (split, image)
            label = _label_path(image)
            if not label.is_file():
                errors.append(f"Missing label: {label}")
                continue
            annotations = _validate_label(label, task, errors)
            split_annotations += annotations
            positive_images += int(annotations > 0)
        annotation_counts[split] = split_annotations
    total = sum(split_counts.values())
    if total >= 20:
        split_ratios = {split: count / total for split, count in split_counts.items()}
        for split, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            if abs(split_ratios.get(split, 0.0) - target) > 0.10:
                warnings.append(
                    f"{split} is {split_ratios.get(split, 0.0):.1%}; target is approximately {target:.0%}"
                )
    _validate_manifest(dataset_root, errors, warnings)
    if task == "ball" and total < 800:
        warnings.append(f"Ball dataset has {total} frames; first-round target is 800")
    if task == "ball" and positive_images < 600:
        warnings.append(
            f"Ball dataset has {positive_images} positive frames; first-round target is at least 600"
        )
    if task == "ball" and total == positive_images:
        warnings.append("Ball dataset has no empty/hard-negative frames")
    if task == "pitch" and total < 400:
        warnings.append(f"Pitch dataset has {total} frames; first-round target is 400")
    return {
        "task": task,
        "data": str(data_path),
        "splits": split_counts,
        "annotations": annotation_counts,
        "positive_images": positive_images,
        "errors": errors,
        "warnings": warnings,
    }


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _validate_label(path: Path, task: str, errors: list[str]) -> int:
    expected = 5 if task == "ball" else 5 + 32 * 3
    annotations = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != expected:
            errors.append(f"{path}:{line_number} has {len(fields)} fields; expected {expected}")
            continue
        annotations += 1
        try:
            values = [float(value) for value in fields]
        except ValueError:
            errors.append(f"{path}:{line_number} contains a non-numeric value")
            continue
        if int(values[0]) != 0:
            errors.append(f"{path}:{line_number} class must be 0")
        coordinates = values[1:5] if task == "ball" else values[1:5] + [value for index, value in enumerate(values[5:]) if index % 3 != 2]
        if any(value < 0 or value > 1 for value in coordinates):
            errors.append(f"{path}:{line_number} has coordinates outside [0, 1]")
        if task == "pitch" and any(values[index] not in (0, 1, 2) for index in range(7, len(values), 3)):
            errors.append(f"{path}:{line_number} has invalid keypoint visibility")
    return annotations


def _validate_manifest(
    dataset_root: Path, errors: list[str], warnings: list[str]
) -> None:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        warnings.append(
            "No manifest.csv found; whole-video/pitch split leakage could not be verified"
        )
        return
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "split" not in rows[0]:
        warnings.append(
            "manifest.csv has no split column; whole-video/pitch leakage could not be verified"
        )
        return

    for group_name in ("source_video", "pitch"):
        if group_name not in rows[0]:
            if group_name == "pitch":
                warnings.append(
                    "manifest.csv has no pitch column; unseen-pitch test coverage could not be verified"
                )
            continue
        groups: dict[str, set[str]] = {}
        for row in rows:
            value, split = row.get(group_name, "").strip(), row.get("split", "").strip()
            if value and split:
                groups.setdefault(value, set()).add(split)
        for value, splits in groups.items():
            if len(splits) > 1:
                errors.append(
                    f"{group_name} '{value}' crosses splits: {sorted(splits)}"
                )
        if group_name == "pitch":
            train_pitches = {row["pitch"] for row in rows if row.get("split") == "train"}
            test_pitches = {row["pitch"] for row in rows if row.get("split") == "test"}
            if not (test_pitches - train_pitches):
                errors.append("Test split must contain at least one pitch unseen in train")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "extract":
        extract_frames(args)
    elif args.command == "unpack":
        unpack_export(args)
    elif args.command == "mine-hard-negatives":
        mine_hard_negatives(args)
    else:
        report = validate_dataset(args.data, args.task)
        output = json.dumps(report, ensure_ascii=False, indent=2)
        print(output)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output + "\n", encoding="utf-8")
        if report["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
