"""Build a provenance-preserving staging dataset from ball frame GT.

This tool can package incomplete annotation safely: human keyframes are kept
separate from interpolated boxes, entire source videos stay in one split, and
``dataset_status.json`` records why the result is or is not ready for training.
It never trains or promotes a model.

Example (current demo4 staging data):
  python tools/build_ball_finetune_dataset.py \
      --source demo4 \
      --video "C:/Personal Profile/Profile/Video/demo4.mp4" \
      --gt eval/ball_gt/demo4/frames_gt.jsonl \
      --split staging \
      --hard-negative-manifest eval/ball_crops/demo4/pilot/manifest.jsonl \
      --out eval/finetune_dataset/ball_partial_demo4
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


SPLITS = {"staging", "train", "val", "test"}
MINIMUM_POSITIVES = 600
MINIMUM_HARD_NEGATIVES = 300
MINIMUM_PITCHES = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--video", action="append", type=Path, required=True)
    parser.add_argument("--gt", action="append", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(SPLITS),
        help="One per source; defaults to staging for every source",
    )
    parser.add_argument(
        "--hard-negative-manifest",
        action="append",
        type=Path,
        default=[],
        help="Manifest whose every item has already been confirmed not_ball",
    )
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Also include derived interpolated GT rows (off by default)",
    )
    parser.add_argument(
        "--interpolated-stride",
        type=int,
        default=5,
        help="Keep every Nth interpolated frame when enabled",
    )
    parser.add_argument(
        "--max-interpolation-span",
        type=int,
        default=None,
        help="Only keep interpolation bracketed by human keyframes at most N frames apart",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/finetune_dataset/ball_partial"),
    )
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_gt_rows(
    rows: list[dict],
    include_interpolated: bool,
    interpolated_stride: int,
    max_interpolation_span: int | None = None,
) -> list[dict]:
    if interpolated_stride < 1:
        raise ValueError("interpolated_stride must be at least 1")
    if max_interpolation_span is not None and max_interpolation_span < 1:
        raise ValueError("max_interpolation_span must be at least 1")
    human_frames_by_segment: dict[int, list[int]] = {}
    for row in rows:
        if not row.get("interp", False):
            human_frames_by_segment.setdefault(int(row.get("seg", 0)), []).append(
                int(row["frame"])
            )
    for frames in human_frames_by_segment.values():
        frames.sort()
    selected = []
    seen_frames: set[int] = set()
    for row in sorted(rows, key=lambda item: int(item["frame"])):
        frame = int(row["frame"])
        if frame in seen_frames:
            raise ValueError(f"duplicate GT frame: {frame}")
        seen_frames.add(frame)
        interpolated = bool(row.get("interp", False))
        if interpolated and (
            not include_interpolated or frame % interpolated_stride != 0
        ):
            continue
        if interpolated and max_interpolation_span is not None:
            human_frames = human_frames_by_segment.get(int(row.get("seg", 0)), [])
            insertion = bisect_left(human_frames, frame)
            if insertion == 0 or insertion >= len(human_frames):
                continue
            if human_frames[insertion] - human_frames[insertion - 1] > max_interpolation_span:
                continue
        selected.append(row)
    return selected


def _yolo_box(
    bbox: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0 or len(bbox) != 4:
        raise ValueError("invalid frame size or bbox")
    x1, y1, x2, y2 = map(float, bbox)
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty bbox after clipping: {bbox}")
    return (
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _resolve_crop(manifest: Path, crop: str) -> Path:
    relative = Path(crop)
    candidates = [manifest.parent / relative, manifest.parent.parent / relative]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve crop {crop!r} relative to {manifest.parent}"
    )


def _dataset_status(rows: list[dict]) -> dict:
    sample_counts = Counter(row["sample_type"] for row in rows)
    sources = sorted({row["source"] for row in rows if row["sample_type"] != "hard_negative"})
    source_splits: dict[str, set[str]] = {}
    for row in rows:
        if row["sample_type"] == "hard_negative":
            continue
        source_splits.setdefault(row["source"], set()).add(row["split"])
    leaking_sources = sorted(
        source for source, splits in source_splits.items() if len(splits) > 1
    )
    positives = sample_counts["human_keyframe"] + sample_counts["interpolated"]
    human_keyframes = sample_counts["human_keyframe"]
    hard_negatives = sample_counts["hard_negative"]
    present_splits = sorted({row["split"] for row in rows})
    blockers = []
    if positives < MINIMUM_POSITIVES:
        blockers.append(f"positive frames {positives} < {MINIMUM_POSITIVES}")
    if human_keyframes < MINIMUM_POSITIVES:
        blockers.append(
            f"independent human keyframes {human_keyframes} < {MINIMUM_POSITIVES}; "
            "interpolated frames do not satisfy this gate"
        )
    if hard_negatives < MINIMUM_HARD_NEGATIVES:
        blockers.append(
            f"human-confirmed hard negatives {hard_negatives} < {MINIMUM_HARD_NEGATIVES}"
        )
    if len(sources) < MINIMUM_PITCHES:
        blockers.append(f"source pitches {len(sources)} < {MINIMUM_PITCHES}")
    if "staging" in present_splits or not {"train", "val", "test"}.issubset(present_splits):
        blockers.append("train/val/test source-level split is incomplete")
    if leaking_sources:
        blockers.append(f"sources cross splits: {', '.join(leaking_sources)}")
    return {
        "train_ready": not blockers,
        "model_promotion_allowed": False,
        "promotion_note": "Training output still requires unseen-pitch 75% recall / 90% precision validation.",
        "samples": len(rows),
        "positive_frames": positives,
        "human_keyframes": human_keyframes,
        "interpolated_frames": sample_counts["interpolated"],
        "human_confirmed_hard_negatives": hard_negatives,
        "sources": sources,
        "splits": present_splits,
        "blockers": blockers,
    }


def _write_positive_frames(
    source: str,
    video_path: Path,
    gt_path: Path,
    split: str,
    out: Path,
    include_interpolated: bool,
    interpolated_stride: int,
    max_interpolation_span: int | None,
    jpeg_quality: int,
) -> list[dict]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    rows = _select_gt_rows(
        _read_jsonl(gt_path),
        include_interpolated,
        interpolated_stride,
        max_interpolation_span,
    )
    selected = {int(row["frame"]): row for row in rows}
    if not selected:
        return []

    images_dir = out / "images" / split
    labels_dir = out / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    invalid = [frame for frame in selected if frame < 0 or frame >= total_frames]
    if invalid:
        cap.release()
        raise ValueError(f"GT frames outside video: {invalid[:10]}")

    manifest_rows = []
    final_frame = max(selected)
    frame = 0
    while frame <= final_frame:
        ok, image = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Video ended before selected frame {frame}")
        row = selected.get(frame)
        if row is not None:
            stem = f"{source}_{frame:08d}"
            image_relative = Path("images") / split / f"{stem}.jpg"
            label_relative = Path("labels") / split / f"{stem}.txt"
            if not cv2.imwrite(
                str(out / image_relative),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                cap.release()
                raise RuntimeError(f"Failed to write {out / image_relative}")
            box = _yolo_box(row["bbox"], width, height)
            (out / label_relative).write_text(
                "0 " + " ".join(f"{value:.8f}" for value in box) + "\n",
                encoding="utf-8",
            )
            sample_type = "interpolated" if row.get("interp", False) else "human_keyframe"
            manifest_rows.append(
                {
                    "image": image_relative.as_posix(),
                    "label": label_relative.as_posix(),
                    "source": source,
                    "source_video": str(video_path.resolve()),
                    "source_frame": frame,
                    "split": split,
                    "sample_type": sample_type,
                    "provenance": "derived_interpolation" if row.get("interp", False) else "human_box",
                    "human_verified": not bool(row.get("interp", False)),
                }
            )
        frame += 1
    cap.release()
    return manifest_rows


def _write_hard_negatives(
    manifests: list[Path], out: Path, source_splits: dict[str, str]
) -> list[dict]:
    rows = []
    seen_ids: set[str] = set()
    for manifest in manifests:
        for item in _read_jsonl(manifest):
            item_id = str(item["id"])
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            crop_path = _resolve_crop(manifest, str(item["crop"]))
            source = str(item.get("src") or item_id.split("_", 1)[0])
            split = source_splits.get(source, "staging")
            images_dir = out / "images" / split
            labels_dir = out / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            suffix = crop_path.suffix.lower() or ".png"
            image_relative = Path("images") / split / f"negative_{item_id}{suffix}"
            label_relative = Path("labels") / split / f"negative_{item_id}.txt"
            shutil.copy2(crop_path, out / image_relative)
            (out / label_relative).write_text("", encoding="utf-8")
            rows.append(
                {
                    "image": image_relative.as_posix(),
                    "label": label_relative.as_posix(),
                    "source": source,
                    "source_video": "",
                    "source_frame": int(item.get("frame", -1)),
                    "split": split,
                    "sample_type": "hard_negative",
                    "provenance": f"human_not_ball:{manifest.as_posix()}",
                    "human_verified": True,
                }
            )
    return rows


def main() -> None:
    args = _parser().parse_args()
    count = len(args.source)
    if not (count == len(args.video) == len(args.gt)):
        raise SystemExit("--source/--video/--gt must have the same count")
    splits = args.split or ["staging"] * count
    if len(splits) != count:
        raise SystemExit("--split must be omitted or supplied once per source")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    rows: list[dict] = []
    for source, video, gt, split in zip(args.source, args.video, args.gt, splits):
        rows.extend(
            _write_positive_frames(
                source,
                video,
                gt,
                split,
                args.out,
                args.include_interpolated,
                args.interpolated_stride,
                args.max_interpolation_span,
                args.jpeg_quality,
            )
        )
    source_splits = dict(zip(args.source, splits))
    rows.extend(
        _write_hard_negatives(
            args.hard_negative_manifest, args.out, source_splits
        )
    )
    rows.sort(key=lambda row: (row["split"], row["source"], row["source_frame"], row["image"]))

    args.out.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "label",
        "source",
        "source_video",
        "source_frame",
        "split",
        "sample_type",
        "provenance",
        "human_verified",
    ]
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    status = _dataset_status(rows)
    (args.out / "dataset_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"wrote staging dataset -> {args.out}")


if __name__ == "__main__":
    main()
