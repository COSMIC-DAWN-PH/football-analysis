"""Build a Roboflow-compatible YOLO fine-tuning dataset from sampled frames.

Frames are sampled from the three source videos and prelabeled with the
current object-detection model (classes: 0=ball, 1=goalkeeper, 2=player,
3=referee). Human-labeled crops (from tools/extract_referee_candidates.py)
correct the class of the matching bbox where YOLO is wrong, so the referee/
goalkeeper classes get labeled examples the base model keeps missing.

Label quality is tracked in manifest.csv: 'human' for corrections applied
from manual labels, 'auto' for model prelabels.

Usage:
  python tools/build_finetune_dataset.py \
      --source demo2 --video output_videos/demo2-30s-test/demo2-30s-test-input.mp4 \
      --tracks output_videos/demo2-30s-test/raw/object_tracks.jsonl \
      --labels eval/referee_crops/demo2/candidates.jsonl \
      --labels eval/referee_crops/demo2/verdict/candidates.jsonl \
      --source raw1 --video C:/.../raw1.mp4 --tracks output_videos/raw1/raw/object_tracks.jsonl \
      --source raw2 --video C:/.../raw2.mp4 --tracks output_videos/raw2/raw/object_tracks.jsonl \
      --out eval/finetune_dataset --interval 5 --device intel:GPU
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]
LABEL_TO_CLASS = {
    "referee": 3,
    "maroon": 2,
    "navy": 2,
    "maroon_gk": 1,
    "navy_gk": 1,
}
SPLITS = {"demo2": "test", "raw1": "train", "raw2": "val"}
DEFAULT_MODEL = Path("models/weights/object-detection_openvino_model_1280_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/object-detection_openvino_model_fp16")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = Path("models/weights/object-detection.pt")


def _box_iou(a, b):
    """IoU of two xywhn boxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="Source name (demo2/raw1/raw2)")
    parser.add_argument("--video", action="append", type=Path, required=True)
    parser.add_argument("--tracks", action="append", type=Path, required=True)
    parser.add_argument("--labels", action="append", type=Path, default=None, help="Candidate manifests with manual_label")
    parser.add_argument("--out", type=Path, default=Path("eval/finetune_dataset"))
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not (len(args.source) == len(args.video) == len(args.tracks)):
        raise SystemExit("--source/--video/--tracks must have the same length")

    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")

    out = args.out
    images_dir = out / "images"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load human corrections: {(source, frame): [(track_type, track_id, class_id)]}
    corrections = defaultdict(list)
    for label_path in args.labels or []:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            meta = json.loads(line)
            manual = meta.get("manual_label")
            if manual not in LABEL_TO_CLASS:
                continue
            corrections[(meta["source"], meta["frame"])].append(
                (meta["track_type"], int(meta["track_id"]), LABEL_TO_CLASS[manual])
            )

    manifest_rows = []
    n_images = 0
    for source, video_path, tracks_path in zip(args.source, args.video, args.tracks):
        if not video_path.is_file():
            print(f"missing video {video_path}, skipping {source}")
            continue
        track_lines = tracks_path.read_text(encoding="utf-8").splitlines()
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps * args.interval)))
        frames = list(range(0, total, step))
        # Always include frames that carry human corrections so the labeled
        # examples are baked into the dataset (sampling grid and crop frames
        # rarely coincide).
        for (src, fi), _ in corrections.items():
            if src == source and 0 <= fi < total and fi not in frames:
                frames.append(fi)
        frames.sort()
        print(f"{source}: sampling {len(frames)} frames (every {step} frames + {sum(1 for (s, f) in corrections if s == source)} correction frames)")

        for fi in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            result = model.predict(frame, conf=args.conf, imgsz=1280, verbose=False, device=args.device)[0]

            labels = []  # (cls, xywhn, quality)
            if result.boxes is not None:
                for xywhn, cls, conf in zip(
                    result.boxes.xywhn.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                ):
                    labels.append([int(cls), [float(v) for v in xywhn], "auto"])

            # Apply human corrections for this frame
            applied = 0
            for track_type, track_id, class_id in corrections.get((source, fi), []):
                if fi >= len(track_lines):
                    continue
                d = json.loads(track_lines[fi])
                tr = d.get(track_type, {}).get(str(track_id))
                if tr is None:
                    continue
                bbox = tr["bbox"]
                x1, y1, x2, y2 = (float(v) for v in bbox)
                H, W = frame.shape[:2]
                xywhn = [
                    (x1 + x2) / 2 / W,
                    (y1 + y2) / 2 / H,
                    (x2 - x1) / W,
                    (y2 - y1) / H,
                ]
                best_iou, best_idx = 0.0, -1
                for idx, (_, box, _) in enumerate(labels):
                    iou = _box_iou(box, xywhn)
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx
                if best_idx >= 0 and best_iou >= 0.3:
                    labels[best_idx][0] = class_id
                    labels[best_idx][2] = "human"
                else:
                    labels.append([class_id, xywhn, "human"])
                applied += 1

            if not labels:
                continue
            stem = f"{source}_{fi:08d}"
            image_name = f"{stem}.jpg"
            cv2.imwrite(str(images_dir / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for cls, xywhn, _quality in labels:
                lines.append(
                    f"{cls} " + " ".join(f"{v:.7f}" for v in xywhn)
                )
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest_rows.append(
                {
                    "image": image_name,
                    "source": source,
                    "source_video": str(video_path),
                    "source_frame": fi,
                    "split": SPLITS.get(source, "train"),
                    "n_labels": len(labels),
                    "human_corrections": applied,
                }
            )
            n_images += 1
        cap.release()

    # manifest.csv
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image", "source", "source_video", "source_frame", "split",
            "n_labels", "human_corrections",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # split lists + data.yaml
    for split in ("train", "val", "test"):
        rows = [r for r in manifest_rows if r["split"] == split]
        (out / f"{split}.txt").write_text(
            "\n".join(str(images_dir / r["image"]) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    (out / "data.yaml").write_text(
        "\n".join([
            f"path: {out.resolve().as_posix()}",
            "train: train.txt",
            "val: val.txt",
            "test: test.txt",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
            "",
        ]),
        encoding="utf-8",
    )

    human = sum(r["human_corrections"] for r in manifest_rows)
    print(f"dataset: {n_images} images, {human} human-corrected boxes -> {out}")


if __name__ == "__main__":
    main()
