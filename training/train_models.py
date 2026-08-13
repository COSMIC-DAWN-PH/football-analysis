from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from training.prepare_dataset import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune XbotGo ball or pitch models on a GPU")
    parser.add_argument("--task", choices=("ball", "pitch"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/xbotgo"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-export", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    validation = validate_dataset(args.data, args.task)
    if validation["errors"]:
        raise ValueError("Dataset validation failed:\n" + "\n".join(validation["errors"]))

    if args.task == "ball":
        model = YOLO("yolo11s.pt")
        train_options = {
            "task": "detect",
            "mosaic": 1.0,
            "close_mosaic": 20,
            "scale": 0.5,
            "fliplr": 0.5,
            "flipud": 0.0,
        }
    else:
        model = YOLO("models/weights/keypoints-detection.pt")
        train_options = {
            "task": "pose",
            "mosaic": 0.0,
            "scale": 0.35,
            "degrees": 0.0,
            "perspective": 0.0,
            "fliplr": 0.5,
            "flipud": 0.0,
        }

    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=40,
        imgsz=1280,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=f"xbotgo-{args.task}",
        plots=True,
        cache=False,
        **train_options,
    )
    best_path = Path(result.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    metrics = best_model.val(data=str(args.data), imgsz=1280, device=args.device)
    summary = {
        "task": args.task,
        "best_checkpoint": str(best_path),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "promoted": False,
        "note": "Weights are never copied into models/weights automatically; compare held-out-field metrics first.",
    }
    if not args.no_export:
        summary["openvino_export"] = str(
            best_model.export(format="openvino", half=True, imgsz=1280, batch=1, dynamic=True)
        )
    summary_path = Path(result.save_dir) / "training_result.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
