from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from training.prepare_dataset import validate_dataset
from tracking import (
    validate_ball_model_for_promotion,
    validate_keypoint_model_for_promotion,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def select_precision_first_threshold(
    scores: list[float],
    labels: list[bool],
    *,
    minimum_recall: float = 0.75,
) -> dict[str, float]:
    """Choose the highest-precision threshold that preserves required recall."""
    if len(scores) != len(labels) or not scores or not any(labels):
        raise ValueError("Threshold selection needs paired scores and positive samples")
    best: tuple[float, float, float] | None = None
    for threshold in sorted(set(map(float, scores)), reverse=True):
        predictions = [score >= threshold for score in scores]
        true_positive = sum(predicted and label for predicted, label in zip(predictions, labels))
        false_positive = sum(predicted and not label for predicted, label in zip(predictions, labels))
        false_negative = sum(not predicted and label for predicted, label in zip(predictions, labels))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        if recall >= minimum_recall:
            candidate = (precision, threshold, recall)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise ValueError(f"No verifier threshold reaches recall {minimum_recall:.0%}")
    return {
        "decision_threshold": best[1],
        "precision": best[0],
        "recall": best[2],
    }


def _classifier_scores(model: YOLO, root: Path, device: str) -> tuple[list[float], list[bool]]:
    paths: list[Path] = []
    labels: list[bool] = []
    for class_name, label in (("ball", True), ("non_ball", False)):
        class_paths = sorted(
            path
            for path in (root / class_name).rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    if not paths:
        raise ValueError(f"No classifier images found under {root}")
    names = model.names
    normalized = {
        str(name).casefold(): int(index) for index, name in names.items()
    } if isinstance(names, dict) else {
        str(name).casefold(): index for index, name in enumerate(names)
    }
    if "ball" not in normalized:
        raise ValueError("Ball verifier classes must include 'ball'")
    ball_index = normalized["ball"]
    results = model.predict(
        [str(path) for path in paths],
        imgsz=128,
        device=device,
        verbose=False,
    )
    scores = [
        float(result.probs.data.detach().cpu().numpy()[ball_index])
        for result in results
    ]
    return scores, labels


def _evaluate_threshold(
    scores: list[float], labels: list[bool], threshold: float
) -> dict[str, float]:
    predictions = [score >= threshold for score in scores]
    true_positive = sum(predicted and label for predicted, label in zip(predictions, labels))
    false_positive = sum(predicted and not label for predicted, label in zip(predictions, labels))
    false_negative = sum(not predicted and label for predicted, label in zip(predictions, labels))
    return {
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune XbotGo ball or pitch models on a GPU")
    parser.add_argument("--task", choices=("ball", "ball-verifier", "pitch"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/xbotgo"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-export", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.task == "ball-verifier":
        missing = [
            str(args.data / split / class_name)
            for split in ("train", "val", "test")
            for class_name in ("ball", "non_ball")
            if not (args.data / split / class_name).is_dir()
        ]
        if missing:
            raise ValueError(
                "Ball verifier dataset must contain train/val/test ball and non_ball folders:\n"
                + "\n".join(missing)
            )
    else:
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
    elif args.task == "ball-verifier":
        model = YOLO("yolo11n-cls.pt")
        train_options = {
            "task": "classify",
            "scale": 0.35,
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
        imgsz=128 if args.task == "ball-verifier" else 1280,
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
    validation_imgsz = 128 if args.task == "ball-verifier" else 1280
    metrics = best_model.val(data=str(args.data), imgsz=validation_imgsz, device=args.device)
    summary = {
        "task": args.task,
        "best_checkpoint": str(best_path),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "promoted": False,
        "note": "Weights are never copied into models/weights automatically; compare held-out-field metrics first.",
    }
    verifier_threshold_payload = None
    if args.task == "ball-verifier":
        validation_scores, validation_labels = _classifier_scores(
            best_model, args.data / "val", args.device
        )
        verifier_threshold_payload = select_precision_first_threshold(
            validation_scores,
            validation_labels,
        )
        test_scores, test_labels = _classifier_scores(
            best_model, args.data / "test", args.device
        )
        summary["verifier_operating_point"] = {
            **verifier_threshold_payload,
            "test": _evaluate_threshold(
                test_scores,
                test_labels,
                verifier_threshold_payload["decision_threshold"],
            ),
        }
        best_path.with_suffix(".verifier.json").write_text(
            json.dumps(verifier_threshold_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if not args.no_export:
        summary["openvino_export"] = str(
            best_model.export(
                format="openvino",
                half=True,
                imgsz=validation_imgsz,
                batch=1,
                dynamic=True,
            )
        )
        if args.task == "ball":
            summary["promotion_errors"] = validate_ball_model_for_promotion(
                summary["openvino_export"]
            )
        elif args.task == "pitch":
            summary["promotion_errors"] = validate_keypoint_model_for_promotion(
                summary["openvino_export"]
            )
        else:
            summary["promotion_errors"] = []
        if args.task == "ball-verifier" and verifier_threshold_payload is not None:
            (Path(summary["openvino_export"]) / "verifier_threshold.json").write_text(
                json.dumps(verifier_threshold_payload, indent=2) + "\n",
                encoding="utf-8",
            )
    precision = float(summary["metrics"].get("metrics/precision(B)", 0.0))
    recall = float(summary["metrics"].get("metrics/recall(B)", 0.0))
    summary["promotion_thresholds"] = {
        "minimum_precision": 0.95 if args.task == "ball" else None,
        "minimum_recall": 0.75 if args.task == "ball" else None,
    }
    summary["promotion_eligible"] = bool(
        (
            args.task == "ball-verifier"
            and summary.get("verifier_operating_point", {})
            .get("test", {})
            .get("recall", 0.0) >= 0.75
            and not summary.get("promotion_errors", [])
        )
        or (
            args.task == "pitch"
            and not args.no_export
            and not summary.get("promotion_errors", [])
        )
        or (
            args.task == "ball"
            and not args.no_export
            and precision >= 0.95
            and recall >= 0.75
            and not summary.get("promotion_errors", [])
        )
    )
    summary_path = Path(result.save_dir) / "training_result.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
