import shutil
import tempfile
import unittest
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from training.prepare_dataset import PITCH_FLIP_INDEX, mine_hard_negatives, validate_dataset
from training.train_models import select_precision_first_threshold


class DatasetValidationTests(unittest.TestCase):
    def _dataset(self, root: Path, use_train_list: bool = False) -> Path:
        for index, split in enumerate(("train", "val", "test")):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image_path = image_dir / f"{split}.jpg"
            image = np.full((16, 16, 3), index * 50, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            (label_dir / f"{split}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )

        config = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test"}
        if use_train_list:
            list_dir = root / "lists"
            list_dir.mkdir()
            (list_dir / "train.txt").write_text(
                "../images/train/train.jpg\n", encoding="utf-8"
            )
            config["train"] = "lists/train.txt"
        data_path = root / "data.yaml"
        data_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return data_path

    def test_local_dataset_and_relative_image_list_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = validate_dataset(self._dataset(Path(temporary), True), "ball")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["splits"], {"train": 1, "val": 1, "test": 1})

    def test_exact_duplicate_across_splits_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = self._dataset(root)
            shutil.copyfile(root / "images/train/train.jpg", root / "images/val/val.jpg")
            report = validate_dataset(data_path, "ball")
        self.assertTrue(any("Duplicate image across splits" in error for error in report["errors"]))

    def test_pitch_flip_permutation_is_an_involution(self) -> None:
        self.assertEqual(len(PITCH_FLIP_INDEX), 32)
        self.assertEqual(
            [PITCH_FLIP_INDEX[index] for index in PITCH_FLIP_INDEX],
            list(range(32)),
        )

    def test_verifier_threshold_maximizes_precision_at_required_recall(self) -> None:
        result = select_precision_first_threshold(
            [0.90, 0.80, 0.30, 0.20, 0.85, 0.10],
            [True, True, True, True, False, False],
            minimum_recall=0.75,
        )
        self.assertEqual(result["decision_threshold"], 0.20)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 0.80)

    def test_manifest_rejects_video_and_pitch_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = self._dataset(root)
            with (root / "manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("image", "source_video", "source_frame", "split", "pitch"),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"image": "train.jpg", "source_video": "match-a", "source_frame": 1, "split": "train", "pitch": "A"},
                        {"image": "val.jpg", "source_video": "match-a", "source_frame": 2, "split": "val", "pitch": "A"},
                        {"image": "test.jpg", "source_video": "match-b", "source_frame": 3, "split": "test", "pitch": "A"},
                    ]
                )
            report = validate_dataset(data_path, "ball")
        self.assertTrue(any("source_video 'match-a' crosses splits" in error for error in report["errors"]))
        self.assertTrue(any("pitch 'A' crosses splits" in error for error in report["errors"]))
        self.assertIn("Test split must contain at least one pitch unseen in train", report["errors"])

    def test_hard_negative_mining_exports_review_required_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            tracks_path = root / "object_tracks.jsonl"
            tracks_path.write_text(
                json.dumps(
                    {
                        "ball": {
                            "1": {
                                "bbox": [10, 10, 15, 15],
                                "observed": True,
                                "line_score": 0.9,
                                "rejection_reasons": ["pitch_marking_overlap"],
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "mined"
            mine_hard_negatives(
                SimpleNamespace(
                    video=video_path,
                    object_tracks=tracks_path,
                    output_dir=output,
                    max_frames=10,
                )
            )
            with (output / "hard_negative_manifest.csv").open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_required"], "true")


if __name__ == "__main__":
    unittest.main()
