import shutil
import tempfile
import unittest
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.prepare_dataset import PITCH_FLIP_INDEX, validate_dataset


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


if __name__ == "__main__":
    unittest.main()
