import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import video_annotate_server as server


class VideoAnnotationServerTests(unittest.TestCase):
    def setUp(self) -> None:
        server.app.config.update(TESTING=True)
        self.client = server.app.test_client()

    def test_annotation_validation_rejects_duplicate_and_out_of_bounds_rows(self) -> None:
        valid = server._validate_annotations(
            [{"frame": 3, "bbox": [10, 20, 18, 28], "seg": 2}]
        )
        self.assertEqual(valid[3]["t"], round(3 / server.FPS, 4))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            server._validate_annotations(
                [
                    {"frame": 3, "bbox": [10, 20, 18, 28], "seg": 2},
                    {"frame": 3, "bbox": [11, 20, 19, 28], "seg": 2},
                ]
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            server._validate_annotations(
                [{"frame": 4, "bbox": [-1, 20, 18, 28], "seg": 2}]
            )

    def test_save_is_atomic_and_preserves_previous_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gt_dir = Path(temporary)
            source_dir = gt_dir / "demo4"
            source_dir.mkdir(parents=True)
            annotations = source_dir / "annotations.jsonl"
            original = {"frame": 1, "t": 0.0334, "bbox": [1, 1, 9, 9], "seg": 0}
            annotations.write_text(json.dumps(original) + "\n", encoding="utf-8")
            with patch.object(server, "GT_DIR", gt_dir):
                response = self.client.post(
                    "/api/gt/save/demo4",
                    json={
                        "annotations": [
                            {"frame": 2, "bbox": [2, 2, 10, 10], "seg": 0}
                        ]
                    },
                )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["changed"])
            saved = json.loads(annotations.read_text(encoding="utf-8"))
            backup = json.loads(
                (source_dir / "annotations.jsonl.bak").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["frame"], 2)
            self.assertEqual(backup, original)
            self.assertFalse((source_dir / "annotations.jsonl.tmp").exists())

    def test_review_progress_is_monotonic_and_snapshot_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gt_dir = Path(temporary)
            source_dir = gt_dir / "demo4"
            source_dir.mkdir(parents=True)
            (source_dir / "annotations.jsonl").write_text(
                json.dumps(
                    {"frame": 1, "t": 0.0334, "bbox": [1, 1, 9, 9], "seg": 0}
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(server, "GT_DIR", gt_dir):
                first = self.client.post(
                    "/api/gt/progress/demo4", json={"reviewed_through_frame": 100}
                )
                second = self.client.post(
                    "/api/gt/progress/demo4", json={"reviewed_through_frame": 80}
                )
                snapshot = self.client.post("/api/gt/snapshot/demo4")
                loaded = self.client.get("/api/gt/demo4")
            self.assertEqual(first.get_json()["reviewed_through_frame"], 100)
            self.assertEqual(second.get_json()["reviewed_through_frame"], 100)
            self.assertEqual(loaded.get_json()["reviewed_through_frame"], 100)
            self.assertEqual(snapshot.status_code, 200)
            snapshot_path = Path(snapshot.get_json()["path"])
            if not snapshot_path.is_absolute():
                snapshot_path = server.ROOT / snapshot_path
            self.assertEqual(
                snapshot_path.read_text(encoding="utf-8"),
                (source_dir / "annotations.jsonl").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
