import json
import tempfile
import unittest
from pathlib import Path

from evaluation import evaluate_run


class EvaluationTests(unittest.TestCase):
    def test_ground_truth_run_can_pass_promotion_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration_tracks.jsonl"
            objects = root / "object_tracks.jsonl"
            truth = root / "ground_truth.jsonl"
            with calibration.open("w", encoding="utf-8") as cal_handle, objects.open(
                "w", encoding="utf-8"
            ) as object_handle, truth.open("w", encoding="utf-8") as truth_handle:
                for _ in range(30):
                    cal_handle.write(
                        json.dumps(
                            {
                                "image_to_pitch": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                "age_seconds": 0.0,
                                "quality": 0.9,
                                "median_error_px": 2.0,
                            }
                        )
                        + "\n"
                    )
                    object_handle.write(
                        json.dumps(
                            {
                                "ball": {
                                    "1": {
                                        "bbox": [10, 10, 20, 20],
                                        "observed": True,
                                        "track_confirmed": True,
                                        "track_segment": 1,
                                        "speed_state": "reliable",
                                        "speed": 20.0,
                                        "motion_mode": "ground",
                                    }
                                }
                            }
                        )
                        + "\n"
                    )
                    truth_handle.write(
                        json.dumps(
                            {
                                "ball_visible": True,
                                "bbox": [10, 10, 20, 20],
                                "speed_supported": True,
                                "speed_3d_kmh": 21.0,
                                "motion_mode": "ground",
                            }
                        )
                        + "\n"
                    )
            report = evaluate_run(root, ground_truth_path=truth, fps=30.0)
        self.assertTrue(report["promotion_eligible"])
        self.assertEqual(report["ball_precision"], 1.0)
        self.assertEqual(report["visible_ball_recall"], 1.0)
        self.assertEqual(report["ground_speed_median_absolute_error_kmh"], 1.0)
        self.assertAlmostEqual(report["ground_speed_median_relative_error"], 1.0 / 21.0)


if __name__ == "__main__":
    unittest.main()
