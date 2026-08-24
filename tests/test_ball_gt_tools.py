import json
import tempfile
import unittest
from pathlib import Path

from tools.eval_ball import _frame_gt_report
from tools.interp_ball_gt import (
    _interpolate,
    _overlapping_segment_pairs,
    _read_annotations,
    _repair_overlapping_segments,
    _segment_intervals,
)
from tools.smoke_ball_fp import _report as _smoke_report
from tools.replay_ball_candidates import _gate_candidates
from tools.build_ball_finetune_dataset import (
    _dataset_status,
    _select_gt_rows,
    _yolo_box,
)
from tracking.ball_rescue import build_ball_rescue_rois, point_in_rescue_roi
from tracking import BallCandidate


class BallGroundTruthInterpolationTests(unittest.TestCase):
    def test_overlapping_segment_ids_are_merged_and_renumbered(self) -> None:
        rows = [
            {"frame": 0, "t": 0.0, "bbox": [0.0, 0.0, 8.0, 8.0], "seg": 1},
            {"frame": 5, "t": 0.1, "bbox": [5.0, 0.0, 13.0, 8.0], "seg": 0},
            {"frame": 10, "t": 0.2, "bbox": [10.0, 0.0, 18.0, 8.0], "seg": 1},
            {"frame": 100, "t": 3.3, "bbox": [20.0, 0.0, 28.0, 8.0], "seg": 3},
        ]
        intervals = _segment_intervals(rows)
        self.assertEqual(_overlapping_segment_pairs(intervals), [(0, 1)])
        normalized, mapping = _repair_overlapping_segments(rows)
        self.assertEqual(mapping, {1: 0, 0: 0, 3: 1})
        self.assertEqual([row["seg"] for row in normalized], [0, 0, 0, 1])

    def test_interpolation_does_not_cross_segment_or_large_gap(self) -> None:
        rows = [
            {"frame": 0, "t": 0.0, "bbox": [0.0, 0.0, 8.0, 8.0], "seg": 0},
            {"frame": 2, "t": 0.1, "bbox": [2.0, 0.0, 10.0, 8.0], "seg": 0},
            {"frame": 4, "t": 0.2, "bbox": [4.0, 0.0, 12.0, 8.0], "seg": 1},
            {"frame": 10, "t": 0.3, "bbox": [10.0, 0.0, 18.0, 8.0], "seg": 1},
        ]
        expanded = _interpolate(rows, max_gap_frames=3)
        self.assertEqual([row["frame"] for row in expanded], [0, 1, 2, 4, 10])
        self.assertTrue(next(row for row in expanded if row["frame"] == 1)["interp"])

    def test_reader_rejects_duplicate_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations.jsonl"
            row = {"frame": 1, "bbox": [0, 0, 8, 8], "seg": 0}
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate frame"):
                _read_annotations(path, 1920, 1080)


class BallGroundTruthEvaluationTests(unittest.TestCase):
    def test_center_metric_accepts_tight_detection_inside_click_box(self) -> None:
        gt = [
            {
                "frame": 10,
                "bbox": [100.0, 100.0, 116.0, 116.0],
                "seg": 0,
                "interp": False,
            }
        ]
        tracks = [
            {
                "frame": 10,
                "candidates": [[105.5, 105.5, 110.5, 110.5, 0.2]],
                "track": {
                    "bbox": [105.5, 105.5, 110.5, 110.5],
                    "observed": True,
                    "track_segment": 1,
                },
            }
        ]
        report = _frame_gt_report(gt, tracks, iou_threshold=0.3, details_limit=10)
        self.assertEqual(report["keyframes"]["iou"]["track_matched"], 0)
        self.assertEqual(report["keyframes"]["center"]["track_matched"], 1)
        self.assertEqual(report["keyframes"]["center"]["candidate_matched"], 1)

    def test_smoke_report_detects_removed_false_target(self) -> None:
        pilot = [{"id": "false-1", "frame": 3, "category": "confirmed"}]
        candidates = [{"id": "false-1", "bbox": [10.0, 10.0, 14.0, 14.0]}]
        removed = _smoke_report(
            pilot,
            candidates,
            [{"frame": 3, "track": {"bbox": [100.0, 100.0, 104.0, 104.0]}}],
            radius=24.0,
        )
        self.assertEqual(removed["survived_false_targets"], 0)
        self.assertEqual(removed["removed_false_targets"], 1)

    def test_smoke_report_separates_tentative_prediction_from_confirmed_fp(self) -> None:
        pilot = [{"id": "false-1", "frame": 3, "category": "confirmed"}]
        candidates = [{"id": "false-1", "bbox": [10.0, 10.0, 14.0, 14.0]}]
        report = _smoke_report(
            pilot,
            candidates,
            [
                {
                    "frame": 3,
                    "track": {
                        "bbox": [10.0, 10.0, 14.0, 14.0],
                        "observed": False,
                        "track_confirmed": False,
                    },
                }
            ],
            radius=24.0,
        )
        self.assertEqual(report["survived_false_targets"], 1)
        self.assertEqual(report["survived_observed_false_targets"], 0)
        self.assertEqual(report["survived_confirmed_false_targets"], 0)

    def test_player_distance_and_size_gate_filters_only_physical_candidates(self) -> None:
        objects = {
            "player": {
                1: {"bbox": [80, 40, 100, 104]},
                2: {"bbox": [180, 40, 200, 104]},
            },
            "goalkeeper": {},
        }
        near_ball = BallCandidate((92, 96, 100, 104), 0.2)
        far_speck = BallCandidate((500, 96, 508, 104), 0.9)
        oversized = BallCandidate((70, 80, 110, 120), 0.8)
        kept = _gate_candidates(
            [near_ball, far_speck, oversized],
            objects,
            distance_gate=1.1,
            size_factor_gate=2.1,
            minimum_players=2,
        )
        self.assertEqual(kept, [near_ball])

    def test_player_gate_disables_itself_when_too_few_players_exist(self) -> None:
        candidate = BallCandidate((500, 96, 508, 104), 0.9)
        objects = {"player": {1: {"bbox": [80, 40, 100, 104]}}}
        self.assertEqual(
            _gate_candidates(
                [candidate],
                objects,
                distance_gate=1.1,
                size_factor_gate=2.1,
                minimum_players=2,
            ),
            [candidate],
        )


class BallFinetuneDatasetTests(unittest.TestCase):
    def test_staging_selector_keeps_human_rows_and_strides_interpolation(self) -> None:
        rows = [
            {"frame": 0, "bbox": [0, 0, 8, 8], "interp": False},
            {"frame": 1, "bbox": [1, 0, 9, 8], "interp": True},
            {"frame": 5, "bbox": [5, 0, 13, 8], "interp": True},
        ]
        selected = _select_gt_rows(rows, True, 5)
        self.assertEqual([row["frame"] for row in selected], [0, 5])

    def test_staging_selector_rejects_long_interpolation_span(self) -> None:
        rows = [
            {"frame": 0, "bbox": [0, 0, 8, 8], "seg": 0, "interp": False},
            {"frame": 5, "bbox": [5, 0, 13, 8], "seg": 0, "interp": True},
            {"frame": 20, "bbox": [20, 0, 28, 8], "seg": 0, "interp": False},
        ]
        selected = _select_gt_rows(
            rows,
            include_interpolated=True,
            interpolated_stride=1,
            max_interpolation_span=15,
        )
        self.assertEqual([row["frame"] for row in selected], [0, 20])

    def test_yolo_box_is_clipped_and_normalized(self) -> None:
        self.assertEqual(_yolo_box([-10, 20, 50, 60], 100, 100), (0.25, 0.4, 0.5, 0.4))

    def test_single_source_staging_dataset_is_not_train_ready(self) -> None:
        rows = [
            {
                "sample_type": "human_keyframe",
                "source": "demo4",
                "split": "staging",
            }
            for _ in range(425)
        ]
        status = _dataset_status(rows)
        self.assertFalse(status["train_ready"])
        self.assertIn("positive frames 425 < 600", status["blockers"])


class BallRescueRoiTests(unittest.TestCase):
    def test_player_foot_roi_is_clipped_and_covers_nearby_ball(self) -> None:
        objects = {
            "player": {1: {"bbox": [80, 40, 100, 104], "confidence": 0.9}},
            "goalkeeper": {},
        }
        rois = build_ball_rescue_rois(objects, (120, 160, 3), maximum_rois=2)
        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].source, "player_foot")
        self.assertGreaterEqual(rois[0].bbox[0], 0)
        self.assertLessEqual(rois[0].bbox[2], 160)
        self.assertTrue(point_in_rescue_roi((104, 104), rois[0]))

    def test_prediction_roi_works_without_players(self) -> None:
        rois = build_ball_rescue_rois(
            {}, (1080, 1920, 3), predicted_center=(500, 300), maximum_rois=1
        )
        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].source, "prediction")
        self.assertTrue(point_in_rescue_roi((500, 300), rois[0]))

    def test_prediction_prioritizes_near_player_when_roi_budget_is_small(self) -> None:
        objects = {
            "player": {
                1: {"bbox": [80, 40, 100, 104], "confidence": 0.5},
                2: {"bbox": [880, 40, 900, 104], "confidence": 0.9},
            },
            "goalkeeper": {},
        }
        rois = build_ball_rescue_rois(
            objects,
            (1080, 1920, 3),
            predicted_center=(100, 100),
            maximum_rois=2,
            dedup_iou=1.0,
        )
        self.assertEqual(rois[0].source, "prediction")
        self.assertEqual(rois[1].player_id, 1)


if __name__ == "__main__":
    unittest.main()
