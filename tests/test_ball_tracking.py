import unittest
import tempfile
from pathlib import Path

import numpy as np
import yaml

from ball_to_player_assignment import BallToPlayerAssigner
from club_assignment import Club
from position_mappers import CalibrationResult, PitchGeometry
from tracking.ball_tracker import (
    BallCandidate,
    BallDetector,
    BallTracker,
    ConstantVelocityKalman,
    validate_ball_model_for_promotion,
    validate_ball_verifier_for_promotion,
    non_max_suppression,
    offset_bbox,
    overlapping_tiles,
)


class BallDetectionUtilitiesTests(unittest.TestCase):
    def test_overlapping_tiles_cover_frame_edges(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        tiles = overlapping_tiles(frame)
        self.assertEqual(len(tiles), 4)
        covered_right = max(x + tile.shape[1] for tile, x, _ in tiles)
        covered_bottom = max(y + tile.shape[0] for tile, _, y in tiles)
        self.assertEqual(covered_right, 1920)
        self.assertEqual(covered_bottom, 1080)

    def test_tile_bbox_is_mapped_back_to_source_coordinates(self) -> None:
        self.assertEqual(offset_bbox((10, 20, 14, 25), 900, 400), (910, 420, 914, 425))

    def test_nms_keeps_highest_confidence_duplicate(self) -> None:
        candidates = [
            BallCandidate((10, 10, 20, 20), 0.8),
            BallCandidate((11, 11, 21, 21), 0.4),
            BallCandidate((100, 100, 108, 108), 0.3),
        ]
        result = non_max_suppression(candidates, 0.3)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].confidence, 0.8)

    def test_kalman_prediction_uses_variable_timestamps(self) -> None:
        tracker = ConstantVelocityKalman(1.0, 0.01)
        tracker.correct(np.asarray([0.0, 0.0]), 0.0)
        tracker.correct(np.asarray([1.0, 0.0]), 0.1)
        predicted = tracker.predict(0.2)
        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(float(predicted[0]), 2.0, delta=0.15)

    def test_dynamic_openvino_export_keeps_requested_inference_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "metadata.yaml").write_text(
                yaml.safe_dump({"imgsz": [1280, 1280], "args": {"dynamic": True}}),
                encoding="utf-8",
            )
            self.assertIsNone(BallDetector._fixed_export_size(model_dir))

    def test_static_small_openvino_export_is_blocked_from_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "metadata.yaml").write_text(
                yaml.safe_dump(
                    {
                        "description": "Ultralytics YOLO11n model trained for balls",
                        "imgsz": [640, 640],
                        "args": {"dynamic": False},
                        "names": {0: "ball"},
                    }
                ),
                encoding="utf-8",
            )
            errors = validate_ball_model_for_promotion(model_dir)
        self.assertTrue(any("dynamic=True" in error for error in errors))
        self.assertTrue(any("at least 1280" in error for error in errors))
        self.assertTrue(any("YOLO11s" in error for error in errors))

    def test_checkpoint_without_export_metadata_is_not_formally_promoted(self) -> None:
        errors = validate_ball_model_for_promotion("ball-detection.pt")
        self.assertTrue(any("metadata.yaml" in error for error in errors))

    def test_verifier_promotion_requires_validated_threshold_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            verifier_path = Path(temporary) / "verifier.pt"
            errors = validate_ball_verifier_for_promotion(verifier_path)
            self.assertTrue(any("sidecar is missing" in error for error in errors))
            verifier_path.with_suffix(".verifier.json").write_text(
                '{"decision_threshold": 0.62, "precision": 0.97, "recall": 0.76}',
                encoding="utf-8",
            )
            self.assertEqual(
                validate_ball_verifier_for_promotion(verifier_path),
                [],
            )


class BallTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = PitchGeometry(105, 68)
        self.calibration = CalibrationResult(
            image_to_pitch=np.eye(3, dtype=np.float32),
            status="detected",
            age_seconds=0.0,
        )

    def test_observation_and_short_prediction_are_distinguished(self) -> None:
        tracker = BallTracker(self.geometry)
        observed = tracker.update(
            [BallCandidate((8, 8, 12, 12), 0.8)],
            0.0,
            self.calibration,
            {"player": {}, "goalkeeper": {}},
            (68, 105, 3),
        )
        self.assertTrue(observed[1]["observed"])
        predicted = tracker.update(
            [],
            0.2,
            self.calibration,
            {"player": {}, "goalkeeper": {}},
            (68, 105, 3),
        )
        self.assertFalse(predicted[1]["observed"])
        expired = tracker.update(
            [],
            0.6,
            self.calibration,
            {"player": {}, "goalkeeper": {}},
            (68, 105, 3),
        )
        self.assertEqual(expired, {})

    def test_reset_discards_short_term_prediction(self) -> None:
        tracker = BallTracker(self.geometry)
        tracker.update(
            [BallCandidate((8, 8, 12, 12), 0.8)],
            0.0,
            self.calibration,
            {"player": {}, "goalkeeper": {}},
            (68, 105, 3),
        )
        tracker.reset()
        self.assertEqual(
            tracker.update(
                [],
                0.1,
                self.calibration,
                {"player": {}, "goalkeeper": {}},
                (68, 105, 3),
            ),
            {},
        )

    def test_candidate_outside_metric_pitch_is_rejected(self) -> None:
        tracker = BallTracker(self.geometry)
        result = tracker.update(
            [BallCandidate((150, 80, 155, 85), 0.9)],
            0.0,
            self.calibration,
            {"player": {}, "goalkeeper": {}},
            (108, 192, 3),
        )
        self.assertEqual(result, {})

    def test_large_candidate_jump_starts_unconfirmed_segment(self) -> None:
        tracker = BallTracker(self.geometry)
        players = {"player": {}, "goalkeeper": {}}
        first = tracker.update(
            [BallCandidate((8, 8, 12, 12), 0.8)],
            0.0,
            self.calibration,
            players,
            (1080, 1920, 3),
        )
        self.assertEqual(first[1]["track_segment"], 1)
        jumped = tracker.update(
            [BallCandidate((48, 8, 52, 12), 0.95)],
            1 / 30.0,
            self.calibration,
            players,
            (1080, 1920, 3),
        )
        self.assertEqual(jumped[1]["track_segment"], 2)
        self.assertFalse(jumped[1]["track_confirmed"])

    def test_segment_requires_three_continuous_observations(self) -> None:
        tracker = BallTracker(self.geometry)
        players = {"player": {}, "goalkeeper": {}}
        latest = None
        for frame, center_x in enumerate((10.0, 10.2, 10.4)):
            latest = tracker.update(
                [BallCandidate((center_x - 2, 8, center_x + 2, 12), 0.8)],
                frame / 30.0,
                self.calibration,
                players,
                (1080, 1920, 3),
            )
            if frame < 2:
                self.assertFalse(latest[1]["track_confirmed"])
        self.assertIsNotNone(latest)
        self.assertTrue(latest[1]["track_confirmed"])
        self.assertEqual(latest[1]["track_segment"], 1)

    def test_fixed_lag_requires_temporal_evidence_before_confirmation(self) -> None:
        tracker = BallTracker(self.geometry, fixed_lag_seconds=0.20)
        players = {"player": {}, "goalkeeper": {}}
        latest = None
        for frame in range(7):
            latest = tracker.update(
                [BallCandidate((8 + frame * 0.1, 8, 12 + frame * 0.1, 12), 0.8)],
                frame / 30.0,
                self.calibration,
                players,
                (1080, 1920, 3),
            )
            if frame < 6:
                self.assertFalse(latest[1]["track_confirmed"])
        self.assertTrue(latest[1]["track_confirmed"])

    def test_observation_after_long_gap_starts_new_segment(self) -> None:
        tracker = BallTracker(self.geometry)
        players = {"player": {}, "goalkeeper": {}}
        first = tracker.update(
            [BallCandidate((8, 8, 12, 12), 0.8)],
            0.0,
            self.calibration,
            players,
            (1080, 1920, 3),
        )
        restarted = tracker.update(
            [BallCandidate((8, 8, 12, 12), 0.8)],
            0.2,
            self.calibration,
            players,
            (1080, 1920, 3),
        )
        self.assertNotEqual(
            first[1]["track_segment"], restarted[1]["track_segment"]
        )
        self.assertFalse(restarted[1]["track_confirmed"])

    def test_uncalibrated_large_pixel_jump_starts_new_segment(self) -> None:
        tracker = BallTracker(self.geometry)
        invalid = CalibrationResult(image_to_pitch=None, status="invalid")
        players = {"player": {}, "goalkeeper": {}}
        first = tracker.update(
            [BallCandidate((98, 98, 102, 102), 0.8)],
            0.0,
            invalid,
            players,
            (1080, 1920, 3),
        )
        jumped = tracker.update(
            [BallCandidate((398, 98, 402, 102), 0.95)],
            1 / 30.0,
            invalid,
            players,
            (1080, 1920, 3),
        )
        self.assertNotEqual(first[1]["track_segment"], jumped[1]["track_segment"])

    def test_low_confidence_pitch_marking_never_confirms(self) -> None:
        tracker = BallTracker(self.geometry)
        players = {"player": {}, "goalkeeper": {}}
        latest = None
        for frame in range(10):
            latest = tracker.update(
                [
                    BallCandidate(
                        (50.5, 32.0, 54.5, 36.0),
                        0.20,
                        line_score=1.0,
                    )
                ],
                frame / 30.0,
                self.calibration,
                players,
                (1080, 1920, 3),
            )
        self.assertIsNotNone(latest)
        self.assertFalse(latest[1]["track_confirmed"])
        self.assertIn("pitch_marking_overlap", latest[1]["rejection_reasons"])

    def test_strong_ball_on_pitch_marking_can_confirm(self) -> None:
        tracker = BallTracker(self.geometry)
        players = {"player": {}, "goalkeeper": {}}
        latest = None
        for frame in range(4):
            latest = tracker.update(
                [
                    BallCandidate(
                        (50.5, 32.0, 54.5, 36.0),
                        0.85,
                        appearance_score=0.95,
                        line_score=1.0,
                        verifier_score=0.95,
                        verifier_threshold=0.60,
                    )
                ],
                frame / 30.0,
                self.calibration,
                players,
                (1080, 1920, 3),
            )
        self.assertIsNotNone(latest)
        self.assertTrue(latest[1]["track_confirmed"])

    def test_verifier_rejection_blocks_confirmation(self) -> None:
        tracker = BallTracker(self.geometry)
        latest = None
        for frame in range(6):
            latest = tracker.update(
                [
                    BallCandidate(
                        (18, 18, 22, 22),
                        0.95,
                        appearance_score=0.10,
                        verifier_score=0.10,
                        verifier_threshold=0.60,
                    )
                ],
                frame / 30.0,
                self.calibration,
                {"player": {}, "goalkeeper": {}},
                (1080, 1920, 3),
            )
        self.assertFalse(latest[1]["track_confirmed"])
        self.assertIn("appearance_rejected", latest[1]["rejection_reasons"])

    def test_close_candidate_scores_remain_ambiguous(self) -> None:
        tracker = BallTracker(self.geometry, ambiguity_margin=0.10)
        result = tracker.update(
            [
                BallCandidate((8, 8, 12, 12), 0.80),
                BallCandidate((13, 8, 17, 12), 0.79),
            ],
            0.0,
            self.calibration,
            {"player": {}, "goalkeeper": {}},
            (1080, 1920, 3),
        )
        self.assertEqual(result[1]["track_state"], "ambiguous")
        self.assertFalse(result[1]["track_confirmed"])
        self.assertEqual(result[1]["hypothesis_count"], 2)


class MetricPossessionTests(unittest.TestCase):
    def test_ball_within_two_metres_is_assigned_once(self) -> None:
        red = Club("Red", (255, 0, 0), (0, 0, 0))
        blue = Club("Blue", (0, 0, 255), (255, 255, 0))
        assigner = BallToPlayerAssigner(red, blue)
        tracks = {
            "ball": {
                1: {
                    "bbox": [0, 0, 2, 2],
                    "position_m": (11.0, 10.0),
                    "observed": True,
                    "track_confidence": 0.8,
                }
            },
            "player": {
                7: {
                    "bbox": [0, 0, 10, 20],
                    "position_m": (12.5, 10.0),
                    "club": "Red",
                }
            },
            "goalkeeper": {},
            "referee": {},
        }
        result, player_id = assigner.assign(tracks, 0, timestamp_seconds=0.0)
        self.assertEqual(player_id, 7)
        self.assertTrue(result["player"][7]["has_ball"])
        self.assertEqual(assigner.get_current_possession(), "Red")
        possession = assigner.get_ball_possessions()
        self.assertEqual(len(possession), 1)
        self.assertEqual(possession[-1][0], 1.0)

    def test_low_confidence_prediction_does_not_assign_possession(self) -> None:
        red = Club("Red", (255, 0, 0), (0, 0, 0))
        blue = Club("Blue", (0, 0, 255), (255, 255, 0))
        assigner = BallToPlayerAssigner(red, blue)
        tracks = {
            "ball": {
                1: {
                    "bbox": [0, 0, 2, 2],
                    "position_m": (11.0, 10.0),
                    "observed": False,
                    "track_confidence": 0.10,
                }
            },
            "player": {
                7: {
                    "bbox": [0, 0, 10, 20],
                    "position_m": (11.5, 10.0),
                    "club": "Red",
                }
            },
            "goalkeeper": {},
            "referee": {},
        }
        result, player_id = assigner.assign(tracks, 0, timestamp_seconds=0.0)
        self.assertEqual(player_id, -1)
        self.assertNotIn("has_ball", result["player"][7])
        self.assertEqual(assigner.get_current_possession(), -1)

    def test_tentative_ball_does_not_assign_possession(self) -> None:
        red = Club("Red", (255, 0, 0), (0, 0, 0))
        blue = Club("Blue", (0, 0, 255), (255, 255, 0))
        assigner = BallToPlayerAssigner(red, blue)
        tracks = {
            "ball": {
                1: {
                    "bbox": [0, 0, 2, 2],
                    "position_m": (11.0, 10.0),
                    "observed": True,
                    "track_confidence": 0.8,
                    "track_confirmed": False,
                }
            },
            "player": {
                7: {
                    "bbox": [0, 0, 10, 20],
                    "position_m": (11.5, 10.0),
                    "club": "Red",
                }
            },
            "goalkeeper": {},
            "referee": {},
        }
        result, player_id = assigner.assign(tracks, 0, timestamp_seconds=0.0)
        self.assertEqual(player_id, -1)
        self.assertNotIn("has_ball", result["player"][7])


if __name__ == "__main__":
    unittest.main()
