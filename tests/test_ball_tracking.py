import unittest

import numpy as np

from ball_to_player_assignment import BallToPlayerAssigner
from club_assignment import Club
from position_mappers import CalibrationResult, PitchGeometry
from tracking.ball_tracker import (
    BallCandidate,
    BallTracker,
    non_max_suppression,
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

    def test_nms_keeps_highest_confidence_duplicate(self) -> None:
        candidates = [
            BallCandidate((10, 10, 20, 20), 0.8),
            BallCandidate((11, 11, 21, 21), 0.4),
            BallCandidate((100, 100, 108, 108), 0.3),
        ]
        result = non_max_suppression(candidates, 0.3)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].confidence, 0.8)


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
        possession = assigner.get_ball_possessions()
        self.assertEqual(len(possession), 1)
        self.assertEqual(possession[-1][0], 1.0)


if __name__ == "__main__":
    unittest.main()
