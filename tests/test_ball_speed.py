import unittest

import numpy as np

from position_mappers import CalibrationResult
from speed_estimation import BallSpeedEstimator


def calibration(
    matrix: np.ndarray | None = None,
    quality: float = 0.8,
    status: str = "detected",
) -> CalibrationResult:
    return CalibrationResult(
        image_to_pitch=np.eye(3, dtype=np.float32) if matrix is None else matrix,
        status=status,
        quality=quality,
        age_seconds=0.0,
    )


def ball_track(
    timestamp: float,
    segment: int = 1,
    observed: bool = True,
    confirmed: bool = True,
    position: tuple[float, float] | None = None,
) -> dict:
    position = position or (10.0 * timestamp, 2.0)
    return {
        "ball": {
            1: {
                "bbox": [position[0] - 2, position[1] - 2, position[0] + 2, position[1] + 2],
                "position_m": position,
                "confidence": 0.6,
                "track_confidence": 0.6,
                "observed": observed,
                "track_confirmed": confirmed,
                "track_segment": segment,
            }
        }
    }


class BallSpeedEstimatorTests(unittest.TestCase):
    def test_stable_observed_segment_recovers_speed(self) -> None:
        estimator = BallSpeedEstimator()
        latest = None
        for frame in range(8):
            timestamp = frame / 30.0
            latest = estimator.calculate_speed(ball_track(timestamp), timestamp, calibration())
        self.assertIsNotNone(latest)
        ball = latest["ball"][1]
        self.assertEqual(ball["speed_status"], "reliable")
        self.assertAlmostEqual(ball["speed"], 36.0, delta=0.2)

    def test_speed_is_not_capped_after_track_passes_quality_gates(self) -> None:
        estimator = BallSpeedEstimator()
        latest = None
        for frame in range(8):
            timestamp = frame / 30.0
            latest = estimator.calculate_speed(
                ball_track(timestamp, position=(50.0 * timestamp, 2.0)),
                timestamp,
                calibration(),
            )
        self.assertIsNotNone(latest)
        self.assertEqual(latest["ball"][1]["speed_status"], "reliable")
        self.assertAlmostEqual(latest["ball"][1]["speed"], 180.0, delta=0.5)

    def test_low_detection_confidence_remains_pending(self) -> None:
        estimator = BallSpeedEstimator()
        latest = None
        for frame in range(8):
            timestamp = frame / 30.0
            tracks = ball_track(timestamp)
            tracks["ball"][1]["confidence"] = 0.10
            latest = estimator.calculate_speed(tracks, timestamp, calibration())
        self.assertIsNotNone(latest)
        self.assertEqual(latest["ball"][1]["speed_status"], "pending")
        self.assertNotIn("speed", latest["ball"][1])

    def test_prediction_never_produces_ball_speed(self) -> None:
        estimator = BallSpeedEstimator()
        for frame in range(12):
            timestamp = frame / 30.0
            estimator.calculate_speed(ball_track(timestamp), timestamp, calibration())
        predicted = estimator.calculate_speed(
            ball_track(0.4, observed=False), 0.4, calibration()
        )
        self.assertEqual(predicted["ball"][1]["speed_status"], "pending")
        self.assertNotIn("speed", predicted["ball"][1])

    def test_tentative_observations_seed_history_after_confirmation(self) -> None:
        estimator = BallSpeedEstimator()
        latest = None
        for frame in range(8):
            timestamp = frame / 30.0
            latest = estimator.calculate_speed(
                ball_track(timestamp, confirmed=frame >= 2),
                timestamp,
                calibration(),
            )
            if frame < 7:
                self.assertEqual(latest["ball"][1]["speed_status"], "pending")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["ball"][1]["speed_status"], "reliable")
        self.assertAlmostEqual(latest["ball"][1]["speed"], 36.0, delta=0.2)

    def test_new_segment_does_not_inherit_old_speed(self) -> None:
        estimator = BallSpeedEstimator()
        for frame in range(12):
            timestamp = frame / 30.0
            estimator.calculate_speed(ball_track(timestamp), timestamp, calibration())
        restarted = estimator.calculate_speed(
            ball_track(0.4, segment=2, position=(80.0, 30.0)),
            0.4,
            calibration(),
        )
        self.assertEqual(restarted["ball"][1]["speed_status"], "pending")
        self.assertNotIn("speed", restarted["ball"][1])
        self.assertEqual(len(estimator.history), 1)

    def test_calibration_shift_clears_speed_history(self) -> None:
        estimator = BallSpeedEstimator()
        for frame in range(12):
            timestamp = frame / 30.0
            estimator.calculate_speed(ball_track(timestamp), timestamp, calibration())
        shifted = np.asarray([[1, 0, 3], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = estimator.calculate_speed(ball_track(0.4), 0.4, calibration(shifted))
        self.assertEqual(result["ball"][1]["speed_status"], "pending")
        self.assertNotIn("speed", result["ball"][1])
        self.assertEqual(len(estimator.history), 1)

    def test_propagated_camera_motion_does_not_look_like_calibration_switch(self) -> None:
        estimator = BallSpeedEstimator()
        for frame in range(7):
            timestamp = frame / 30.0
            matrix = np.asarray(
                [[1, 0, frame * 0.4], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            )
            estimator.calculate_speed(
                ball_track(timestamp),
                timestamp,
                calibration(matrix, status="propagated"),
            )
        timestamp = 7 / 30.0
        matrix = np.asarray([[1, 0, 2.8], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = estimator.calculate_speed(
            ball_track(timestamp),
            timestamp,
            calibration(matrix, status="propagated"),
        )
        self.assertEqual(result["ball"][1]["speed_status"], "reliable")
        self.assertAlmostEqual(result["ball"][1]["speed"], 36.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
