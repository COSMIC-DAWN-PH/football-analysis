import unittest

import cv2
import numpy as np

from position_mappers import DynamicCameraCalibrator, PitchGeometry
from speed_estimation import SpeedEstimator


class DynamicCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = PitchGeometry(105.0, 68.0)
        self.world_to_image = np.asarray(
            [[12.0, 1.2, 220.0], [0.4, 9.0, 170.0], [0.001, 0.0015, 1.0]],
            dtype=np.float32,
        )

    def _image_points(self) -> dict[int, tuple[float, float]]:
        points = cv2.perspectiveTransform(
            self.geometry.vertices.reshape(-1, 1, 2), self.world_to_image
        ).reshape(-1, 2)
        return {index: tuple(map(float, point)) for index, point in enumerate(points)}

    def test_pitch_geometry_preserves_semantic_landmarks(self) -> None:
        vertices = self.geometry.vertices
        self.assertEqual(vertices.shape, (32, 2))
        np.testing.assert_allclose(vertices[8], [11.0, 34.0])
        np.testing.assert_allclose(vertices[21], [94.0, 34.0])
        np.testing.assert_allclose(vertices[30], [43.35, 34.0])

    def test_ransac_rejects_outlier_and_recovers_metric_projection(self) -> None:
        keypoints = self._image_points()
        keypoints[10] = (1800.0, 900.0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        calibrator = DynamicCameraCalibrator(self.geometry)
        result = calibrator.update(
            frame, keypoints, {index: 0.9 for index in keypoints}, 0.0
        )
        self.assertTrue(result.valid)
        self.assertGreaterEqual(result.inliers, 30)
        sample_world = np.asarray([[[52.5, 34.0]]], dtype=np.float32)
        sample_image = cv2.perspectiveTransform(sample_world, self.world_to_image)
        recovered = cv2.perspectiveTransform(sample_image, result.image_to_pitch)
        np.testing.assert_allclose(recovered, sample_world, atol=0.15)

    def test_flow_propagation_expires_after_one_second(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for y in range(180, 850, 30):
            for x in range(230, 1450, 30):
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
        calibrator = DynamicCameraCalibrator(self.geometry)
        first = calibrator.update(
            frame,
            self._image_points(),
            {index: 0.9 for index in range(32)},
            0.0,
        )
        self.assertTrue(first.valid)
        shifted = cv2.warpAffine(
            frame, np.asarray([[1, 0, 4], [0, 1, 2]], dtype=np.float32), (1920, 1080)
        )
        propagated = calibrator.update(shifted, {}, {}, 0.1)
        self.assertEqual(propagated.status, "propagated")
        self.assertGreater(propagated.flow_quality, 0.0)
        self.assertLess(propagated.quality, first.quality)
        expired = calibrator.update(shifted, {}, {}, 1.2)
        self.assertFalse(expired.valid)

    def test_camera_cut_marks_calibration_for_dependent_state_reset(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:] = (0, 180, 0)
        calibrator = DynamicCameraCalibrator(self.geometry)
        first = calibrator.update(
            frame,
            self._image_points(),
            {index: 0.9 for index in range(32)},
            0.0,
        )
        self.assertTrue(first.valid)
        cut_frame = np.zeros_like(frame)
        cut_frame[:] = (180, 0, 0)
        after_cut = calibrator.update(
            cut_frame,
            self._image_points(),
            {index: 0.9 for index in range(32)},
            0.04,
        )
        self.assertTrue(after_cut.valid)
        self.assertTrue(after_cut.reset_required)
        self.assertEqual(after_cut.status, "detected_reset")


class SpeedEstimatorTests(unittest.TestCase):
    def test_speed_uses_metric_positions_and_timestamps(self) -> None:
        estimator = SpeedEstimator()
        latest = None
        for frame in range(19):
            timestamp = frame / 30.0
            tracks = {
                "player": {7: {"position_m": (2.0 * timestamp, 4.0)}},
                "goalkeeper": {},
            }
            latest = estimator.calculate_speed(tracks, timestamp)
        self.assertIsNotNone(latest)
        self.assertAlmostEqual(latest["player"][7]["speed"], 7.2, delta=0.15)

    def test_invalid_jump_is_not_clamped_into_output(self) -> None:
        estimator = SpeedEstimator()
        for frame in range(16):
            timestamp = frame / 30.0
            tracks = {
                "player": {1: {"position_m": (timestamp, 0.0)}},
                "goalkeeper": {},
            }
            estimator.calculate_speed(tracks, timestamp)
        bad = {"player": {1: {"position_m": (100.0, 0.0)}}, "goalkeeper": {}}
        result = estimator.calculate_speed(bad, 16 / 30.0)
        self.assertNotIn("speed", result["player"][1])


if __name__ == "__main__":
    unittest.main()
