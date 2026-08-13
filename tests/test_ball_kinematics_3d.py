import unittest

import cv2
import numpy as np

from position_mappers import CameraGeometry, CameraProfile, PitchGeometry
from speed_estimation import BallKinematics3D


class BallKinematics3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pitch = PitchGeometry(105, 68)
        matrix = np.asarray([[1200.0, 0, 960], [0, 1200.0, 540], [0, 0, 1]])
        self.profile = CameraProfile(matrix, np.zeros(5), 1920, 1080, "test")
        camera = np.asarray([52.5, -35.0, 22.0])
        target = np.asarray([52.5, 34.0, 0.0])
        forward = target - camera
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.rotation = np.vstack([right, down, forward])
        translation = -self.rotation @ camera
        rotation_vector = cv2.Rodrigues(self.rotation)[0]
        world = np.column_stack([self.pitch.vertices, np.zeros(len(self.pitch.vertices))])
        image = cv2.projectPoints(
            world, rotation_vector, translation, matrix, np.zeros(5)
        )[0].reshape(-1, 2)
        self.geometry = CameraGeometry(self.pitch, self.profile)
        self.pose = self.geometry.pose_from_correspondences(
            image, self.pitch.vertices, status="anchored"
        )

    def _tracks(self, point: np.ndarray) -> dict:
        pixel = self.geometry.project_world(point, self.pose)
        rotation = cv2.Rodrigues(self.pose.rotation_vector)[0]
        depth = (rotation @ point + self.pose.translation_vector)[2]
        size = self.profile.camera_matrix[0, 0] * 0.22 / depth
        return {
            "ball": {
                1: {
                    "bbox": [
                        pixel[0] - size / 2,
                        pixel[1] - size / 2,
                        pixel[0] + size / 2,
                        pixel[1] + size / 2,
                    ],
                    "confidence": 0.9,
                    "track_confidence": 0.9,
                    "observed": True,
                    "track_confirmed": True,
                    "track_segment": 1,
                }
            }
        }

    def test_ground_motion_recovers_metric_speed(self) -> None:
        estimator = BallKinematics3D(self.pitch, self.profile)
        latest = None
        for frame in range(20):
            timestamp = frame / 30.0
            latest = estimator.calculate_speed(
                self._tracks(np.asarray([20 + 5 * timestamp, 30.0, 0.11])),
                timestamp,
                self.pose,
            )
        ball = latest["ball"][1]
        self.assertEqual(ball["speed_state"], "reliable")
        self.assertEqual(ball["motion_mode"], "ground")
        self.assertAlmostEqual(ball["speed"], 18.0, delta=0.2)

    def test_ballistic_motion_selects_air_model(self) -> None:
        estimator = BallKinematics3D(self.pitch, self.profile)
        latest = None
        for frame in range(20):
            timestamp = frame / 30.0
            point = np.asarray(
                [
                    20 + 12 * timestamp,
                    30.0,
                    0.5 + 8 * timestamp - 0.5 * 9.81 * timestamp**2,
                ]
            )
            latest = estimator.calculate_speed(
                self._tracks(point), timestamp, self.pose
            )
        ball = latest["ball"][1]
        self.assertEqual(ball["speed_state"], "reliable")
        self.assertEqual(ball["motion_mode"], "air")
        self.assertLessEqual(ball["speed_uncertainty_kmh"], 5.0)
        self.assertEqual(ball["speed_model"], "monocular_physics_estimate")

    def test_low_confidence_never_outputs_speed(self) -> None:
        estimator = BallKinematics3D(self.pitch, self.profile)
        latest = None
        for frame in range(20):
            timestamp = frame / 30.0
            tracks = self._tracks(np.asarray([20 + timestamp, 30.0, 0.11]))
            tracks["ball"][1]["confidence"] = 0.05
            latest = estimator.calculate_speed(tracks, timestamp, self.pose)
        ball = latest["ball"][1]
        self.assertEqual(ball["speed_state"], "unavailable")
        self.assertEqual(ball["speed_reason"], "low_ball_confidence")
        self.assertNotIn("speed", ball)

    def test_ground_contact_direction_change_starts_new_motion_segment(self) -> None:
        estimator = BallKinematics3D(self.pitch, self.profile)
        latest = None
        positions = [20.0, 20.2, 20.4, 20.0]
        for frame, x_position in enumerate(positions):
            latest = estimator.calculate_speed(
                self._tracks(np.asarray([x_position, 30.0, 0.11])),
                frame / 30.0,
                self.pose,
            )
        ball = latest["ball"][1]
        self.assertEqual(ball["motion_segment"], 2)
        self.assertEqual(ball["speed_state"], "warming_up")


if __name__ == "__main__":
    unittest.main()
