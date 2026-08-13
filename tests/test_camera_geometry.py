import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from position_mappers import (
    CameraGeometry,
    CameraProfile,
    PitchAnchorSet,
    PitchGeometry,
)


def synthetic_camera():
    pitch = PitchGeometry(105, 68)
    matrix = np.asarray([[1200.0, 0, 960], [0, 1200.0, 540], [0, 0, 1]])
    profile = CameraProfile(matrix, np.zeros(5), 1920, 1080, "synthetic")
    camera = np.asarray([52.5, -35.0, 22.0])
    target = np.asarray([52.5, 34.0, 0.0])
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ camera
    rotation_vector = cv2.Rodrigues(rotation)[0]
    world = np.column_stack([pitch.vertices, np.zeros(len(pitch.vertices))])
    image = cv2.projectPoints(
        world, rotation_vector, translation, matrix, np.zeros(5)
    )[0].reshape(-1, 2)
    geometry = CameraGeometry(pitch, profile)
    pose = geometry.pose_from_correspondences(
        image, pitch.vertices, status="anchored"
    )
    return pitch, profile, geometry, pose


class CameraGeometryTests(unittest.TestCase):
    def test_profile_and_anchor_round_trip(self) -> None:
        matrix = np.asarray([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
        profile = CameraProfile(matrix, np.zeros(5), 1280, 720, "phone-main")
        anchors = PitchAnchorSet(
            image_points=np.asarray([[10, 10], [100, 10], [100, 80], [10, 80]]),
            pitch_points=np.asarray([[0, 0], [105, 0], [105, 68], [0, 68]]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "camera.json"
            anchor_path = Path(temporary) / "anchors.json"
            profile.save(profile_path)
            anchors.save(anchor_path)
            loaded_profile = CameraProfile.load(profile_path)
            loaded_anchors = PitchAnchorSet.load(anchor_path)
        np.testing.assert_allclose(loaded_profile.camera_matrix, matrix)
        np.testing.assert_allclose(loaded_anchors.pitch_points, anchors.pitch_points)

    def test_anchor_pose_recovers_ground_intersection(self) -> None:
        _pitch, _profile, geometry, pose = synthetic_camera()
        self.assertTrue(pose.valid)
        self.assertTrue(pose.has_metric_pose)
        point = np.asarray([20.0, 30.0, 0.11])
        pixel = geometry.project_world(point, pose)
        recovered = geometry.intersect_height_plane(tuple(pixel), pose, 0.11)
        self.assertIsNotNone(recovered)
        np.testing.assert_allclose(recovered, point, atol=1e-5)

    def test_profile_rejects_resolution_change(self) -> None:
        _pitch, profile, _geometry, _pose = synthetic_camera()
        with self.assertRaises(ValueError):
            profile.validate_frame_shape((720, 1280, 3))


if __name__ == "__main__":
    unittest.main()
