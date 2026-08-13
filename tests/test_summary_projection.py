import unittest

import cv2
import numpy as np

from position_mappers import PitchGeometry
from summarize_match import frame_homography, project_players


class SummaryProjectionTests(unittest.TestCase):
    def test_metric_position_takes_priority(self) -> None:
        geometry = PitchGeometry(105, 68)
        objects = {
            "player": {
                "7": {
                    "bbox": [0, 0, 10, 20],
                    "club": "Red",
                    "position_m": [30.0, 20.0],
                }
            },
            "goalkeeper": {},
        }
        result = project_players(objects, np.eye(3), geometry)
        self.assertEqual(result["Red"], [(30.0, 20.0)])

    def test_legacy_homography_uses_metric_pitch_geometry(self) -> None:
        geometry = PitchGeometry(100, 64)
        world_to_image = np.asarray(
            [[9.0, 0.8, 120.0], [0.2, 7.0, 90.0], [0.001, 0.0005, 1.0]],
            dtype=np.float32,
        )
        image = cv2.perspectiveTransform(
            geometry.vertices.reshape(-1, 1, 2), world_to_image
        ).reshape(-1, 2)
        keypoints = {str(index): point.tolist() for index, point in enumerate(image)}
        homography, quality = frame_homography(keypoints, geometry)
        self.assertIsNotNone(homography)
        self.assertEqual(quality["inliers"], 32)
        recovered = cv2.perspectiveTransform(
            np.asarray([[[50.0, 32.0]]], dtype=np.float32), world_to_image
        )
        recovered = cv2.perspectiveTransform(recovered, homography)
        np.testing.assert_allclose(recovered.reshape(2), [50.0, 32.0], atol=0.1)


if __name__ == "__main__":
    unittest.main()
