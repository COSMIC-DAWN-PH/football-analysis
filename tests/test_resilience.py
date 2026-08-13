import unittest

import numpy as np

from annotation.projection_annotator import ProjectionAnnotator
from ball_to_player_assignment import BallToPlayerAssigner
from club_assignment import Club
from position_mappers import ObjectPositionMapper


class ResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.club1 = Club("one", (255, 0, 0), (0, 0, 0))
        self.club2 = Club("two", (0, 0, 255), (255, 255, 0))

    def test_mapping_waits_for_enough_keypoints(self) -> None:
        mapper = ObjectPositionMapper(np.zeros((32, 2), dtype=np.float32))
        tracks = {
            "keypoints": {},
            "object": {
                "ball": {},
                "goalkeeper": {},
                "player": {1: {"bbox": [0, 0, 10, 20]}},
                "referee": {},
            },
        }
        mapped = mapper.map(tracks)
        self.assertNotIn("projection", mapped["object"]["player"][1])

    def test_ball_assignment_skips_unprojected_ball(self) -> None:
        assigner = BallToPlayerAssigner(self.club1, self.club2)
        tracks = {
            "ball": {1: {"bbox": [0, 0, 10, 10]}},
            "goalkeeper": {},
            "player": {},
            "referee": {},
        }
        result, player_id = assigner.assign(tracks, 0, None, None)
        self.assertEqual(player_id, -1)
        self.assertIn(1, result["ball"])

    def test_projection_annotation_skips_unprojected_objects(self) -> None:
        annotator = ProjectionAnnotator()
        field = np.zeros((352, 528, 3), dtype=np.uint8)
        tracks = {
            "ball": {1: {"bbox": [0, 0, 10, 10]}},
            "goalkeeper": {},
            "player": {1: {"bbox": [0, 0, 10, 20]}},
            "referee": {},
        }
        annotated = annotator.annotate(field, tracks)
        self.assertEqual(annotated.shape, field.shape)


if __name__ == "__main__":
    unittest.main()
