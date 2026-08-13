import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from annotation.football_video_processor import FootballVideoProcessor
from annotation.object_annotator import ObjectAnnotator
from annotation.projection_annotator import ProjectionAnnotator
from ball_to_player_assignment import BallToPlayerAssigner
from club_assignment import Club
from position_mappers import ObjectPositionMapper
from position_mappers import PitchGeometry


class ResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.club1 = Club("one", (255, 0, 0), (0, 0, 0))
        self.club2 = Club("two", (0, 0, 255), (255, 255, 0))

    def test_mapping_waits_for_enough_keypoints(self) -> None:
        mapper = ObjectPositionMapper(PitchGeometry(105, 68))
        tracks = {
            "keypoints": {},
            "object": {
                "ball": {},
                "goalkeeper": {},
                "player": {1: {"bbox": [0, 0, 10, 20]}},
                "referee": {},
            },
        }
        mapped = mapper.map(tracks, np.zeros((1080, 1920, 3), dtype=np.uint8), 0.0)
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

    def test_ball_speed_label_is_drawn(self) -> None:
        annotator = ObjectAnnotator()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracks = {
            "ball": {
                1: {
                    "bbox": [100, 100, 108, 108],
                    "speed": 36.0,
                    "speed_status": "reliable",
                }
            },
            "goalkeeper": {},
            "player": {},
            "referee": {},
        }
        with patch("annotation.object_annotator.cv2.putText") as put_text:
            annotator.annotate(frame, tracks)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Ball 36.0 km/h", labels)

    def test_pending_ball_speed_label_is_drawn(self) -> None:
        annotator = ObjectAnnotator()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracks = {
            "ball": {1: {"bbox": [100, 100, 108, 108], "speed_status": "pending"}},
            "goalkeeper": {},
            "player": {},
            "referee": {},
        }
        with patch("annotation.object_annotator.cv2.putText") as put_text:
            annotator.annotate(frame, tracks)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Ball speed: pending", labels)

    def test_unconfirmed_possession_is_explicitly_labeled(self) -> None:
        processor = FootballVideoProcessor.__new__(FootballVideoProcessor)
        processor.ball_to_player_assigner = SimpleNamespace(
            get_current_possession=lambda: -1,
            get_ball_possessions=lambda: [{-1: 1.0, 0: 0.0, 1: 0.0}],
        )
        processor.club_assigner = SimpleNamespace(
            club1=SimpleNamespace(player_jersey_color=(255, 0, 0)),
            club2=SimpleNamespace(player_jersey_color=(0, 0, 255)),
        )
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        with patch("annotation.football_video_processor.cv2.putText") as put_text:
            processor._annotate_possession(frame)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Possession: Unconfirmed", labels)


if __name__ == "__main__":
    unittest.main()
