import unittest
from types import SimpleNamespace

import numpy as np

from tools.eval_ball_rescue_detection import (
    _detect_rois,
    _evenly_spaced,
    _triggered,
)
from tracking import BallCandidate


class _Detector:
    def __init__(self) -> None:
        self.calls = []

    def _predict(self, crop, offset_x, offset_y, image_size):
        self.calls.append((crop.shape, offset_x, offset_y, image_size))
        return [
            BallCandidate(
                (offset_x + 1, offset_y + 2, offset_x + 5, offset_y + 6),
                0.5,
            )
        ]


class BallRescueInferenceTests(unittest.TestCase):
    def test_confirmed_gap_trigger_does_not_scan_unconfirmed_or_observed_rows(self) -> None:
        self.assertTrue(
            _triggered(
                {"track": {"observed": False, "track_confirmed": True}},
                "confirmed-gap",
            )
        )
        self.assertFalse(
            _triggered(
                {"track": {"observed": False, "track_confirmed": False}},
                "confirmed-gap",
            )
        )
        self.assertFalse(
            _triggered(
                {"track": {"observed": True, "track_confirmed": True}},
                "confirmed-gap",
            )
        )

    def test_roi_detection_maps_crop_candidate_to_source_frame(self) -> None:
        detector = _Detector()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        rois = [SimpleNamespace(bbox=(10.2, 20.8, 40.1, 60.2))]
        candidates = _detect_rois(detector, frame, rois, 640)
        self.assertEqual(candidates[0].bbox, (11, 22, 15, 26))
        self.assertEqual(detector.calls, [((41, 31, 3), 10, 20, 640)])

    def test_evenly_spaced_subset_keeps_endpoints(self) -> None:
        items = [{"frame": frame} for frame in range(10)]
        selected = _evenly_spaced(items, 4)
        self.assertEqual([item["frame"] for item in selected], [0, 3, 6, 9])


if __name__ == "__main__":
    unittest.main()
