import unittest

import numpy as np

from tools.measure_ball_dense_tiles import (
    _dense_candidates,
    _evenly_spaced,
    _parse_layout,
)
from tracking import BallCandidate


class _Detector:
    tile_imgsz = 1280

    def __init__(self) -> None:
        self.calls = []

    def _predict(self, tile, offset_x, offset_y, image_size):
        self.calls.append((tile.shape, offset_x, offset_y, image_size))
        return [
            BallCandidate(
                (offset_x + 1, offset_y + 2, offset_x + 5, offset_y + 6),
                0.5,
            )
        ]


class BallDenseTileTests(unittest.TestCase):
    def test_layout_parser(self) -> None:
        self.assertEqual(_parse_layout("2x3"), (2, 3))
        with self.assertRaises(ValueError):
            _parse_layout("2-by-3")
        with self.assertRaises(ValueError):
            _parse_layout("0x3")

    def test_dense_candidates_keep_source_offsets(self) -> None:
        detector = _Detector()
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        candidates = _dense_candidates(
            detector, frame, rows=1, columns=2, overlap=0.0
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].bbox, (1, 2, 5, 6))
        self.assertEqual(candidates[1].bbox, (61, 2, 65, 6))
        self.assertEqual(
            detector.calls,
            [
                ((40, 60, 3), 0, 0, 1280),
                ((40, 60, 3), 60, 0, 1280),
            ],
        )

    def test_evenly_spaced_subset_keeps_endpoints(self) -> None:
        items = [{"frame": frame} for frame in range(10)]
        selected = _evenly_spaced(items, 4)
        self.assertEqual([item["frame"] for item in selected], [0, 3, 6, 9])

if __name__ == "__main__":
    unittest.main()
