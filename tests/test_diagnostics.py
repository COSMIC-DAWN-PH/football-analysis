import unittest

import numpy as np

from diagnostics import RunDiagnostics
from position_mappers import CameraPoseResult


class RunDiagnosticsTests(unittest.TestCase):
    def test_summary_counts_pose_track_and_reason_coverage(self) -> None:
        diagnostics = RunDiagnostics()
        pose = CameraPoseResult(
            image_to_pitch=np.eye(3, dtype=np.float32),
            status="detected",
            quality=0.9,
            age_seconds=0.0,
        )
        frame = diagnostics.record_frame(
            0.0,
            pose,
            {
                1: {
                    "observed": True,
                    "track_confirmed": False,
                    "track_segment": 2,
                    "track_length": 3,
                    "confidence": 0.1,
                    "track_confidence": 0.1,
                    "speed_state": "unavailable",
                    "speed_reason": "low_ball_confidence",
                }
            },
            candidate_count=4,
        )
        summary = diagnostics.summary()
        self.assertEqual(frame["ball_speed_reason"], "low_ball_confidence")
        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["pose_speed_usable_ratio"], 1.0)
        self.assertEqual(summary["track_segments"], 1)
        self.assertEqual(summary["reason_counts"]["low_ball_confidence"], 1)


if __name__ == "__main__":
    unittest.main()
