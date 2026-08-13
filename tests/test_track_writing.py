import unittest
import json
import tempfile
from pathlib import Path

from file_writing.tracks_json_writer import TracksJsonWriter


class TrackWritingTests(unittest.TestCase):
    def test_display_only_speed_is_not_serialized(self) -> None:
        writer = TracksJsonWriter.__new__(TracksJsonWriter)
        result = writer._make_serializable(
            {"player": {7: {"speed": 12.0, "_display_speed": 11.5}}}
        )
        self.assertEqual(result["player"]["7"]["speed"], 12.0)
        self.assertNotIn("_display_speed", result["player"]["7"])

    def test_quality_summary_is_written_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = TracksJsonWriter(temporary)
            writer.write_summary({"frames": 12, "pose_valid_ratio": 0.75})
            payload = json.loads(Path(writer.get_summary_path()).read_text(encoding="utf-8"))
        self.assertEqual(payload["frames"], 12)


if __name__ == "__main__":
    unittest.main()
