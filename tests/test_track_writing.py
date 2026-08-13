import unittest

from file_writing.tracks_json_writer import TracksJsonWriter


class TrackWritingTests(unittest.TestCase):
    def test_display_only_speed_is_not_serialized(self) -> None:
        writer = TracksJsonWriter.__new__(TracksJsonWriter)
        result = writer._make_serializable(
            {"player": {7: {"speed": 12.0, "_display_speed": 11.5}}}
        )
        self.assertEqual(result["player"]["7"]["speed"], 12.0)
        self.assertNotIn("_display_speed", result["player"]["7"])


if __name__ == "__main__":
    unittest.main()
