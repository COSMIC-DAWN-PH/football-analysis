import unittest
from pathlib import Path

from main import _resolve_output_layout, _run_name
from summarize_match import default_summary_dir


class OutputLayoutTests(unittest.TestCase):
    def test_input_suffix_is_removed_from_run_name(self) -> None:
        self.assertEqual(_run_name(Path("demo2-30s-test-input.mp4")), "demo2-30s-test")
        self.assertEqual(_run_name(Path("match.mp4")), "match")

    def test_default_run_layout_groups_video_and_raw_tracks(self) -> None:
        run_dir, output, tracks_dir = _resolve_output_layout(
            Path("input_videos/match.mp4")
        )
        self.assertEqual(run_dir, Path("output_videos/match"))
        self.assertEqual(output, run_dir / "match-analysis.mp4")
        self.assertEqual(tracks_dir, run_dir / "raw")

    def test_explicit_paths_remain_supported(self) -> None:
        run_dir, output, tracks_dir = _resolve_output_layout(
            Path("match.mp4"),
            run_dir=Path("runs/demo"),
            output=Path("custom/video.mp4"),
            tracks_dir=Path("custom/tracks"),
        )
        self.assertEqual(run_dir, Path("runs/demo"))
        self.assertEqual(output, Path("custom/video.mp4"))
        self.assertEqual(tracks_dir, Path("custom/tracks"))

    def test_summary_is_sibling_of_raw_directory(self) -> None:
        self.assertEqual(
            default_summary_dir(Path("output_videos/match/raw")),
            Path("output_videos/match/summary"),
        )


if __name__ == "__main__":
    unittest.main()
