import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from annotation import AbstractVideoProcessor
from utils.video_utils import process_video


class _RecordingProcessor(AbstractVideoProcessor):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.source_frames: list[int] = []
        self.finalized = False

    def process(
        self,
        frames,
        fps=1e-6,
        timestamps=None,
        source_frame_numbers=None,
    ):
        if self.fail:
            raise ValueError("synthetic processing failure")
        self.source_frames.extend(source_frame_numbers or [])
        return [frame.copy() for frame in frames]

    def finalize(self) -> None:
        self.finalized = True


class VideoProcessingTests(unittest.TestCase):
    def _video(self, root: Path, frames: int = 12, fps: float = 10.0) -> Path:
        path = root / "source.avi"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (32, 24),
        )
        for index in range(frames):
            writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
        writer.release()
        return path

    def test_source_frame_numbers_survive_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            processor = _RecordingProcessor()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                process_video(
                    processor,
                    str(self._video(Path(temporary))),
                    output_video=None,
                    batch_size=2,
                    skip_seconds=1,
                    preview=False,
                )
        self.assertEqual(processor.source_frames, [10, 11])
        self.assertTrue(processor.finalized)

    def test_processing_failure_is_not_silently_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            processor = _RecordingProcessor(fail=True)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "processing stage failed"):
                    process_video(
                        processor,
                        str(self._video(Path(temporary), frames=2)),
                        output_video=None,
                        batch_size=1,
                        preview=False,
                    )
        self.assertTrue(processor.finalized)


if __name__ == "__main__":
    unittest.main()
