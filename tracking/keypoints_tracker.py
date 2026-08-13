from __future__ import annotations

from typing import List, Optional

import numpy as np
import supervision as sv
from ultralytics.engine.results import Results

from tracking.abstract_tracker import AbstractTracker


class KeypointsTracker(AbstractTracker):
    """Detect pitch landmarks periodically while exposing per-point confidence."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.1,
        kp_conf: float = 0.5,
        imgsz: int = 1280,
        detection_interval: int = 5,
    ) -> None:
        super().__init__(model_path, conf, task="pose")
        self.kp_conf = kp_conf
        self.imgsz = self.inference_imgsz(imgsz)
        self.detection_interval = max(1, detection_interval)
        self.cur_frame = 0
        self._detect_frame = 0
        self._force_next = True
        self.current_confidences: dict[int, float] = {}

    def detect(self, frames: List[np.ndarray]) -> List[Optional[Results]]:
        selected_indexes: list[int] = []
        selected_frames: list[np.ndarray] = []
        force = self._force_next
        for index, frame in enumerate(frames):
            absolute_index = self._detect_frame + index
            if force or absolute_index % self.detection_interval == 0:
                selected_indexes.append(index)
                selected_frames.append(frame)
                force = False
        self._force_next = False
        self._detect_frame += len(frames)

        output: List[Optional[Results]] = [None] * len(frames)
        if selected_frames:
            detections = self.model.predict(
                selected_frames,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
            )
            for index, detection in zip(selected_indexes, detections):
                output[index] = detection
        return output

    def track(self, detection: Optional[Results]) -> dict[int, tuple[float, float]]:
        self.cur_frame += 1
        self.current_confidences = {}
        if detection is None or detection.keypoints is None:
            return {}

        keypoints = sv.KeyPoints.from_ultralytics(detection)
        if not keypoints or len(keypoints.xy) == 0:
            return {}
        xy = keypoints.xy[0]
        confidence = keypoints.confidence[0]
        height, width = detection.orig_shape
        result: dict[int, tuple[float, float]] = {}
        for index, (coords, point_confidence) in enumerate(zip(xy, confidence)):
            value = float(point_confidence)
            if (
                value >= self.kp_conf
                and np.isfinite(coords).all()
                and 0 <= coords[0] < width
                and 0 <= coords[1] < height
            ):
                result[index] = (float(coords[0]), float(coords[1]))
                self.current_confidences[index] = value
        return result

    def request_detection(self) -> None:
        self._force_next = True
