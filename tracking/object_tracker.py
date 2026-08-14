from __future__ import annotations

from typing import List

import numpy as np
import supervision as sv
from ultralytics.engine.results import Results

from tracking.abstract_tracker import AbstractTracker


class ObjectTracker(AbstractTracker):
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        ball_conf: float = 0.05,
        imgsz: int = 1280,
        include_ball: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__(model_path, conf, task="detect", device=device)
        self.ball_conf = ball_conf
        self.imgsz = self.inference_imgsz(imgsz)
        self.include_ball = include_ball
        self.classes = ["ball", "goalkeeper", "player", "referee"]
        self.frame_rate = 30.0
        self.tracker = self._new_tracker(self.frame_rate)
        self.cur_frame = 0

    @staticmethod
    def _new_tracker(frame_rate: float) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=max(1, int(round(frame_rate))),
            minimum_consecutive_frames=1,
        )

    def set_frame_rate(self, frame_rate: float) -> None:
        frame_rate = max(float(frame_rate), 1.0)
        if abs(frame_rate - self.frame_rate) < 0.5:
            return
        self.frame_rate = frame_rate
        self.tracker = self._new_tracker(frame_rate)

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        return self.model.predict(
            frames,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )

    def track(self, detection: Results) -> dict:
        detections = sv.Detections.from_ultralytics(detection)
        if not self.include_ball and len(detections):
            detections = detections[detections.class_id != 0]
        tracks = self.tracker.update_with_detections(detections)
        result = self._tracks_mapper(tracks, self.classes)
        self.cur_frame += 1
        return result

    def _tracks_mapper(self, tracks: sv.Detections, class_names: List[str]) -> dict:
        result = {class_name: {} for class_name in class_names}
        if len(tracks) == 0:
            return result

        for bbox, class_id, track_id, confidence in zip(
            tracks.xyxy,
            tracks.class_id,
            tracks.tracker_id,
            tracks.confidence,
        ):
            if track_id is None or class_id is None:
                continue
            class_index = int(class_id)
            if class_index < 0 or class_index >= len(class_names):
                continue
            class_name = class_names[class_index]
            confidence_value = float(confidence)
            if class_name == "ball" and confidence_value < self.ball_conf:
                continue
            if not np.isfinite(bbox).all():
                continue
            result[class_name][int(track_id)] = {
                "bbox": [float(value) for value in bbox],
                "confidence": confidence_value,
            }
        return result
