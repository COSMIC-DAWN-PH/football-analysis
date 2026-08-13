from __future__ import annotations

from typing import Mapping, Optional

import cv2
import numpy as np

from .abstract_mapper import AbstractMapper
from .camera_calibrator import CalibrationResult, DynamicCameraCalibrator
from .camera_geometry import CameraProfile, PitchAnchorSet
from .pitch_geometry import PitchGeometry
from utils.bbox_utils import get_bbox_center, get_feet_pos


class ObjectPositionMapper(AbstractMapper):
    """Map tracked image objects to stable metric pitch coordinates."""

    def __init__(
        self,
        geometry: PitchGeometry,
        display_size: tuple[int, int] = (528, 352),
        camera_profile: Optional[CameraProfile] = None,
        pitch_anchors: Optional[PitchAnchorSet] = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.display_size = display_size
        self.calibrator = DynamicCameraCalibrator(
            geometry,
            camera_profile=camera_profile,
            pitch_anchors=pitch_anchors,
        )
        self.last_calibration = CalibrationResult(image_to_pitch=None)

    def map(
        self,
        detection: dict,
        frame: np.ndarray,
        timestamp_seconds: float,
        keypoint_confidences: Optional[Mapping[int, float]] = None,
        frame_index: Optional[int] = None,
    ) -> dict:
        detection = detection.copy()
        keypoints = detection.get("keypoints", {})
        object_data = detection.get("object", {})
        moving_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for object_type in ("player", "goalkeeper", "referee"):
            for track_info in object_data.get(object_type, {}).values():
                bbox = track_info.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                pad = max(4, int(max(x2 - x1, y2 - y1) * 0.10))
                cv2.rectangle(
                    moving_mask,
                    (max(0, x1 - pad), max(0, y1 - pad)),
                    (min(frame.shape[1] - 1, x2 + pad), min(frame.shape[0] - 1, y2 + pad)),
                    255,
                    -1,
                )
        self.last_calibration = self.calibrator.update(
            frame,
            keypoints,
            keypoint_confidences,
            timestamp_seconds,
            frame_index=frame_index,
            moving_mask=moving_mask,
        )
        detection["calibration"] = self.last_calibration
        homography = self.last_calibration.image_to_pitch
        if homography is None:
            return detection

        for object_type, tracks in object_data.items():
            for track_info in tracks.values():
                bbox = track_info.get("bbox")
                if bbox is None:
                    continue
                if object_type == "ball":
                    anchor = get_bbox_center(bbox)
                else:
                    anchor = get_feet_pos(bbox)
                source = np.asarray(anchor, dtype=np.float32).reshape(1, 1, 2)
                position = cv2.perspectiveTransform(source, homography).reshape(2)
                metric_position = (float(position[0]), float(position[1]))
                if not self.geometry.contains(metric_position, margin_m=0.0):
                    continue
                track_info["position_m"] = metric_position
                track_info["projection"] = self.geometry.to_display(
                    metric_position, *self.display_size
                )
        return detection

    def reset(self) -> None:
        self.calibrator.reset()
        self.last_calibration = CalibrationResult(image_to_pitch=None)
