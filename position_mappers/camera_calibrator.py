from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

import cv2
import numpy as np

from .pitch_geometry import PitchGeometry


@dataclass
class CalibrationResult:
    image_to_pitch: Optional[np.ndarray]
    status: str = "invalid"
    keypoints: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_error_px: Optional[float] = None
    span_length_ratio: float = 0.0
    span_width_ratio: float = 0.0
    flow_points: int = 0
    flow_inliers: int = 0
    age_seconds: Optional[float] = None

    @property
    def valid(self) -> bool:
        return self.image_to_pitch is not None

    @property
    def speed_usable(self) -> bool:
        return self.valid and self.age_seconds is not None and self.age_seconds <= 0.5

    def serializable(self, timestamp_seconds: float) -> dict:
        result = asdict(self)
        matrix = result.pop("image_to_pitch")
        result["timestamp_seconds"] = timestamp_seconds
        if matrix is not None:
            result["image_to_pitch"] = matrix.tolist()
        return result


class DynamicCameraCalibrator:
    """Estimate and propagate a moving camera's pitch homography."""

    def __init__(
        self,
        geometry: PitchGeometry,
        min_keypoints: int = 6,
        min_inliers: int = 5,
        min_inlier_ratio: float = 0.60,
        max_median_error_px: float = 5.0,
        min_length_span_ratio: float = 0.30,
        min_width_span_ratio: float = 0.25,
        max_propagation_seconds: float = 1.0,
        max_flow_error_px: float = 1.5,
        min_flow_points: int = 20,
    ) -> None:
        self.geometry = geometry
        self.min_keypoints = min_keypoints
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_median_error_px = max_median_error_px
        self.min_length_span_ratio = min_length_span_ratio
        self.min_width_span_ratio = min_width_span_ratio
        self.max_propagation_seconds = max_propagation_seconds
        self.max_flow_error_px = max_flow_error_px
        self.min_flow_points = min_flow_points

        self._image_to_pitch: Optional[np.ndarray] = None
        self._last_direct_timestamp: Optional[float] = None
        self._previous_gray: Optional[np.ndarray] = None
        self._feature_image_points: Optional[np.ndarray] = None
        self._feature_world_points: Optional[np.ndarray] = None
        self._previous_histogram: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._image_to_pitch = None
        self._last_direct_timestamp = None
        self._previous_gray = None
        self._feature_image_points = None
        self._feature_world_points = None
        self._previous_histogram = None

    def update(
        self,
        frame: np.ndarray,
        keypoints: Mapping[int, tuple[float, float]],
        confidences: Optional[Mapping[int, float]],
        timestamp_seconds: float,
    ) -> CalibrationResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = self._frame_histogram(frame)
        direct = self._estimate_direct(keypoints, confidences)
        flow = self._estimate_flow(gray)

        cut_likely = False
        if self._previous_histogram is not None and flow is None:
            correlation = cv2.compareHist(
                self._previous_histogram, histogram, cv2.HISTCMP_CORREL
            )
            cut_likely = correlation < 0.35
        if cut_likely:
            self._image_to_pitch = None
            self._last_direct_timestamp = None
            flow = None

        result: CalibrationResult
        if direct.valid:
            selected = direct
            if flow is not None:
                fused = self._fuse_direct_and_flow(keypoints, confidences, flow)
                if fused is not None:
                    selected.image_to_pitch = fused
                    selected.status = "fused"
                    selected.flow_points = flow[2]
                    selected.flow_inliers = flow[3]
            self._image_to_pitch = selected.image_to_pitch
            self._last_direct_timestamp = timestamp_seconds
            selected.age_seconds = 0.0
            result = selected
        elif flow is not None and self._last_direct_timestamp is not None:
            age = max(0.0, timestamp_seconds - self._last_direct_timestamp)
            if age <= self.max_propagation_seconds:
                self._image_to_pitch = flow[0]
                result = CalibrationResult(
                    image_to_pitch=flow[0],
                    status="propagated",
                    keypoints=direct.keypoints,
                    flow_points=flow[2],
                    flow_inliers=flow[3],
                    age_seconds=age,
                )
            else:
                self._image_to_pitch = None
                result = direct
        else:
            self._image_to_pitch = None
            result = direct

        self._previous_gray = gray
        self._previous_histogram = histogram
        if self._image_to_pitch is not None:
            self._seed_features(gray, self._image_to_pitch)
        else:
            self._feature_image_points = None
            self._feature_world_points = None
        return result

    def _estimate_direct(
        self,
        keypoints: Mapping[int, tuple[float, float]],
        confidences: Optional[Mapping[int, float]],
    ) -> CalibrationResult:
        indexes = [
            int(index)
            for index in keypoints
            if 0 <= int(index) < len(self.geometry.vertices)
            and (confidences is None or confidences.get(int(index), 0.0) > 0.0)
        ]
        result = CalibrationResult(image_to_pitch=None, keypoints=len(indexes))
        if len(indexes) < self.min_keypoints:
            return result

        world = self.geometry.vertices[indexes]
        image = np.asarray([keypoints[index] for index in indexes], dtype=np.float32)
        world_to_image, mask = cv2.findHomography(world, image, cv2.RANSAC, 5.0)
        if world_to_image is None or mask is None:
            return result

        mask = mask.ravel().astype(bool)
        projected = cv2.perspectiveTransform(
            world.reshape(-1, 1, 2), world_to_image
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - image, axis=1)
        inliers = int(mask.sum())
        ratio = inliers / len(indexes)
        median_error = float(np.median(errors[mask])) if inliers else float("inf")
        length_span = (
            float(np.ptp(world[mask, 0]) / self.geometry.length_m) if inliers else 0.0
        )
        width_span = (
            float(np.ptp(world[mask, 1]) / self.geometry.width_m) if inliers else 0.0
        )
        result.inliers = inliers
        result.inlier_ratio = ratio
        result.median_error_px = median_error
        result.span_length_ratio = length_span
        result.span_width_ratio = width_span
        if (
            inliers < self.min_inliers
            or ratio < self.min_inlier_ratio
            or median_error > self.max_median_error_px
            or length_span < self.min_length_span_ratio
            or width_span < self.min_width_span_ratio
        ):
            return result

        try:
            image_to_pitch = np.linalg.inv(world_to_image)
        except np.linalg.LinAlgError:
            return result
        if not np.isfinite(image_to_pitch).all():
            return result
        result.image_to_pitch = image_to_pitch.astype(np.float32)
        result.status = "detected"
        return result

    def _estimate_flow(
        self, gray: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray, int, int]]:
        if (
            self._previous_gray is None
            or self._feature_image_points is None
            or self._feature_world_points is None
            or len(self._feature_image_points) < self.min_flow_points
        ):
            return None

        previous = self._feature_image_points.reshape(-1, 1, 2).astype(np.float32)
        current, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            previous,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current is None or status_forward is None:
            return None
        backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self._previous_gray,
            current,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward is None or status_backward is None:
            return None

        forward_backward = np.linalg.norm(
            previous.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
        )
        valid = (
            status_forward.ravel().astype(bool)
            & status_backward.ravel().astype(bool)
            & (forward_backward <= self.max_flow_error_px)
        )
        valid_count = int(valid.sum())
        if valid_count < self.min_flow_points:
            return None

        world = self._feature_world_points[valid]
        image = current.reshape(-1, 2)[valid]
        world_to_image, mask = cv2.findHomography(world, image, cv2.RANSAC, 3.0)
        if world_to_image is None or mask is None:
            return None
        flow_inliers = int(mask.sum())
        if flow_inliers < self.min_flow_points:
            return None
        try:
            image_to_pitch = np.linalg.inv(world_to_image)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(image_to_pitch).all():
            return None
        return image_to_pitch.astype(np.float32), image, valid_count, flow_inliers

    def _fuse_direct_and_flow(
        self,
        keypoints: Mapping[int, tuple[float, float]],
        confidences: Optional[Mapping[int, float]],
        flow: tuple[np.ndarray, np.ndarray, int, int],
    ) -> Optional[np.ndarray]:
        if self._feature_world_points is None:
            return None
        indexes = [
            int(index)
            for index in keypoints
            if 0 <= int(index) < len(self.geometry.vertices)
            and (confidences is None or confidences.get(int(index), 0.0) > 0.0)
        ]
        if not indexes:
            return None

        # Direct landmarks are repeated so that dense optical-flow points smooth the
        # camera motion without overpowering the absolute field registration.
        direct_world = np.repeat(self.geometry.vertices[indexes], 4, axis=0)
        direct_image = np.repeat(
            np.asarray([keypoints[index] for index in indexes], dtype=np.float32), 4, axis=0
        )
        current_flow_image = flow[1]
        flow_world = cv2.perspectiveTransform(
            current_flow_image.reshape(-1, 1, 2), flow[0]
        ).reshape(-1, 2)
        world = np.concatenate([direct_world, flow_world], axis=0)
        image = np.concatenate([direct_image, current_flow_image], axis=0)
        world_to_image, _ = cv2.findHomography(world, image, cv2.RANSAC, 4.0)
        if world_to_image is None:
            return None
        projected_direct = cv2.perspectiveTransform(
            self.geometry.vertices[indexes].reshape(-1, 1, 2), world_to_image
        ).reshape(-1, 2)
        error = np.median(
            np.linalg.norm(
                projected_direct
                - np.asarray([keypoints[index] for index in indexes], dtype=np.float32),
                axis=1,
            )
        )
        if error > self.max_median_error_px:
            return None
        try:
            result = np.linalg.inv(world_to_image)
        except np.linalg.LinAlgError:
            return None
        return result.astype(np.float32) if np.isfinite(result).all() else None

    def _seed_features(self, gray: np.ndarray, image_to_pitch: np.ndarray) -> None:
        height, width = gray.shape
        mask = np.zeros_like(gray)
        try:
            pitch_to_image = np.linalg.inv(image_to_pitch)
        except np.linalg.LinAlgError:
            self._feature_image_points = None
            self._feature_world_points = None
            return
        corners = np.asarray(
            [[0, 0], [self.geometry.length_m, 0], [self.geometry.length_m, self.geometry.width_m], [0, self.geometry.width_m]],
            dtype=np.float32,
        )
        image_corners = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), pitch_to_image
        ).reshape(-1, 2)
        image_corners[:, 0] = np.clip(image_corners[:, 0], 0, width - 1)
        image_corners[:, 1] = np.clip(image_corners[:, 1], 0, height - 1)
        cv2.fillConvexPoly(mask, image_corners.astype(np.int32), 255)
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask,
            blockSize=7,
        )
        if features is None:
            self._feature_image_points = None
            self._feature_world_points = None
            return
        image_points = features.reshape(-1, 2)
        world_points = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2), image_to_pitch
        ).reshape(-1, 2)
        valid = np.asarray(
            [self.geometry.contains(tuple(point), margin_m=0.0) for point in world_points],
            dtype=bool,
        )
        self._feature_image_points = image_points[valid]
        self._feature_world_points = world_points[valid]

    @staticmethod
    def _frame_histogram(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        return cv2.normalize(histogram, histogram).astype(np.float32)
