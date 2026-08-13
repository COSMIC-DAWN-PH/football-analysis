from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from .pitch_geometry import PitchGeometry


@dataclass(frozen=True)
class CalibrationQualityPolicy:
    """One quality policy shared by projection and speed callers."""

    maximum_age_seconds: float = 0.25
    minimum_quality: float = 0.35
    maximum_median_error_px: float = 3.0

    def failure_reason(self, pose: "CameraPoseResult") -> Optional[str]:
        if not pose.valid:
            return pose.failure_reason or "pose_invalid"
        if pose.zoom_changed:
            return "zoom_changed"
        if pose.age_seconds is None or pose.age_seconds > self.maximum_age_seconds:
            return "pose_stale"
        if pose.quality < self.minimum_quality:
            return "pose_invalid"
        if (
            pose.median_error_px is not None
            and pose.status != "propagated"
            and pose.median_error_px > self.maximum_median_error_px
        ):
            return "pose_invalid"
        return None

    def usable(self, pose: "CameraPoseResult") -> bool:
        return self.failure_reason(pose) is None


DEFAULT_CALIBRATION_POLICY = CalibrationQualityPolicy()


@dataclass(frozen=True)
class CameraProfile:
    """Intrinsic calibration for one locked camera resolution and focal setting."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_width: int
    image_height: int
    profile_id: str = "default"
    focal_setting: str = "locked"
    version: int = 1
    rms_error_px: Optional[float] = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        if distortion.size < 4 or not np.isfinite(distortion).all():
            raise ValueError("distortion_coefficients must contain at least four finite values")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Camera Profile resolution must be positive")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height

    def validate_frame_shape(self, frame_shape: tuple[int, ...]) -> None:
        height, width = frame_shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(
                "Camera Profile resolution "
                f"{self.image_width}x{self.image_height} does not match frame {width}x{height}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "profile_id": self.profile_id,
            "focal_setting": self.focal_setting,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "rms_error_px": self.rms_error_px,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera_matrix=np.asarray(payload["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(
                payload["distortion_coefficients"], dtype=np.float64
            ),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            profile_id=str(payload.get("profile_id", "default")),
            focal_setting=str(payload.get("focal_setting", "locked")),
            version=int(payload.get("version", 1)),
            rms_error_px=(
                None
                if payload.get("rms_error_px") is None
                else float(payload["rms_error_px"])
            ),
        )


@dataclass(frozen=True)
class PitchAnchorSet:
    """Metric pitch correspondences for one reference frame of a continuous shot."""

    image_points: np.ndarray
    pitch_points: np.ndarray
    reference_frame: int = 0
    source: str = "manual"
    version: int = 1

    def __post_init__(self) -> None:
        image = np.asarray(self.image_points, dtype=np.float64).reshape(-1, 2)
        pitch = np.asarray(self.pitch_points, dtype=np.float64).reshape(-1, 2)
        if len(image) != len(pitch) or len(image) < 4:
            raise ValueError("Pitch Anchor Set requires at least four paired points")
        if not np.isfinite(image).all() or not np.isfinite(pitch).all():
            raise ValueError("Pitch Anchor Set points must be finite")
        if np.ptp(pitch[:, 0]) <= 0 or np.ptp(pitch[:, 1]) <= 0:
            raise ValueError("Pitch anchors must span both pitch length and width")
        object.__setattr__(self, "image_points", image)
        object.__setattr__(self, "pitch_points", pitch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "reference_frame": self.reference_frame,
            "image_points": self.image_points.tolist(),
            "pitch_points": self.pitch_points.tolist(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PitchAnchorSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            image_points=np.asarray(payload["image_points"], dtype=np.float64),
            pitch_points=np.asarray(payload["pitch_points"], dtype=np.float64),
            reference_frame=int(payload.get("reference_frame", 0)),
            source=str(payload.get("source", "manual")),
            version=int(payload.get("version", 1)),
        )


@dataclass
class CameraPoseResult:
    """Per-frame pitch projection, camera pose, quality, and failure state."""

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
    flow_quality: float = 0.0
    quality: float = 0.0
    age_seconds: Optional[float] = None
    reset_required: bool = False
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    pose_uncertainty: Optional[float] = None
    marking_error_px: Optional[float] = None
    camera_profile_id: Optional[str] = None
    failure_reason: Optional[str] = None
    zoom_changed: bool = False

    @property
    def valid(self) -> bool:
        return self.image_to_pitch is not None

    @property
    def has_metric_pose(self) -> bool:
        return (
            self.valid
            and self.rotation_vector is not None
            and self.translation_vector is not None
        )

    @property
    def speed_usable(self) -> bool:
        return DEFAULT_CALIBRATION_POLICY.usable(self)

    @property
    def speed_failure_reason(self) -> Optional[str]:
        return DEFAULT_CALIBRATION_POLICY.failure_reason(self)

    def serializable(
        self,
        timestamp_seconds: float,
        *,
        source_frame: Optional[int] = None,
        processed_frame: Optional[int] = None,
    ) -> dict[str, Any]:
        result = asdict(self)
        for key in ("image_to_pitch", "rotation_vector", "translation_vector"):
            value = result[key]
            if value is not None:
                result[key] = np.asarray(value).tolist()
        result["timestamp_seconds"] = float(timestamp_seconds)
        result["source_frame"] = source_frame
        result["processed_frame"] = processed_frame
        return result


class CameraGeometry:
    """Recover camera pose and perform metric ray/project operations behind one Interface."""

    def __init__(
        self,
        pitch: PitchGeometry,
        profile: Optional[CameraProfile] = None,
    ) -> None:
        self.pitch = pitch
        self.profile = profile

    def pose_from_correspondences(
        self,
        image_points: Iterable[Iterable[float]],
        pitch_points: Iterable[Iterable[float]],
        *,
        status: str,
        timestamp_age_seconds: float = 0.0,
        ransac_threshold_px: float = 4.0,
    ) -> CameraPoseResult:
        image = np.asarray(list(image_points), dtype=np.float32).reshape(-1, 2)
        pitch = np.asarray(list(pitch_points), dtype=np.float32).reshape(-1, 2)
        result = CameraPoseResult(image_to_pitch=None, status="invalid", keypoints=len(image))
        if len(image) < 4 or len(image) != len(pitch):
            result.failure_reason = "pose_invalid"
            return result
        world_to_image, mask = cv2.findHomography(
            pitch, image, cv2.RANSAC, ransac_threshold_px
        )
        if world_to_image is None or mask is None:
            result.failure_reason = "pose_invalid"
            return result
        inliers = mask.ravel().astype(bool)
        result.inliers = int(inliers.sum())
        result.inlier_ratio = result.inliers / len(image)
        projected = cv2.perspectiveTransform(
            pitch.reshape(-1, 1, 2), world_to_image
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - image, axis=1)
        result.median_error_px = (
            float(np.median(errors[inliers])) if result.inliers else None
        )
        result.span_length_ratio = float(
            np.ptp(pitch[inliers, 0]) / self.pitch.length_m
        ) if result.inliers else 0.0
        result.span_width_ratio = float(
            np.ptp(pitch[inliers, 1]) / self.pitch.width_m
        ) if result.inliers else 0.0
        try:
            image_to_pitch = np.linalg.inv(world_to_image)
        except np.linalg.LinAlgError:
            result.failure_reason = "pose_invalid"
            return result
        if not np.isfinite(image_to_pitch).all():
            result.failure_reason = "pose_invalid"
            return result
        result.image_to_pitch = image_to_pitch.astype(np.float32)
        result.status = status
        result.age_seconds = float(timestamp_age_seconds)
        error = float(result.median_error_px or 0.0)
        result.quality = float(
            max(0.0, 1.0 - error / 5.0)
            * min(1.0, result.inlier_ratio)
            * min(1.0, result.span_length_ratio / 0.30)
            * min(1.0, result.span_width_ratio / 0.25)
        )
        self.attach_metric_pose(result)
        return result

    def attach_metric_pose(self, result: CameraPoseResult) -> CameraPoseResult:
        if result.image_to_pitch is None or self.profile is None:
            return result
        try:
            world_to_image = np.linalg.inv(result.image_to_pitch)
            intrinsic_inverse = np.linalg.inv(self.profile.camera_matrix)
        except np.linalg.LinAlgError:
            result.failure_reason = "pose_invalid"
            return result
        normalized = intrinsic_inverse @ world_to_image
        scale = 2.0 / (
            np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1])
        )
        r1 = normalized[:, 0] * scale
        r2 = normalized[:, 1] * scale
        r3 = np.cross(r1, r2)
        rotation_guess = np.column_stack([r1, r2, r3])
        u, _singular, vt = np.linalg.svd(rotation_guess)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
        translation = normalized[:, 2] * scale
        if translation[2] < 0:
            rotation[:, :2] *= -1
            translation *= -1
        result.rotation_vector = cv2.Rodrigues(rotation)[0].reshape(3)
        result.translation_vector = translation.reshape(3)
        result.camera_profile_id = self.profile.profile_id
        if result.median_error_px is not None:
            result.pose_uncertainty = float(result.median_error_px)
        return result

    def ray_from_pixel(
        self, pixel: tuple[float, float], pose: CameraPoseResult
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.profile is None or not pose.has_metric_pose:
            raise ValueError("Camera Profile and metric Camera Pose are required")
        point = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
        normalized = cv2.undistortPoints(
            point,
            self.profile.camera_matrix,
            self.profile.distortion_coefficients,
        ).reshape(2)
        direction_camera = np.asarray([normalized[0], normalized[1], 1.0])
        rotation = cv2.Rodrigues(
            np.asarray(pose.rotation_vector, dtype=np.float64).reshape(3)
        )[0]
        translation = np.asarray(pose.translation_vector, dtype=np.float64).reshape(3)
        origin_world = -rotation.T @ translation
        direction_world = rotation.T @ direction_camera
        direction_world /= np.linalg.norm(direction_world)
        return origin_world, direction_world

    def intersect_height_plane(
        self,
        pixel: tuple[float, float],
        pose: CameraPoseResult,
        height_m: float,
    ) -> Optional[np.ndarray]:
        origin, direction = self.ray_from_pixel(pixel, pose)
        if abs(direction[2]) < 1e-8:
            return None
        distance = (height_m - origin[2]) / direction[2]
        if distance <= 0:
            return None
        point = origin + distance * direction
        return point if np.isfinite(point).all() else None

    def project_world(
        self, point: Iterable[float], pose: CameraPoseResult
    ) -> np.ndarray:
        if self.profile is None or not pose.has_metric_pose:
            raise ValueError("Camera Profile and metric Camera Pose are required")
        projected, _ = cv2.projectPoints(
            np.asarray(point, dtype=np.float64).reshape(1, 1, 3),
            np.asarray(pose.rotation_vector, dtype=np.float64),
            np.asarray(pose.translation_vector, dtype=np.float64),
            self.profile.camera_matrix,
            self.profile.distortion_coefficients,
        )
        return projected.reshape(2)
