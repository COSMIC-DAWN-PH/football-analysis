from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from diagnostics import BallSpeedReason, BallSpeedState
from position_mappers import (
    CameraGeometry,
    CameraPoseResult,
    CameraProfile,
    PitchGeometry,
)


@dataclass(frozen=True)
class BallKinematicsSample:
    timestamp: float
    pixel: tuple[float, float]
    bbox_size_px: float
    pose: CameraPoseResult
    candidate_confidence: float
    track_confidence: float


@dataclass(frozen=True)
class BallSpeedResult:
    state: str
    reason: Optional[str] = None
    speed_3d_kmh: Optional[float] = None
    velocity_mps: Optional[tuple[float, float, float]] = None
    motion_mode: Optional[str] = None
    uncertainty_kmh: Optional[float] = None
    position_3d_m: Optional[tuple[float, float, float]] = None
    sample_count: int = 0
    sample_span_seconds: float = 0.0
    reprojection_error_px: Optional[float] = None


@dataclass(frozen=True)
class _MotionFit:
    mode: str
    speed_kmh: float
    velocity_mps: np.ndarray
    position_m: np.ndarray
    uncertainty_kmh: float
    reprojection_error_px: float


class BallKinematics3D:
    """Estimate conservative ground/air Ball Kinematics from a calibrated monocular shot."""

    def __init__(
        self,
        geometry: PitchGeometry,
        camera_profile: CameraProfile,
        *,
        ball_diameter_m: float = 0.22,
        minimum_ground_samples: int = 8,
        minimum_air_samples: int = 12,
        minimum_ground_span_seconds: float = 0.25,
        minimum_air_span_seconds: float = 0.35,
        maximum_gap_seconds: float = 0.12,
        history_window_seconds: float = 0.80,
        maximum_uncertainty_kmh: float = 5.0,
        maximum_reprojection_error_px: float = 3.0,
        maximum_speed_mps: float = 40.0,
        minimum_candidate_confidence: float = 0.20,
        minimum_track_confidence: float = 0.25,
    ) -> None:
        self.geometry = geometry
        self.camera_profile = camera_profile
        self.camera_geometry = CameraGeometry(geometry, camera_profile)
        self.ball_diameter_m = ball_diameter_m
        self.ball_radius_m = ball_diameter_m / 2.0
        self.minimum_ground_samples = minimum_ground_samples
        self.minimum_air_samples = minimum_air_samples
        self.minimum_ground_span_seconds = minimum_ground_span_seconds
        self.minimum_air_span_seconds = minimum_air_span_seconds
        self.maximum_gap_seconds = maximum_gap_seconds
        self.history_window_seconds = history_window_seconds
        self.maximum_uncertainty_kmh = maximum_uncertainty_kmh
        self.maximum_reprojection_error_px = maximum_reprojection_error_px
        self.maximum_speed_mps = maximum_speed_mps
        self.minimum_candidate_confidence = minimum_candidate_confidence
        self.minimum_track_confidence = minimum_track_confidence
        self.history: Deque[BallKinematicsSample] = deque()
        self.active_key: Optional[tuple[Any, int]] = None
        self.motion_segment = 0
        self._motion_observations: Deque[
            tuple[float, np.ndarray, float]
        ] = deque(maxlen=2)

    def calculate_speed(
        self,
        tracks: dict[str, Any],
        timestamp_seconds: float,
        pose: CameraPoseResult,
    ) -> dict[str, Any]:
        balls = tracks.get("ball", {})
        for ball in balls.values():
            ball.pop("speed", None)
            ball.pop("velocity_mps", None)
            ball.pop("position_3d_m", None)
            self._apply_result(
                ball,
                BallSpeedResult(
                    state=BallSpeedState.WARMING_UP.value,
                    reason=BallSpeedReason.INSUFFICIENT_SAMPLES.value,
                ),
            )

        if not balls:
            self.reset()
            return tracks
        if len(balls) != 1:
            self.reset()
            for ball in balls.values():
                self._apply_result(
                    ball,
                    BallSpeedResult(
                        state=BallSpeedState.UNAVAILABLE.value,
                        reason=BallSpeedReason.TRACK_AMBIGUOUS.value,
                    ),
                )
            return tracks

        ball_id = next(iter(balls))
        ball = balls[ball_id]
        key = (ball_id, int(ball.get("track_segment", 0)))
        if key != self.active_key:
            self.history.clear()
            self._motion_observations.clear()
            self.motion_segment += 1
            self.active_key = key

        if pose.speed_failure_reason is not None:
            self.history.clear()
            self._apply_result(
                ball,
                BallSpeedResult(
                    state=BallSpeedState.UNAVAILABLE.value,
                    reason=pose.speed_failure_reason,
                ),
            )
            return tracks
        if not pose.has_metric_pose:
            self.history.clear()
            self._apply_result(
                ball,
                BallSpeedResult(
                    state=BallSpeedState.UNAVAILABLE.value,
                    reason=BallSpeedReason.NO_CAMERA_PROFILE.value,
                ),
            )
            return tracks
        if not ball.get("observed", False):
            self._apply_result(
                ball,
                BallSpeedResult(
                    state=BallSpeedState.UNAVAILABLE.value,
                    reason=BallSpeedReason.INSUFFICIENT_SAMPLES.value,
                ),
            )
            return tracks
        if not ball.get("track_confirmed", False):
            reason = (
                BallSpeedReason.TRACK_AMBIGUOUS.value
                if ball.get("track_state") == "ambiguous"
                else BallSpeedReason.TRACK_TENTATIVE.value
            )
            self._apply_result(
                ball,
                BallSpeedResult(state=BallSpeedState.WARMING_UP.value, reason=reason),
            )
            return tracks
        if (
            float(ball.get("confidence", 0.0)) < self.minimum_candidate_confidence
            or float(ball.get("track_confidence", 0.0)) < self.minimum_track_confidence
        ):
            self._apply_result(
                ball,
                BallSpeedResult(
                    state=BallSpeedState.UNAVAILABLE.value,
                    reason=BallSpeedReason.LOW_BALL_CONFIDENCE.value,
                ),
            )
            return tracks

        bbox = ball.get("bbox")
        if bbox is None:
            return tracks
        center = (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )
        bbox_size = max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
        sample = BallKinematicsSample(
            timestamp=float(timestamp_seconds),
            pixel=center,
            bbox_size_px=max(1.0, bbox_size),
            pose=pose,
            candidate_confidence=float(ball.get("confidence", 0.0)),
            track_confidence=float(ball.get("track_confidence", 0.0)),
        )
        motion_observation = self._motion_observation(sample)
        if motion_observation is not None and self._contact_event(motion_observation):
            self.history.clear()
            self._motion_observations.clear()
            self.motion_segment += 1
        if motion_observation is not None:
            self._motion_observations.append(motion_observation)
        if self.history and sample.timestamp - self.history[-1].timestamp > self.maximum_gap_seconds:
            self.history.clear()
        self.history.append(sample)
        while self.history and sample.timestamp - self.history[0].timestamp > self.history_window_seconds:
            self.history.popleft()

        result = self._estimate()
        self._apply_result(ball, result)
        return tracks

    def _motion_observation(
        self, sample: BallKinematicsSample
    ) -> Optional[tuple[float, np.ndarray, float]]:
        """Approximate ground motion and height for conservative contact splitting."""
        try:
            ground = self.camera_geometry.intersect_height_plane(
                sample.pixel,
                sample.pose,
                self.ball_radius_m,
            )
            origin, direction = self.camera_geometry.ray_from_pixel(
                sample.pixel,
                sample.pose,
            )
            rotation = cv2.Rodrigues(
                np.asarray(sample.pose.rotation_vector, dtype=np.float64)
            )[0]
            camera_direction = rotation @ direction
            expected_depth = (
                float(self.camera_profile.camera_matrix[0, 0])
                * self.ball_diameter_m
                / sample.bbox_size_px
            )
            ray_distance = expected_depth / max(camera_direction[2], 1e-6)
            apparent_height = float((origin + ray_distance * direction)[2])
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            return None
        if ground is None or not np.isfinite(apparent_height):
            return None
        return sample.timestamp, np.asarray(ground[:2], dtype=np.float64), apparent_height

    def _contact_event(
        self, current: tuple[float, np.ndarray, float]
    ) -> bool:
        if len(self._motion_observations) < 2:
            return False
        first, second = self._motion_observations
        current_time, current_point, current_height = current
        first_dt = second[0] - first[0]
        second_dt = current_time - second[0]
        if first_dt <= 1e-6 or second_dt <= 1e-6:
            return False
        # Only split near a plausible contact plane. Apparent size is deliberately
        # a weak depth cue and is never used by itself to emit a speed.
        if min(second[2], current_height) > self.ball_radius_m + 0.25:
            return False
        previous_velocity = (second[1] - first[1]) / first_dt
        current_velocity = (current_point - second[1]) / second_dt
        previous_speed = float(np.linalg.norm(previous_velocity))
        current_speed = float(np.linalg.norm(current_velocity))
        velocity_change = float(np.linalg.norm(current_velocity - previous_velocity))
        reversed_direction = False
        if previous_speed > 2.0 and current_speed > 2.0:
            cosine = float(
                np.dot(previous_velocity, current_velocity)
                / (previous_speed * current_speed)
            )
            reversed_direction = cosine < 0.0
        return reversed_direction or velocity_change > 15.0

    def _estimate(self) -> BallSpeedResult:
        samples = list(self.history)
        span = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0.0
        if len(samples) < self.minimum_ground_samples or span < self.minimum_ground_span_seconds:
            return BallSpeedResult(
                state=BallSpeedState.WARMING_UP.value,
                reason=BallSpeedReason.INSUFFICIENT_SAMPLES.value,
                sample_count=len(samples),
                sample_span_seconds=span,
            )
        ground = self._fit_ground(samples)
        air = None
        if len(samples) >= self.minimum_air_samples and span >= self.minimum_air_span_seconds:
            air = self._fit_air(samples)
        fit = ground
        if (
            air is not None
            and (
                ground is None
                or air.reprojection_error_px < ground.reprojection_error_px * 0.80
            )
            and air.position_m[2] > self.ball_radius_m + 0.30
        ):
            fit = air
        if fit is None:
            return BallSpeedResult(
                state=BallSpeedState.UNAVAILABLE.value,
                reason=BallSpeedReason.TRAJECTORY_UNOBSERVABLE.value,
                sample_count=len(samples),
                sample_span_seconds=span,
            )
        if fit.speed_kmh > self.maximum_speed_mps * 3.6:
            return self._result_from_fit(
                fit,
                BallSpeedState.UNAVAILABLE.value,
                BallSpeedReason.TRAJECTORY_UNOBSERVABLE.value,
                len(samples),
                span,
            )
        if fit.uncertainty_kmh > self.maximum_uncertainty_kmh:
            return self._result_from_fit(
                fit,
                BallSpeedState.UNAVAILABLE.value,
                BallSpeedReason.UNCERTAINTY_HIGH.value,
                len(samples),
                span,
            )
        if fit.reprojection_error_px > self.maximum_reprojection_error_px:
            return self._result_from_fit(
                fit,
                BallSpeedState.UNAVAILABLE.value,
                BallSpeedReason.TRAJECTORY_UNOBSERVABLE.value,
                len(samples),
                span,
            )
        return self._result_from_fit(
            fit,
            BallSpeedState.RELIABLE.value,
            None,
            len(samples),
            span,
        )

    def _fit_ground(self, samples: list[BallKinematicsSample]) -> Optional[_MotionFit]:
        points = []
        accepted_samples = []
        for sample in samples:
            try:
                point = self.camera_geometry.intersect_height_plane(
                    sample.pixel, sample.pose, self.ball_radius_m
                )
            except (ValueError, cv2.error):
                point = None
            if point is None or not self.geometry.contains(tuple(point[:2]), margin_m=2.0):
                continue
            points.append(point)
            accepted_samples.append(sample)
        if len(points) < self.minimum_ground_samples:
            return None
        values = np.asarray(points, dtype=np.float64)
        times = np.asarray([sample.timestamp for sample in accepted_samples], dtype=np.float64)
        times -= times[0]
        design = np.column_stack([times, np.ones_like(times)])
        parameters = np.linalg.lstsq(design, values, rcond=None)[0]
        velocity = parameters[0]
        intercept = parameters[1]
        predicted = intercept + times[:, None] * velocity
        world_residuals = np.linalg.norm(predicted - values, axis=1)
        inliers = world_residuals <= max(0.5, float(np.median(world_residuals)) * 3.0)
        if int(inliers.sum()) < self.minimum_ground_samples:
            return None
        if not inliers.all():
            parameters = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)[0]
            velocity, intercept = parameters
            predicted = intercept + times[:, None] * velocity
        reprojection = self._reprojection_errors(predicted, accepted_samples)
        pairwise = []
        for first in range(len(values) - 1):
            for second in range(first + 1, len(values)):
                dt = times[second] - times[first]
                if dt >= 0.10:
                    pairwise.append((values[second] - values[first]) / dt)
        if not pairwise:
            return None
        velocities = np.asarray(pairwise)
        speed_samples = np.linalg.norm(velocities, axis=1) * 3.6
        speed = float(np.linalg.norm(velocity) * 3.6)
        mad = float(np.median(np.abs(speed_samples - np.median(speed_samples))))
        uncertainty = min(99.0, 1.96 * 1.4826 * mad / np.sqrt(max(1, len(speed_samples))))
        return _MotionFit(
            mode="ground",
            speed_kmh=speed,
            velocity_mps=velocity,
            position_m=predicted[-1],
            uncertainty_kmh=uncertainty,
            reprojection_error_px=float(np.median(reprojection)),
        )

    def _fit_air(self, samples: list[BallKinematicsSample]) -> Optional[_MotionFit]:
        times = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
        times -= times[0]
        ground_points = []
        for sample in samples:
            try:
                point = self.camera_geometry.intersect_height_plane(
                    sample.pixel, sample.pose, self.ball_radius_m
                )
            except (ValueError, cv2.error):
                point = None
            if point is None:
                return None
            ground_points.append(point)
        ground_values = np.asarray(ground_points)
        design = np.column_stack([times, np.ones_like(times)])
        xy_parameters = np.linalg.lstsq(design, ground_values[:, :2], rcond=None)[0]
        initial = np.asarray(
            [
                xy_parameters[1, 0],
                xy_parameters[1, 1],
                1.0,
                xy_parameters[0, 0],
                xy_parameters[0, 1],
                2.0,
            ],
            dtype=np.float64,
        )
        gravity = np.asarray([0.0, 0.0, -9.81])

        def residual(parameters: np.ndarray) -> np.ndarray:
            position0 = parameters[:3]
            velocity0 = parameters[3:]
            residuals: list[float] = []
            for time_value, sample in zip(times, samples):
                point = position0 + velocity0 * time_value + 0.5 * gravity * time_value**2
                try:
                    projected = self.camera_geometry.project_world(point, sample.pose)
                except (ValueError, cv2.error):
                    return np.full(len(samples) * 3, 1e3)
                residuals.extend(((projected - np.asarray(sample.pixel)) / 2.0).tolist())
                camera_rotation = cv2.Rodrigues(
                    np.asarray(sample.pose.rotation_vector, dtype=np.float64)
                )[0]
                camera_translation = np.asarray(
                    sample.pose.translation_vector, dtype=np.float64
                ).reshape(3)
                camera_point = camera_rotation @ point + camera_translation
                if camera_point[2] > 0:
                    expected_size = (
                        float(self.camera_profile.camera_matrix[0, 0])
                        * self.ball_diameter_m
                        / camera_point[2]
                    )
                    residuals.append((expected_size - sample.bbox_size_px) / 10.0)
                else:
                    residuals.append(100.0)
                if point[2] < self.ball_radius_m:
                    residuals.append((self.ball_radius_m - point[2]) * 10.0)
                else:
                    residuals.append(0.0)
            return np.asarray(residuals, dtype=np.float64)

        lower = np.asarray([-10.0, -10.0, self.ball_radius_m, -60.0, -60.0, -40.0])
        upper = np.asarray(
            [self.geometry.length_m + 10.0, self.geometry.width_m + 10.0, 30.0, 60.0, 60.0, 40.0]
        )
        try:
            result = least_squares(
                residual,
                np.clip(initial, lower + 1e-6, upper - 1e-6),
                bounds=(lower, upper),
                loss="huber",
                f_scale=1.5,
                max_nfev=250,
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not result.success or not np.isfinite(result.x).all():
            return None
        position0, velocity0 = result.x[:3], result.x[3:]
        latest_time = times[-1]
        latest_position = position0 + velocity0 * latest_time + 0.5 * gravity * latest_time**2
        latest_velocity = velocity0 + gravity * latest_time
        predicted = np.asarray(
            [
                position0 + velocity0 * time_value + 0.5 * gravity * time_value**2
                for time_value in times
            ]
        )
        reprojection = self._reprojection_errors(predicted, samples)
        uncertainty = 99.0
        degrees = max(1, len(result.fun) - len(result.x))
        try:
            covariance = np.linalg.pinv(result.jac.T @ result.jac) * (
                float(np.sum(result.fun**2)) / degrees
            )
            speed_mps = float(np.linalg.norm(latest_velocity))
            if speed_mps > 1e-9:
                gradient = np.zeros(6)
                gradient[3:] = latest_velocity / speed_mps
                variance = float(gradient @ covariance @ gradient)
                uncertainty = 1.96 * np.sqrt(max(0.0, variance)) * 3.6
        except np.linalg.LinAlgError:
            pass
        return _MotionFit(
            mode="air",
            speed_kmh=float(np.linalg.norm(latest_velocity) * 3.6),
            velocity_mps=latest_velocity,
            position_m=latest_position,
            uncertainty_kmh=float(min(99.0, uncertainty)),
            reprojection_error_px=float(np.median(reprojection)),
        )

    def _reprojection_errors(
        self,
        positions: np.ndarray,
        samples: list[BallKinematicsSample],
    ) -> np.ndarray:
        errors = []
        for point, sample in zip(positions, samples):
            try:
                projected = self.camera_geometry.project_world(point, sample.pose)
                errors.append(float(np.linalg.norm(projected - np.asarray(sample.pixel))))
            except (ValueError, cv2.error):
                errors.append(float("inf"))
        return np.asarray(errors)

    @staticmethod
    def _result_from_fit(
        fit: _MotionFit,
        state: str,
        reason: Optional[str],
        sample_count: int,
        span: float,
    ) -> BallSpeedResult:
        return BallSpeedResult(
            state=state,
            reason=reason,
            speed_3d_kmh=fit.speed_kmh,
            velocity_mps=tuple(map(float, fit.velocity_mps)),
            motion_mode=fit.mode,
            uncertainty_kmh=fit.uncertainty_kmh,
            position_3d_m=tuple(map(float, fit.position_m)),
            sample_count=sample_count,
            sample_span_seconds=span,
            reprojection_error_px=fit.reprojection_error_px,
        )

    def _apply_result(self, ball: dict[str, Any], result: BallSpeedResult) -> None:
        ball["speed_state"] = result.state
        ball["speed_status"] = "reliable" if result.state == "reliable" else "pending"
        if result.reason is None:
            ball.pop("speed_reason", None)
        else:
            ball["speed_reason"] = result.reason
        ball["speed_sample_count"] = result.sample_count
        ball["speed_sample_span_seconds"] = result.sample_span_seconds
        if result.speed_3d_kmh is not None:
            ball["speed_3d_kmh"] = result.speed_3d_kmh
        if result.state == BallSpeedState.RELIABLE.value and result.speed_3d_kmh is not None:
            ball["speed"] = result.speed_3d_kmh
        if result.velocity_mps is not None:
            ball["velocity_mps"] = result.velocity_mps
        if result.motion_mode is not None:
            ball["motion_mode"] = result.motion_mode
        if result.uncertainty_kmh is not None:
            ball["speed_uncertainty_kmh"] = result.uncertainty_kmh
        if result.position_3d_m is not None:
            ball["position_3d_m"] = result.position_3d_m
        if result.reprojection_error_px is not None:
            ball["speed_reprojection_error_px"] = result.reprojection_error_px
        ball["speed_model"] = "monocular_physics_estimate"
        ball["motion_segment"] = self.motion_segment

    def reset(self) -> None:
        self.history.clear()
        self.active_key = None
        self.motion_segment = 0
        self._motion_observations.clear()
