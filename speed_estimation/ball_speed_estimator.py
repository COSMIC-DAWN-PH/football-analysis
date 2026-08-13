from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Deque, Optional

import cv2
import numpy as np

from position_mappers import CalibrationResult


@dataclass(frozen=True)
class BallSpeedSample:
    timestamp: float
    position_x: float
    position_y: float
    candidate_confidence: float
    track_confidence: float


class BallSpeedEstimator:
    """Emit ball speed only for stable observed metric track segments."""

    def __init__(
        self,
        minimum_samples: int = 8,
        minimum_span_seconds: float = 0.25,
        maximum_gap_seconds: float = 0.12,
        history_window_seconds: float = 0.50,
        minimum_candidate_confidence: float = 0.20,
        minimum_track_confidence: float = 0.25,
        minimum_calibration_quality: float = 0.35,
        maximum_calibration_age_seconds: float = 0.25,
        maximum_fit_residual_m: float = 0.75,
        minimum_inlier_ratio: float = 0.75,
        minimum_direction_consistency: float = 0.75,
        maximum_calibration_shift_m: float = 1.5,
    ) -> None:
        self.minimum_samples = minimum_samples
        self.minimum_span_seconds = minimum_span_seconds
        self.maximum_gap_seconds = maximum_gap_seconds
        self.history_window_seconds = history_window_seconds
        self.minimum_candidate_confidence = minimum_candidate_confidence
        self.minimum_track_confidence = minimum_track_confidence
        self.minimum_calibration_quality = minimum_calibration_quality
        self.maximum_calibration_age_seconds = maximum_calibration_age_seconds
        self.maximum_fit_residual_m = maximum_fit_residual_m
        self.minimum_inlier_ratio = minimum_inlier_ratio
        self.minimum_direction_consistency = minimum_direction_consistency
        self.maximum_calibration_shift_m = maximum_calibration_shift_m
        self.history: Deque[BallSpeedSample] = deque()
        self.active_key: Optional[tuple[Any, int]] = None
        self.previous_homography: Optional[np.ndarray] = None

    def calculate_speed(
        self,
        tracks: dict[str, Any],
        timestamp_seconds: float,
        calibration: CalibrationResult,
    ) -> dict[str, Any]:
        balls = tracks.get("ball", {})
        for track in balls.values():
            track.pop("speed", None)
            track.pop("_display_speed", None)
            track["speed_status"] = "pending"

        if not calibration.valid:
            self.reset()
            return tracks

        current_homography = calibration.image_to_pitch
        assert current_homography is not None
        if self._calibration_shifted(
            balls,
            current_homography,
            calibration.status,
        ):
            self._clear_history()
        self.previous_homography = current_homography.copy()

        if not self._calibration_usable(calibration):
            self._clear_history()
            return tracks

        active_ball_ids = list(balls)
        if len(active_ball_ids) != 1:
            self._clear_history()
            return tracks

        ball_id = active_ball_ids[0]
        track = balls[ball_id]
        segment = int(track.get("track_segment", 0))
        key = (ball_id, segment)
        if key != self.active_key:
            self._clear_history()
            self.active_key = key

        if not track.get("observed", False):
            return tracks
        position = track.get("position_m")
        if position is None:
            return tracks

        sample = BallSpeedSample(
            timestamp=float(timestamp_seconds),
            position_x=float(position[0]),
            position_y=float(position[1]),
            candidate_confidence=float(track.get("confidence", 0.0)),
            track_confidence=float(track.get("track_confidence", 0.0)),
        )
        if (
            self.history
            and sample.timestamp - self.history[-1].timestamp > self.maximum_gap_seconds
        ):
            self._clear_history()
            self.active_key = key
        self.history.append(sample)
        while (
            self.history
            and sample.timestamp - self.history[0].timestamp > self.history_window_seconds
        ):
            self.history.popleft()

        if not track.get("track_confirmed", False):
            return tracks
        speed = self._trusted_speed()
        if speed is not None:
            track["speed"] = speed
            track["speed_status"] = "reliable"
        return tracks

    def _calibration_usable(self, calibration: CalibrationResult) -> bool:
        return (
            calibration.valid
            and calibration.age_seconds is not None
            and calibration.age_seconds <= self.maximum_calibration_age_seconds
            and calibration.quality >= self.minimum_calibration_quality
        )

    def _calibration_shifted(
        self,
        balls: dict[Any, dict],
        current_homography: np.ndarray,
        calibration_status: str,
    ) -> bool:
        direct_statuses = {"detected", "fused", "detected_reset"}
        if (
            calibration_status not in direct_statuses
            or self.previous_homography is None
            or len(balls) != 1
        ):
            return False
        track = next(iter(balls.values()))
        bbox = track.get("bbox")
        if bbox is None:
            return False
        point = np.asarray(
            [[[(float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2]]],
            dtype=np.float32,
        )
        previous_position = cv2.perspectiveTransform(
            point, self.previous_homography
        ).reshape(2)
        current_position = cv2.perspectiveTransform(
            point, current_homography
        ).reshape(2)
        if not np.isfinite(previous_position).all() or not np.isfinite(current_position).all():
            return True
        return bool(
            np.linalg.norm(previous_position - current_position)
            > self.maximum_calibration_shift_m
        )

    def _trusted_speed(self) -> Optional[float]:
        if len(self.history) < self.minimum_samples:
            return None
        samples = list(self.history)
        span = samples[-1].timestamp - samples[0].timestamp
        sample_intervals = np.diff([sample.timestamp for sample in samples])
        covered_duration = span + (
            float(np.median(sample_intervals)) if len(sample_intervals) else 0.0
        )
        if covered_duration < self.minimum_span_seconds:
            return None
        if (
            median(sample.candidate_confidence for sample in samples)
            < self.minimum_candidate_confidence
        ):
            return None
        if samples[-1].track_confidence < self.minimum_track_confidence:
            return None

        values = np.asarray(
            [
                (sample.timestamp, sample.position_x, sample.position_y)
                for sample in samples
            ],
            dtype=np.float64,
        )
        times = values[:, 0] - values[0, 0]
        velocities = []
        for first in range(len(values) - 1):
            for second in range(first + 1, len(values)):
                elapsed = times[second] - times[first]
                if elapsed >= 0.10:
                    velocities.append(
                        (values[second, 1:] - values[first, 1:]) / elapsed
                    )
        if not velocities:
            return None
        velocity = np.median(np.asarray(velocities), axis=0)
        intercept = np.median(values[:, 1:] - times[:, None] * velocity, axis=0)
        predicted = intercept + times[:, None] * velocity
        residuals = np.linalg.norm(predicted - values[:, 1:], axis=1)
        inliers = residuals <= self.maximum_fit_residual_m
        if float(np.mean(inliers)) < self.minimum_inlier_ratio:
            return None
        if float(np.median(residuals)) > self.maximum_fit_residual_m:
            return None

        displacements = np.diff(values[:, 1:], axis=0)
        meaningful = np.linalg.norm(displacements, axis=1) >= 0.05
        speed_mps = float(np.linalg.norm(velocity))
        if speed_mps > 1e-6 and meaningful.any():
            aligned = (displacements[meaningful] @ velocity) >= 0.0
            if float(np.mean(aligned)) < self.minimum_direction_consistency:
                return None
        return speed_mps * 3.6

    def _clear_history(self) -> None:
        self.history.clear()
        self.active_key = None

    def reset(self) -> None:
        self._clear_history()
        self.previous_homography = None
