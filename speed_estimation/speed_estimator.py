from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict

import numpy as np


class SpeedEstimator:
    """Estimate tracked-object speed from robust timestamped metric positions."""

    def __init__(
        self,
        track_types: tuple[str, ...] = ("player", "goalkeeper"),
        window_seconds: float = 0.5,
        minimum_span_seconds: float = 0.4,
        minimum_samples: int = 6,
        maximum_gap_seconds: float = 0.25,
        maximum_speed_kmh: float = 45.0,
        maximum_acceleration_mps2: float = 12.0,
        maximum_residual_m: float = 1.0,
        display_hold_seconds: float = 1.0,
    ) -> None:
        self.track_types = track_types
        self.window_seconds = window_seconds
        self.minimum_span_seconds = minimum_span_seconds
        self.minimum_samples = minimum_samples
        self.maximum_gap_seconds = maximum_gap_seconds
        self.maximum_speed_kmh = maximum_speed_kmh
        self.maximum_acceleration_mps2 = maximum_acceleration_mps2
        self.maximum_residual_m = maximum_residual_m
        self.display_hold_seconds = display_hold_seconds
        self.history: Dict[tuple[str, Any], Deque[tuple[float, float, float]]] = defaultdict(deque)
        self.last_speed: Dict[tuple[str, Any], tuple[float, float]] = {}
        self.display_speed: Dict[tuple[str, Any], tuple[float, float]] = {}

    def calculate_speed(
        self,
        tracks: Dict[str, Any],
        timestamp_seconds: float,
        projection_usable: bool = True,
    ) -> Dict[str, Any]:
        active_keys: set[tuple[str, Any]] = set()
        for track_type in self.track_types:
            for track_id, track in tracks.get(track_type, {}).items():
                track.pop("speed", None)
                track.pop("_display_speed", None)
                key = (track_type, track_id)
                active_keys.add(key)
                position = track.get("position_m")
                speed = None
                if projection_usable and position is not None:
                    speed = self._calculate_track_speed(key, position, timestamp_seconds)
                if speed is not None:
                    track["speed"] = speed
                    self.display_speed[key] = (timestamp_seconds, speed)
                else:
                    cached = self.display_speed.get(key)
                    if (
                        cached is not None
                        and timestamp_seconds - cached[0] <= self.display_hold_seconds
                    ):
                        track["_display_speed"] = cached[1]

        stale = [key for key in self.history if key not in active_keys]
        for key in stale:
            history = self.history[key]
            if history and timestamp_seconds - history[-1][0] > self.maximum_gap_seconds:
                del self.history[key]
                self.last_speed.pop(key, None)
        expired_display = [
            key
            for key, (updated_at, _speed) in self.display_speed.items()
            if timestamp_seconds - updated_at > self.display_hold_seconds
        ]
        for key in expired_display:
            self.display_speed.pop(key, None)
        return tracks

    def _calculate_track_speed(
        self,
        key: tuple[str, Any],
        position: tuple[float, float],
        timestamp_seconds: float,
    ) -> float | None:
        history = self.history[key]
        if history and timestamp_seconds - history[-1][0] > self.maximum_gap_seconds:
            history.clear()
            self.last_speed.pop(key, None)
        if history:
            elapsed = timestamp_seconds - history[-1][0]
            distance = float(
                np.linalg.norm(np.asarray(position) - np.asarray(history[-1][1:]))
            )
            instantaneous = distance / elapsed * 3.6 if elapsed > 0 else float("inf")
            if instantaneous > self.maximum_speed_kmh:
                history.clear()
                self.last_speed.pop(key, None)
        history.append((timestamp_seconds, float(position[0]), float(position[1])))
        while history and timestamp_seconds - history[0][0] > self.window_seconds:
            history.popleft()
        speed = self._fit_speed(history)
        if speed is None:
            return None
        previous = self.last_speed.get(key)
        if previous is not None:
            dt = timestamp_seconds - previous[0]
            acceleration = abs(speed - previous[1]) / 3.6 / dt if dt > 0 else float("inf")
            if acceleration > self.maximum_acceleration_mps2:
                history.clear()
                history.append((timestamp_seconds, float(position[0]), float(position[1])))
                self.last_speed.pop(key, None)
                return None
        self.last_speed[key] = (timestamp_seconds, speed)
        return speed

    def _fit_speed(
        self, history: Deque[tuple[float, float, float]]
    ) -> float | None:
        if len(history) < self.minimum_samples:
            return None
        values = np.asarray(history, dtype=np.float64)
        times = values[:, 0] - values[0, 0]
        if times[-1] < self.minimum_span_seconds:
            return None
        design = np.column_stack([times, np.ones_like(times)])
        velocity_x, intercept_x = np.linalg.lstsq(design, values[:, 1], rcond=None)[0]
        velocity_y, intercept_y = np.linalg.lstsq(design, values[:, 2], rcond=None)[0]
        predicted = np.column_stack(
            [velocity_x * times + intercept_x, velocity_y * times + intercept_y]
        )
        residuals = np.linalg.norm(predicted - values[:, 1:3], axis=1)
        inliers = residuals <= self.maximum_residual_m
        if int(inliers.sum()) < self.minimum_samples:
            return None
        if not inliers.all():
            inlier_design = design[inliers]
            velocity_x = np.linalg.lstsq(inlier_design, values[inliers, 1], rcond=None)[0][0]
            velocity_y = np.linalg.lstsq(inlier_design, values[inliers, 2], rcond=None)[0][0]
        speed = float(np.hypot(velocity_x, velocity_y) * 3.6)
        return speed if speed <= self.maximum_speed_kmh else None

    def reset(self) -> None:
        self.history.clear()
        self.last_speed.clear()
        self.display_speed.clear()
