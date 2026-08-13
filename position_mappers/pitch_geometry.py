from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchGeometry:
    """Metric football-pitch geometry matching the 32-keypoint model schema."""

    length_m: float
    width_m: float
    penalty_area_depth_m: float = 16.5
    penalty_area_width_m: float = 40.32
    goal_area_depth_m: float = 5.5
    goal_area_width_m: float = 18.32
    centre_circle_radius_m: float = 9.15
    penalty_spot_distance_m: float = 11.0

    def __post_init__(self) -> None:
        if self.length_m <= 2 * self.penalty_area_depth_m:
            raise ValueError("Pitch length is too small for a regulation penalty area")
        if self.width_m < self.penalty_area_width_m:
            raise ValueError("Pitch width is too small for a regulation penalty area")

    @property
    def vertices(self) -> np.ndarray:
        """Return the world coordinates for keypoint indexes 0 through 31."""
        length = self.length_m
        width = self.width_m
        penalty_top = (width - self.penalty_area_width_m) / 2
        penalty_bottom = (width + self.penalty_area_width_m) / 2
        goal_top = (width - self.goal_area_width_m) / 2
        goal_bottom = (width + self.goal_area_width_m) / 2
        centre_y = width / 2
        centre_x = length / 2

        return np.asarray(
            [
                (0, 0),
                (0, penalty_top),
                (0, goal_top),
                (0, goal_bottom),
                (0, penalty_bottom),
                (0, width),
                (self.goal_area_depth_m, goal_top),
                (self.goal_area_depth_m, goal_bottom),
                (self.penalty_spot_distance_m, centre_y),
                (self.penalty_area_depth_m, penalty_top),
                (self.penalty_area_depth_m, goal_top),
                (self.penalty_area_depth_m, goal_bottom),
                (self.penalty_area_depth_m, penalty_bottom),
                (centre_x, 0),
                (centre_x, centre_y - self.centre_circle_radius_m),
                (centre_x, centre_y + self.centre_circle_radius_m),
                (centre_x, width),
                (length - self.penalty_area_depth_m, penalty_top),
                (length - self.penalty_area_depth_m, goal_top),
                (length - self.penalty_area_depth_m, goal_bottom),
                (length - self.penalty_area_depth_m, penalty_bottom),
                (length - self.penalty_spot_distance_m, centre_y),
                (length - self.goal_area_depth_m, goal_top),
                (length - self.goal_area_depth_m, goal_bottom),
                (length, 0),
                (length, penalty_top),
                (length, goal_top),
                (length, goal_bottom),
                (length, penalty_bottom),
                (length, width),
                (centre_x - self.centre_circle_radius_m, centre_y),
                (centre_x + self.centre_circle_radius_m, centre_y),
            ],
            dtype=np.float32,
        )

    def contains(self, point: tuple[float, float], margin_m: float = 0.5) -> bool:
        x, y = point
        return (
            np.isfinite(x)
            and np.isfinite(y)
            and -margin_m <= x <= self.length_m + margin_m
            and -margin_m <= y <= self.width_m + margin_m
        )

    def to_display(
        self, point: tuple[float, float], width_px: int, height_px: int
    ) -> tuple[float, float]:
        x, y = point
        return x / self.length_m * (width_px - 1), y / self.width_m * (height_px - 1)

    def distance_to_marking(self, point: tuple[float, float]) -> float:
        """Return metric distance to the nearest known pitch marking or fixed spot."""
        x, y = map(float, point)
        length = self.length_m
        width = self.width_m
        penalty_top = (width - self.penalty_area_width_m) / 2
        penalty_bottom = (width + self.penalty_area_width_m) / 2
        goal_top = (width - self.goal_area_width_m) / 2
        goal_bottom = (width + self.goal_area_width_m) / 2
        distances = [
            abs(x),
            abs(x - length),
            abs(y),
            abs(y - width),
            abs(x - length / 2),
            abs(np.hypot(x - length / 2, y - width / 2) - self.centre_circle_radius_m),
            np.hypot(x - self.penalty_spot_distance_m, y - width / 2),
            np.hypot(x - (length - self.penalty_spot_distance_m), y - width / 2),
        ]
        if penalty_top <= y <= penalty_bottom:
            distances.extend(
                [
                    abs(x - self.penalty_area_depth_m),
                    abs(x - (length - self.penalty_area_depth_m)),
                ]
            )
        if goal_top <= y <= goal_bottom:
            distances.extend(
                [
                    abs(x - self.goal_area_depth_m),
                    abs(x - (length - self.goal_area_depth_m)),
                ]
            )
        if 0 <= x <= self.penalty_area_depth_m or length - self.penalty_area_depth_m <= x <= length:
            distances.extend([abs(y - penalty_top), abs(y - penalty_bottom)])
        if 0 <= x <= self.goal_area_depth_m or length - self.goal_area_depth_m <= x <= length:
            distances.extend([abs(y - goal_top), abs(y - goal_bottom)])
        return float(min(distances))

    def marking_sample_points(self, samples_per_line: int = 40) -> np.ndarray:
        """Sample regulation pitch markings for image-space alignment checks."""
        x_values = np.linspace(0.0, self.length_m, samples_per_line)
        y_values = np.linspace(0.0, self.width_m, samples_per_line)
        points = [
            np.column_stack([x_values, np.zeros_like(x_values)]),
            np.column_stack([x_values, np.full_like(x_values, self.width_m)]),
            np.column_stack([np.zeros_like(y_values), y_values]),
            np.column_stack([np.full_like(y_values, self.length_m), y_values]),
            np.column_stack([np.full_like(y_values, self.length_m / 2), y_values]),
        ]
        angles = np.linspace(0.0, 2.0 * np.pi, samples_per_line, endpoint=False)
        points.append(
            np.column_stack(
                [
                    self.length_m / 2 + self.centre_circle_radius_m * np.cos(angles),
                    self.width_m / 2 + self.centre_circle_radius_m * np.sin(angles),
                ]
            )
        )
        penalty_top = (self.width_m - self.penalty_area_width_m) / 2
        penalty_bottom = (self.width_m + self.penalty_area_width_m) / 2
        for x in (self.penalty_area_depth_m, self.length_m - self.penalty_area_depth_m):
            penalty_y = np.linspace(penalty_top, penalty_bottom, samples_per_line)
            points.append(np.column_stack([np.full_like(penalty_y, x), penalty_y]))
        return np.concatenate(points, axis=0).astype(np.float32)
