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
