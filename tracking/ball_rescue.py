from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import Any


@dataclass(frozen=True)
class BallRescueRoi:
    """An image-space ROI proposed without looking at ball ground truth."""

    bbox: tuple[float, float, float, float]
    source: str
    player_id: Any | None = None


def _clip_roi(
    bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...]
) -> tuple[float, float, float, float] | None:
    height, width = frame_shape[:2]
    x1 = max(0.0, min(float(width), bbox[0]))
    y1 = max(0.0, min(float(height), bbox[1]))
    x2 = max(0.0, min(float(width), bbox[2]))
    y2 = max(0.0, min(float(height), bbox[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _roi_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _players(player_tracks: dict) -> list[dict[str, Any]]:
    result = []
    for player_type in ("player", "goalkeeper"):
        for player_id, player in player_tracks.get(player_type, {}).items():
            x1, y1, x2, y2 = map(float, player["bbox"])
            height = max(1.0, y2 - y1)
            result.append(
                {
                    "id": player_id,
                    "type": player_type,
                    "foot": ((x1 + x2) / 2, y2),
                    "height": height,
                    "confidence": float(player.get("confidence", 0.0)),
                }
            )
    return result


def _player_priority(
    player: dict[str, Any],
    players: list[dict[str, Any]],
    predicted_center: tuple[float, float] | None,
) -> tuple[float, float, float]:
    foot_x, foot_y = player["foot"]
    scale = max(48.0, 2.5 * player["height"])
    if predicted_center is not None:
        primary = hypot(foot_x - predicted_center[0], foot_y - predicted_center[1]) / scale
    else:
        neighbour_distances = [
            hypot(foot_x - other["foot"][0], foot_y - other["foot"][1]) / scale
            for other in players
            if other is not player
        ]
        primary = min(neighbour_distances, default=float("inf"))
    return primary, -player["confidence"], -player["height"]


def build_ball_rescue_rois(
    player_tracks: dict,
    frame_shape: tuple[int, ...],
    predicted_center: tuple[float, float] | None = None,
    maximum_rois: int = 6,
    minimum_half_size: float = 48.0,
    horizontal_player_scale: float = 1.75,
    upward_player_scale: float = 1.50,
    downward_player_scale: float = 0.75,
    predicted_player_scale: float = 2.0,
    dedup_iou: float = 0.70,
) -> list[BallRescueRoi]:
    """Propose a small, deterministic set of ball-search regions.

    Ground truth is intentionally absent from the API.  A prior tracker center
    gets one free-ball ROI; remaining slots cover player feet, ranked by
    distance to that prediction or by local player density when no prediction
    exists.
    """
    if maximum_rois < 1:
        return []
    if minimum_half_size <= 0:
        raise ValueError("minimum_half_size must be positive")
    if not 0 <= dedup_iou <= 1:
        raise ValueError("dedup_iou must be between 0 and 1")

    players = _players(player_tracks)
    proposals: list[BallRescueRoi] = []
    if predicted_center is not None:
        player_height = median([player["height"] for player in players]) if players else 0.0
        half_size = max(minimum_half_size, predicted_player_scale * player_height)
        clipped = _clip_roi(
            (
                predicted_center[0] - half_size,
                predicted_center[1] - half_size,
                predicted_center[0] + half_size,
                predicted_center[1] + half_size,
            ),
            frame_shape,
        )
        if clipped is not None:
            proposals.append(BallRescueRoi(clipped, "prediction"))

    ordered_players = sorted(
        players,
        key=lambda player: _player_priority(player, players, predicted_center),
    )
    for player in ordered_players:
        foot_x, foot_y = player["foot"]
        height = player["height"]
        half_width = max(minimum_half_size, horizontal_player_scale * height)
        upward = max(minimum_half_size, upward_player_scale * height)
        downward = max(minimum_half_size * 0.67, downward_player_scale * height)
        clipped = _clip_roi(
            (
                foot_x - half_width,
                foot_y - upward,
                foot_x + half_width,
                foot_y + downward,
            ),
            frame_shape,
        )
        if clipped is None:
            continue
        proposals.append(
            BallRescueRoi(
                clipped,
                f"{player['type']}_foot",
                player_id=player["id"],
            )
        )

    kept: list[BallRescueRoi] = []
    for proposal in proposals:
        if any(_roi_iou(proposal.bbox, existing.bbox) > dedup_iou for existing in kept):
            continue
        kept.append(proposal)
        if len(kept) >= maximum_rois:
            break
    return kept


def point_in_rescue_roi(
    point: tuple[float, float], roi: BallRescueRoi
) -> bool:
    return (
        roi.bbox[0] <= point[0] <= roi.bbox[2]
        and roi.bbox[1] <= point[1] <= roi.bbox[3]
    )
