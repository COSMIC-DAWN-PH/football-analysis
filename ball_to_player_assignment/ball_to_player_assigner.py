from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from club_assignment import Club
from utils import point_distance

from .possession_tracking import PossessionTracker


class BallToPlayerAssigner:
    """Assign a metric ball track to the nearest visible player."""

    def __init__(
        self,
        club1: Club,
        club2: Club,
        max_ball_distance: float = 2.0,
        grace_period: float = 4.0,
        ball_grace_period: float = 2.0,
        fps: int = 30,
        minimum_track_confidence: float = 0.10,
        **_legacy_options: Any,
    ) -> None:
        self.max_ball_distance = max_ball_distance
        self.grace_period = grace_period
        self.ball_grace_period = ball_grace_period
        self.fps = max(fps, 1)
        self.minimum_track_confidence = minimum_track_confidence
        self.possession_tracker = PossessionTracker(club1, club2)
        self.last_player_key: Optional[tuple[str, Any]] = None
        self.last_possessing_team: str | int = -1
        self.last_possession_timestamp: Optional[float] = None
        self.last_observed_ball_timestamp: Optional[float] = None

    def assign(
        self,
        tracks: Dict[str, Any],
        current_frame: int,
        penalty_point_1_pos: Optional[Tuple[float, float]] = None,
        penalty_point_2_pos: Optional[Tuple[float, float]] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> tuple[Dict[str, Any], int]:
        del penalty_point_1_pos, penalty_point_2_pos
        timestamp = (
            float(timestamp_seconds)
            if timestamp_seconds is not None
            else current_frame / self.fps
        )
        for player_type in ("player", "goalkeeper"):
            for player in tracks.get(player_type, {}).values():
                player.pop("has_ball", None)

        credible_balls = []
        for ball_id, ball in tracks.get("ball", {}).items():
            position = ball.get("position_m")
            confidence = float(ball.get("track_confidence", ball.get("confidence", 0.0)))
            if position is None or confidence < self.minimum_track_confidence:
                continue
            credible_balls.append((ball_id, ball, position, confidence))
            if ball.get("observed", True):
                self.last_observed_ball_timestamp = timestamp

        players: list[tuple[str, Any, dict]] = []
        for player_type in ("player", "goalkeeper"):
            players.extend(
                (player_type, player_id, player)
                for player_id, player in tracks.get(player_type, {}).items()
                if player.get("position_m") is not None
            )

        selected_player: Optional[tuple[str, Any, dict]] = None
        selected_ball_id: Any = None
        minimum_distance = self.max_ball_distance
        for ball_id, _ball, ball_position, _confidence in sorted(
            credible_balls, key=lambda item: item[3], reverse=True
        ):
            for player_type, player_id, player in players:
                distance = point_distance(ball_position, player["position_m"])
                if distance <= minimum_distance:
                    minimum_distance = distance
                    selected_player = (player_type, player_id, player)
                    selected_ball_id = ball_id

        if selected_ball_id is not None:
            for ball_id in list(tracks.get("ball", {})):
                if ball_id != selected_ball_id:
                    del tracks["ball"][ball_id]

        if selected_player is not None and "club" in selected_player[2]:
            player_type, player_id, player = selected_player
            player["has_ball"] = True
            self.last_player_key = (player_type, player_id)
            self.last_possessing_team = player["club"]
            self.last_possession_timestamp = timestamp
            self.possession_tracker.add_possession(player["club"])
            return tracks, int(player_id)

        grace = self.grace_period if credible_balls else self.ball_grace_period
        if (
            self.last_player_key is not None
            and self.last_possession_timestamp is not None
            and timestamp - self.last_possession_timestamp <= grace
        ):
            player_type, player_id = self.last_player_key
            player = tracks.get(player_type, {}).get(player_id)
            if player is not None:
                player["has_ball"] = True
                self.possession_tracker.add_possession(self.last_possessing_team)
                return tracks, int(player_id)

        self.possession_tracker.add_possession(-1)
        if (
            self.last_possession_timestamp is not None
            and timestamp - self.last_possession_timestamp > self.grace_period
        ):
            self.last_player_key = None
            self.last_possessing_team = -1
        return tracks, -1

    def get_ball_possessions(self) -> Any:
        return self.possession_tracker.possession
