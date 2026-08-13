from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any, Optional

from position_mappers import CameraPoseResult


class BallSpeedState(StrEnum):
    WARMING_UP = "warming_up"
    RELIABLE = "reliable"
    UNAVAILABLE = "unavailable"


class BallSpeedReason(StrEnum):
    POSE_INVALID = "pose_invalid"
    POSE_STALE = "pose_stale"
    TRACK_TENTATIVE = "track_tentative"
    LOW_BALL_CONFIDENCE = "low_ball_confidence"
    TRACK_AMBIGUOUS = "track_ambiguous"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    TRAJECTORY_UNOBSERVABLE = "trajectory_unobservable"
    UNCERTAINTY_HIGH = "uncertainty_high"
    ZOOM_CHANGED = "zoom_changed"
    NO_CAMERA_PROFILE = "no_camera_profile"
    NO_BALL = "no_ball"


class RunDiagnostics:
    """Accumulate per-frame evidence and a compact run-quality summary."""

    def __init__(self) -> None:
        self.frames = 0
        self.pose_valid_frames = 0
        self.pose_speed_usable_frames = 0
        self.ball_frames = 0
        self.observed_ball_frames = 0
        self.confirmed_ball_frames = 0
        self.reliable_speed_frames = 0
        self.candidate_total = 0
        self.ambiguous_frames = 0
        self.reason_counts: Counter[str] = Counter()
        self.pose_status_counts: Counter[str] = Counter()
        self.track_segments: set[str] = set()

    def record_frame(
        self,
        timestamp_seconds: float,
        pose: CameraPoseResult,
        balls: dict[Any, dict[str, Any]],
        candidate_count: int,
        *,
        source_frame: Optional[int] = None,
        processed_frame: Optional[int] = None,
    ) -> dict[str, Any]:
        self.frames += 1
        self.pose_valid_frames += int(pose.valid)
        self.pose_speed_usable_frames += int(pose.speed_usable)
        self.candidate_total += int(candidate_count)
        self.pose_status_counts[pose.status] += 1

        frame: dict[str, Any] = {
            "timestamp_seconds": float(timestamp_seconds),
            "source_frame": source_frame,
            "processed_frame": processed_frame,
            "candidate_count": int(candidate_count),
            "pose_status": pose.status,
            "pose_valid": pose.valid,
            "pose_speed_usable": pose.speed_usable,
            "pose_quality": float(pose.quality),
            "pose_age_seconds": pose.age_seconds,
            "pose_failure_reason": pose.speed_failure_reason,
            "ball_count": len(balls),
        }
        if not balls:
            self.reason_counts[BallSpeedReason.NO_BALL.value] += 1
            frame.update(
                ball_speed_state=BallSpeedState.UNAVAILABLE.value,
                ball_speed_reason=BallSpeedReason.NO_BALL.value,
            )
            return frame

        self.ball_frames += 1
        selected = max(
            balls.items(),
            key=lambda item: float(item[1].get("selected_score", item[1].get("track_confidence", 0.0))),
        )
        ball_id, ball = selected
        observed = bool(ball.get("observed", False))
        confirmed = bool(ball.get("track_confirmed", False))
        state = str(ball.get("speed_state", BallSpeedState.WARMING_UP.value))
        reason = ball.get("speed_reason")
        self.observed_ball_frames += int(observed)
        self.confirmed_ball_frames += int(confirmed)
        self.reliable_speed_frames += int(state == BallSpeedState.RELIABLE.value)
        self.ambiguous_frames += int(reason == BallSpeedReason.TRACK_AMBIGUOUS.value)
        if reason:
            self.reason_counts[str(reason)] += 1
        segment = f"{ball_id}:{ball.get('track_segment', 0)}"
        self.track_segments.add(segment)
        frame.update(
            ball_id=ball_id,
            ball_observed=observed,
            ball_confirmed=confirmed,
            ball_track_segment=ball.get("track_segment"),
            ball_track_length=int(ball.get("track_length", 0)),
            ball_candidate_confidence=float(ball.get("confidence", 0.0)),
            ball_track_confidence=float(ball.get("track_confidence", 0.0)),
            ball_selected_score=float(ball.get("selected_score", 0.0)),
            ball_speed_state=state,
            ball_speed_reason=reason,
            ball_speed_kmh=ball.get("speed"),
            ball_speed_uncertainty_kmh=ball.get("speed_uncertainty_kmh"),
            ball_motion_mode=ball.get("motion_mode"),
        )
        return frame

    def summary(self) -> dict[str, Any]:
        denominator = max(1, self.frames)
        return {
            "schema_version": 1,
            "frames": self.frames,
            "pose_valid_frames": self.pose_valid_frames,
            "pose_valid_ratio": self.pose_valid_frames / denominator,
            "pose_speed_usable_frames": self.pose_speed_usable_frames,
            "pose_speed_usable_ratio": self.pose_speed_usable_frames / denominator,
            "ball_frames": self.ball_frames,
            "observed_ball_frames": self.observed_ball_frames,
            "confirmed_ball_frames": self.confirmed_ball_frames,
            "reliable_speed_frames": self.reliable_speed_frames,
            "reliable_speed_ratio": self.reliable_speed_frames / denominator,
            "candidate_total": self.candidate_total,
            "candidates_per_frame": self.candidate_total / denominator,
            "track_segments": len(self.track_segments),
            "ambiguous_frames": self.ambiguous_frames,
            "reason_counts": dict(self.reason_counts),
            "pose_status_counts": dict(self.pose_status_counts),
        }
