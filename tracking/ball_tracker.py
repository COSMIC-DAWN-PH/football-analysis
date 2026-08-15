from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, hypot
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from position_mappers import CalibrationResult, PitchGeometry
from tracking.abstract_tracker import resolve_inference_device


@dataclass(frozen=True)
class BallCandidate:
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


def overlapping_tiles(
    frame: np.ndarray, rows: int = 2, columns: int = 2, overlap: float = 0.20
) -> list[tuple[np.ndarray, int, int]]:
    """Split a frame into overlapping crops while guaranteeing full coverage."""
    if not 0 <= overlap < 1:
        raise ValueError("Tile overlap must be between 0 and 1")
    height, width = frame.shape[:2]
    tile_width = min(width, ceil(width / max(1.0, columns - overlap * (columns - 1))))
    tile_height = min(height, ceil(height / max(1.0, rows - overlap * (rows - 1))))
    x_positions = (
        [0]
        if columns == 1
        else [round(index * (width - tile_width) / (columns - 1)) for index in range(columns)]
    )
    y_positions = (
        [0]
        if rows == 1
        else [round(index * (height - tile_height) / (rows - 1)) for index in range(rows)]
    )
    return [
        (frame[y : y + tile_height, x : x + tile_width], x, y)
        for y in y_positions
        for x in x_positions
    ]


def non_max_suppression(
    candidates: Iterable[BallCandidate], iou_threshold: float = 0.50
) -> list[BallCandidate]:
    ordered = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
    kept: list[BallCandidate] = []
    for candidate in ordered:
        if all(_bbox_iou(candidate.bbox, existing.bbox) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _bbox_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def offset_bbox(
    bbox: Iterable[float], offset_x: int, offset_y: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y


class BallDetector:
    """High-recall ball detector using whole-frame and tiled inference."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.02,
        global_imgsz: int = 1920,
        tile_imgsz: int = 1280,
        overlap: float = 0.20,
        device: str = "auto",
    ) -> None:
        path = Path(model_path)
        self.device = resolve_inference_device(
            model_path, requested=device, priority=("GPU", "NPU", "CPU")
        )
        self.model = YOLO(model_path, task="detect")
        if path.is_file():
            self.model.to(torch.device(self.device))
        fixed_size = self._fixed_export_size(path)
        self.global_imgsz = fixed_size or global_imgsz
        self.tile_imgsz = fixed_size or tile_imgsz
        self.confidence = confidence
        self.overlap = overlap

    @staticmethod
    def _fixed_export_size(path: Path) -> int | None:
        metadata_path = path / "metadata.yaml"
        if not metadata_path.is_file():
            return None
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}
            if bool((metadata.get("args") or {}).get("dynamic")):
                return None
            image_size = metadata.get("imgsz")
            if isinstance(image_size, (list, tuple)):
                return int(max(image_size))
            return int(image_size) if image_size else None
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            return None

    def detect(self, frames: List[np.ndarray]) -> list[list[BallCandidate]]:
        return [self._detect_frame(frame) for frame in frames]

    def _detect_frame(self, frame: np.ndarray) -> list[BallCandidate]:
        candidates: list[BallCandidate] = []
        candidates.extend(self._predict(frame, 0, 0, self.global_imgsz))
        for tile, offset_x, offset_y in overlapping_tiles(frame, overlap=self.overlap):
            candidates.extend(self._predict(tile, offset_x, offset_y, self.tile_imgsz))
        return non_max_suppression(candidates)

    def _predict(
        self, frame: np.ndarray, offset_x: int, offset_y: int, image_size: int
    ) -> list[BallCandidate]:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=image_size,
            verbose=False,
            device=self.device,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names
        candidates: list[BallCandidate] = []
        for bbox, class_id, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            class_index = int(class_id)
            class_name = (
                names.get(class_index, "")
                if isinstance(names, dict)
                else names[class_index] if class_index < len(names) else ""
            )
            if str(class_name).lower() != "ball" and class_index != 0:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0 or not 0.25 <= width / height <= 4.0:
                continue
            candidates.append(
                BallCandidate(
                    bbox=offset_bbox((x1, y1, x2, y2), offset_x, offset_y),
                    confidence=float(confidence),
                )
            )
        return candidates


class ConstantVelocityKalman:
    """Small timestamp-aware 2D Kalman filter with velocity in its state."""

    def __init__(self, process_variance: float, measurement_variance: float) -> None:
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.timestamp: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def predict(self, timestamp: float) -> Optional[np.ndarray]:
        if self.state is None or self.covariance is None or self.timestamp is None:
            return None
        dt = max(0.0, timestamp - self.timestamp)
        if dt > 0:
            transition = np.asarray(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            dt2, dt3, dt4 = dt * dt, dt**3, dt**4
            process = self.process_variance * np.asarray(
                [
                    [dt4 / 4, 0, dt3 / 2, 0],
                    [0, dt4 / 4, 0, dt3 / 2],
                    [dt3 / 2, 0, dt2, 0],
                    [0, dt3 / 2, 0, dt2],
                ],
                dtype=np.float64,
            )
            self.state = transition @ self.state
            self.covariance = transition @ self.covariance @ transition.T + process
            self.timestamp = timestamp
        return self.state[:2].copy()

    def correct(self, measurement: np.ndarray, timestamp: float) -> np.ndarray:
        measurement = np.asarray(measurement, dtype=np.float64).reshape(2)
        if self.state is None:
            self.state = np.asarray(
                [measurement[0], measurement[1], 0.0, 0.0], dtype=np.float64
            )
            self.covariance = np.diag(
                [self.measurement_variance, self.measurement_variance, 1000.0, 1000.0]
            )
            self.timestamp = timestamp
            return measurement.copy()

        self.predict(timestamp)
        assert self.covariance is not None
        observation = np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64
        )
        noise = np.eye(2, dtype=np.float64) * self.measurement_variance
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
        self.state = self.state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        joseph = identity - gain @ observation
        self.covariance = (
            joseph @ self.covariance @ joseph.T + gain @ noise @ gain.T
        )
        return self.state[:2].copy()


class BallTracker:
    """Select and bridge ball observations with metric camera compensation."""

    def __init__(
        self,
        geometry: PitchGeometry,
        maximum_prediction_seconds: float = 0.5,
        restart_gap_seconds: float = 0.15,
        confirmations_required: int = 3,
    ) -> None:
        self.geometry = geometry
        self.maximum_prediction_seconds = maximum_prediction_seconds
        self.restart_gap_seconds = restart_gap_seconds
        self.confirmations_required = confirmations_required
        self.track_segment = 0
        self._reset_motion_state()

    def _reset_motion_state(self) -> None:
        self.image_filter = ConstantVelocityKalman(400.0, 16.0)
        self.metric_filter = ConstantVelocityKalman(16.0, 0.25)
        self.last_bbox_size = np.asarray([8.0, 8.0], dtype=np.float64)
        self.last_observed_timestamp: Optional[float] = None
        self.track_confidence = 0.0
        self.confirmation_count = 0
        self.segment_confirmed = False

    def update(
        self,
        candidates: list[BallCandidate],
        timestamp_seconds: float,
        calibration: CalibrationResult,
        player_tracks: dict,
        frame_shape: tuple[int, ...],
    ) -> dict[int, dict]:
        predicted_center, predicted_position = self._prediction(
            timestamp_seconds, calibration
        )
        valid_candidates = self._metric_candidates(candidates, calibration)
        observation_gap = (
            None
            if self.last_observed_timestamp is None
            else timestamp_seconds - self.last_observed_timestamp
        )
        continuing_segment = (
            observation_gap is not None
            and observation_gap <= self.restart_gap_seconds
        )
        selected = self._select_candidate(
            valid_candidates,
            predicted_center,
            predicted_position,
            player_tracks,
            frame_shape,
            observation_gap,
            enforce_motion_gate=continuing_segment,
        )
        if selected is not None and continuing_segment:
            candidate, position_m = selected
            return self._accept_observation(candidate, position_m, timestamp_seconds)
        if valid_candidates:
            restart = self._select_candidate(
                valid_candidates,
                predicted_center,
                predicted_position,
                player_tracks,
                frame_shape,
                observation_gap,
                enforce_motion_gate=False,
            )
            if restart is not None:
                candidate, position_m = restart
                return self._start_segment(candidate, position_m, timestamp_seconds)
        return self._predicted_track(
            predicted_center, predicted_position, timestamp_seconds, calibration
        )

    def _prediction(
        self, timestamp: float, calibration: CalibrationResult
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.image_filter.initialized:
            return None, None
        predicted_center = self.image_filter.predict(timestamp)
        predicted_position: Optional[np.ndarray] = None
        metric_prediction = self.metric_filter.predict(timestamp)
        if metric_prediction is not None and calibration.image_to_pitch is not None:
            if self.geometry.contains(tuple(metric_prediction), margin_m=0.0):
                predicted_position = metric_prediction
                try:
                    pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
                    predicted_center = cv2.perspectiveTransform(
                        predicted_position.astype(np.float32).reshape(1, 1, 2),
                        pitch_to_image,
                    ).reshape(2)
                except np.linalg.LinAlgError:
                    pass
        return predicted_center, predicted_position

    def _metric_candidates(
        self, candidates: list[BallCandidate], calibration: CalibrationResult
    ) -> list[tuple[BallCandidate, Optional[np.ndarray]]]:
        if calibration.image_to_pitch is None:
            return [(candidate, None) for candidate in candidates]
        result: list[tuple[BallCandidate, Optional[np.ndarray]]] = []
        for candidate in candidates:
            point = np.asarray(candidate.center, dtype=np.float32).reshape(1, 1, 2)
            position = cv2.perspectiveTransform(point, calibration.image_to_pitch).reshape(2)
            if self.geometry.contains(tuple(position), margin_m=0.0):
                result.append((candidate, position.astype(np.float64)))
        return result

    def _select_candidate(
        self,
        candidates: list[tuple[BallCandidate, Optional[np.ndarray]]],
        predicted_center: Optional[np.ndarray],
        predicted_position: Optional[np.ndarray],
        players: dict,
        frame_shape: tuple[int, ...],
        observation_gap: Optional[float],
        enforce_motion_gate: bool,
    ) -> Optional[tuple[BallCandidate, Optional[np.ndarray]]]:
        if not candidates:
            return None
        diagonal = hypot(frame_shape[1], frame_shape[0])
        elapsed = max(float(observation_gap or 0.0), 1.0 / 30.0)
        maximum_metric_distance = min(4.0, 0.75 + 40.0 * elapsed)
        frame_intervals = max(1.0, elapsed * 30.0)
        maximum_image_distance = min(
            diagonal * 0.08,
            diagonal * 0.03 * frame_intervals,
        )
        player_feet = []
        for player_type in ("player", "goalkeeper"):
            for player in players.get(player_type, {}).values():
                x1, _y1, x2, y2 = player["bbox"]
                player_feet.append(np.asarray([(x1 + x2) / 2, y2]))

        best: Optional[tuple[BallCandidate, Optional[np.ndarray]]] = None
        best_score = -float("inf")
        for candidate, position in candidates:
            center = np.asarray(candidate.center)
            bbox_width = candidate.bbox[2] - candidate.bbox[0]
            bbox_height = candidate.bbox[3] - candidate.bbox[1]
            maximum_size = max(48.0, min(frame_shape[:2]) * 0.05)
            if max(bbox_width, bbox_height) > maximum_size:
                continue
            score = candidate.confidence
            if predicted_position is not None and position is not None:
                distance = float(np.linalg.norm(position - predicted_position))
                if enforce_motion_gate and distance > maximum_metric_distance:
                    continue
                score -= 0.35 * min(distance / maximum_metric_distance, 2.0)
            elif predicted_center is not None:
                distance = float(np.linalg.norm(center - predicted_center))
                if enforce_motion_gate and distance > maximum_image_distance:
                    continue
                score -= 0.35 * min(distance / maximum_image_distance, 2.0)
            if player_feet:
                nearest_player = min(float(np.linalg.norm(center - foot)) for foot in player_feet)
                score += 0.10 * max(0.0, 1.0 - nearest_player / 150.0)
            if score > best_score:
                best = (candidate, position)
                best_score = score
        return best

    def _start_segment(
        self,
        candidate: BallCandidate,
        position_m: Optional[np.ndarray],
        timestamp: float,
    ) -> dict[int, dict]:
        self.track_segment += 1
        self._reset_motion_state()
        return self._accept_observation(candidate, position_m, timestamp)

    def _accept_observation(
        self,
        candidate: BallCandidate,
        position_m: Optional[np.ndarray],
        timestamp: float,
    ) -> dict[int, dict]:
        center = np.asarray(candidate.center, dtype=np.float64)
        filtered_center = self.image_filter.correct(center, timestamp)
        filtered_position: Optional[np.ndarray] = None
        if position_m is not None:
            filtered_position = self.metric_filter.correct(position_m, timestamp)
        bbox_size = np.asarray(
            [candidate.bbox[2] - candidate.bbox[0], candidate.bbox[3] - candidate.bbox[1]],
            dtype=np.float64,
        )
        self.last_bbox_size = bbox_size
        self.last_observed_timestamp = timestamp
        self.track_confidence = (
            candidate.confidence
            if self.track_confidence == 0
            else 0.7 * self.track_confidence + 0.3 * candidate.confidence
        )
        self.confirmation_count += 1
        self.segment_confirmed = self.confirmation_count >= self.confirmations_required
        width, height = bbox_size
        center_x, center_y = filtered_center
        track = {
            "bbox": [
                float(center_x - width / 2),
                float(center_y - height / 2),
                float(center_x + width / 2),
                float(center_y + height / 2),
            ],
            "confidence": candidate.confidence,
            "observed": True,
            "track_confidence": float(self.track_confidence),
            "track_segment": self.track_segment,
            "track_confirmed": self.segment_confirmed,
        }
        if filtered_position is not None and self.geometry.contains(
            tuple(filtered_position), margin_m=0.0
        ):
            track["position_m"] = tuple(map(float, filtered_position))
        return {1: track}

    def _predicted_track(
        self,
        center: Optional[np.ndarray],
        position_m: Optional[np.ndarray],
        timestamp: float,
        calibration: CalibrationResult,
    ) -> dict[int, dict]:
        if (
            center is None
            or self.last_observed_timestamp is None
            or timestamp - self.last_observed_timestamp > self.maximum_prediction_seconds
        ):
            return {}
        height, width = self.last_bbox_size[1], self.last_bbox_size[0]
        x, y = center
        gap = timestamp - self.last_observed_timestamp
        confidence = self.track_confidence * exp(-gap / self.maximum_prediction_seconds)
        confirmed = self.segment_confirmed and gap <= self.restart_gap_seconds
        track = {
            "bbox": [
                float(x - width / 2),
                float(y - height / 2),
                float(x + width / 2),
                float(y + height / 2),
            ],
            "confidence": float(confidence),
            "observed": False,
            "track_confidence": float(confidence),
            "track_segment": self.track_segment,
            "track_confirmed": confirmed,
        }
        if position_m is not None and calibration.image_to_pitch is not None:
            track["position_m"] = tuple(map(float, position_m))
        return {1: track}

    def reset(self) -> None:
        self._reset_motion_state()
