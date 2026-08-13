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


@dataclass(frozen=True)
class BallCandidate:
    bbox: tuple[float, float, float, float]
    confidence: float
    source: str = "detector"
    appearance_score: float = 0.5
    line_score: float = 0.0
    verifier_score: Optional[float] = None
    verifier_threshold: Optional[float] = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass(frozen=True)
class _BallHypothesis:
    """One path in the fixed-lag candidate beam."""

    candidate: BallCandidate
    position_m: Optional[np.ndarray]
    cumulative_score: float
    observations: int

    @property
    def mean_score(self) -> float:
        return self.cumulative_score / max(1, self.observations)


def inspect_ball_model_export(path: str | Path) -> dict[str, object]:
    """Read deployment metadata without loading model weights."""
    model_path = Path(path)
    metadata_path = model_path / "metadata.yaml" if model_path.is_dir() else None
    if metadata_path is None or not metadata_path.is_file():
        return {
            "path": str(model_path),
            "format": model_path.suffix.lower().lstrip(".") or "unknown",
            "metadata_available": False,
            "dynamic": None,
            "imgsz": None,
            "model_family": None,
        }
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    image_size = metadata.get("imgsz")
    if isinstance(image_size, (list, tuple)):
        image_size = max(int(value) for value in image_size)
    elif image_size is not None:
        image_size = int(image_size)
    description = str(metadata.get("description", ""))
    return {
        "path": str(model_path),
        "format": "openvino",
        "metadata_available": True,
        "dynamic": bool((metadata.get("args") or {}).get("dynamic")),
        "imgsz": image_size,
        "model_family": description.split(" model", 1)[0].replace("Ultralytics ", "") or None,
        "names": metadata.get("names"),
    }


def validate_ball_model_for_promotion(
    path: str | Path,
    *,
    minimum_imgsz: int = 1280,
    require_dynamic: bool = True,
    required_family: str = "YOLO11s",
) -> list[str]:
    """Return deployment-policy failures that block promotion to runtime weights."""
    profile = inspect_ball_model_export(path)
    errors: list[str] = []
    if not bool(profile.get("metadata_available")):
        errors.append(
            "formal promotion requires an exported model directory with metadata.yaml"
        )
    if require_dynamic and not bool(profile.get("dynamic")):
        errors.append("ball model export must use dynamic=True")
    image_size = profile.get("imgsz")
    if image_size is not None and int(image_size) < minimum_imgsz:
        errors.append(f"ball model export imgsz must be at least {minimum_imgsz}")
    family = profile.get("model_family")
    if family is not None and str(family).casefold() != required_family.casefold():
        errors.append(f"ball model family must be {required_family}, found {family}")
    names = profile.get("names")
    if isinstance(names, dict):
        normalized = [names[key] for key in sorted(names, key=lambda value: int(value))]
        if normalized != ["ball"]:
            errors.append(f"ball model classes must be ['ball'], found {normalized}")
    return errors


def ball_verifier_threshold_path(path: str | Path) -> Path:
    model_path = Path(path)
    return (
        model_path / "verifier_threshold.json"
        if model_path.is_dir()
        else model_path.with_suffix(".verifier.json")
    )


def validate_ball_verifier_for_promotion(path: str | Path) -> list[str]:
    """Require the validation-selected operating point for formal use."""
    import json

    threshold_path = ball_verifier_threshold_path(path)
    if not threshold_path.is_file():
        return [f"ball verifier threshold sidecar is missing: {threshold_path}"]
    try:
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(payload["decision_threshold"])
        recall = float(payload["recall"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return [f"ball verifier threshold sidecar is invalid: {threshold_path}"]
    errors = []
    if not 0.0 <= threshold <= 1.0:
        errors.append("ball verifier decision threshold must be between 0 and 1")
    if recall < 0.75:
        errors.append("ball verifier validation recall must be at least 75%")
    return errors


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
        verifier_model_path: Optional[str] = None,
    ) -> None:
        path = Path(model_path)
        self.model = YOLO(model_path, task="detect")
        if path.is_file():
            self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        fixed_size = self._fixed_export_size(path)
        self.global_imgsz = fixed_size or global_imgsz
        self.tile_imgsz = fixed_size or tile_imgsz
        self.confidence = confidence
        self.overlap = overlap
        self.verifier = (
            BallVerifier(verifier_model_path) if verifier_model_path is not None else None
        )

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
        candidates.extend(self._predict(frame, 0, 0, self.global_imgsz, "global"))
        for tile, offset_x, offset_y in overlapping_tiles(frame, overlap=self.overlap):
            candidates.extend(
                self._predict(tile, offset_x, offset_y, self.tile_imgsz, "tile")
            )
        return [
            self._with_visual_evidence(frame, candidate)
            for candidate in non_max_suppression(candidates)
        ]

    def _predict(
        self,
        frame: np.ndarray,
        offset_x: int,
        offset_y: int,
        image_size: int,
        source: str,
    ) -> list[BallCandidate]:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=image_size,
            verbose=False,
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
                    source=source,
                )
            )
        return candidates

    def _with_visual_evidence(
        self, frame: np.ndarray, candidate: BallCandidate
    ) -> BallCandidate:
        """Estimate whether the candidate sits on an elongated white pitch marking."""
        height, width = frame.shape[:2]
        cx, cy = candidate.center
        box_width = max(2.0, candidate.bbox[2] - candidate.bbox[0])
        box_height = max(2.0, candidate.bbox[3] - candidate.bbox[1])
        radius = int(max(16.0, 4.0 * max(box_width, box_height)))
        x1, x2 = max(0, int(cx) - radius), min(width, int(cx) + radius + 1)
        y1, y2 = max(0, int(cy) - radius), min(height, int(cy) + radius + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return candidate
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.asarray([0, 0, 145]), np.asarray([180, 90, 255]))
        edges = cv2.Canny(white, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(6, radius // 3),
            minLineLength=max(6, int(max(box_width, box_height) * 2.0)),
            maxLineGap=max(2, radius // 5),
        )
        longest = 0.0
        if lines is not None:
            for line in lines[:, 0]:
                longest = max(
                    longest,
                    hypot(float(line[2] - line[0]), float(line[3] - line[1])),
                )
        line_score = min(1.0, longest / max(1.0, radius * 1.5))
        center_radius = max(2, int(min(box_width, box_height) / 2))
        local_center_x = int(round(cx)) - x1
        local_center_y = int(round(cy)) - y1
        local = white[
            max(0, local_center_y - center_radius) : min(
                white.shape[0], local_center_y + center_radius + 1
            ),
            max(0, local_center_x - center_radius) : min(
                white.shape[1], local_center_x + center_radius + 1
            ),
        ]
        white_ratio = float(np.mean(local > 0)) if local.size else 0.0
        heuristic_score = float(np.clip(0.35 + 0.65 * white_ratio, 0.0, 1.0))
        verifier_score = (
            self.verifier.score(frame, candidate)
            if self.verifier is not None
            else None
        )
        appearance_score = verifier_score if verifier_score is not None else heuristic_score
        return BallCandidate(
            bbox=candidate.bbox,
            confidence=candidate.confidence,
            source=candidate.source,
            appearance_score=appearance_score,
            line_score=line_score,
            verifier_score=verifier_score,
            verifier_threshold=(
                self.verifier.decision_threshold
                if self.verifier is not None
                else None
            ),
        )


class BallVerifier:
    """Optional learned ball/non-ball Adapter at the candidate-verification Seam."""

    def __init__(self, model_path: str, image_size: int = 128) -> None:
        self.model = YOLO(model_path, task="classify")
        self.image_size = image_size
        path = Path(model_path)
        threshold_path = ball_verifier_threshold_path(path)
        self.decision_threshold = 0.50
        if threshold_path.is_file():
            import json

            payload = json.loads(threshold_path.read_text(encoding="utf-8"))
            self.decision_threshold = float(payload["decision_threshold"])
            if not 0.0 <= self.decision_threshold <= 1.0:
                raise ValueError("Ball verifier decision threshold must be between 0 and 1")
        names = self.model.names
        normalized = {
            str(name).casefold(): int(index) for index, name in names.items()
        } if isinstance(names, dict) else {
            str(name).casefold(): index for index, name in enumerate(names)
        }
        if "ball" not in normalized:
            raise ValueError("Ball verifier classes must include 'ball'")
        self.ball_index = normalized["ball"]

    def score(self, frame: np.ndarray, candidate: BallCandidate) -> float:
        height, width = frame.shape[:2]
        cx, cy = candidate.center
        side = max(
            32,
            int(
                max(
                    candidate.bbox[2] - candidate.bbox[0],
                    candidate.bbox[3] - candidate.bbox[1],
                )
                * 4.0
            ),
        )
        half = side // 2
        x1, x2 = max(0, int(cx) - half), min(width, int(cx) + half + 1)
        y1, y2 = max(0, int(cy) - half), min(height, int(cy) + half + 1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        result = self.model.predict(crop, imgsz=self.image_size, verbose=False)[0]
        if result.probs is None:
            return 0.0
        probabilities = result.probs.data.detach().cpu().numpy()
        return float(probabilities[self.ball_index])


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
        minimum_confirmation_confidence: float = 0.15,
        fixed_lag_seconds: float = 0.0,
        ambiguity_margin: float = 0.04,
    ) -> None:
        self.geometry = geometry
        self.maximum_prediction_seconds = maximum_prediction_seconds
        self.restart_gap_seconds = restart_gap_seconds
        self.confirmations_required = confirmations_required
        self.minimum_confirmation_confidence = minimum_confirmation_confidence
        self.fixed_lag_seconds = fixed_lag_seconds
        self.ambiguity_margin = ambiguity_margin
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
        self.segment_started_timestamp: Optional[float] = None
        self.track_length = 0
        self.selected_score = 0.0
        self.track_state = "tentative"
        self.rejection_reasons: list[str] = []
        self._last_ambiguous = False
        self._last_hypothesis_count = 0
        self._hypotheses: list[_BallHypothesis] = []

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
            return self._accept_observation(
                candidate,
                position_m,
                timestamp_seconds,
                candidate_count=len(valid_candidates),
            )
        if valid_candidates:
            if not continuing_segment:
                self._hypotheses = []
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
                return self._start_segment(
                    candidate,
                    position_m,
                    timestamp_seconds,
                    candidate_count=len(valid_candidates),
                )
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

        scored_candidates: list[
            tuple[float, BallCandidate, Optional[np.ndarray]]
        ] = []
        for candidate, position in candidates:
            center = np.asarray(candidate.center)
            bbox_width = candidate.bbox[2] - candidate.bbox[0]
            bbox_height = candidate.bbox[3] - candidate.bbox[1]
            maximum_size = max(48.0, min(frame_shape[:2]) * 0.05)
            if max(bbox_width, bbox_height) > maximum_size:
                continue
            score = candidate.confidence
            score += 0.12 * candidate.appearance_score
            if candidate.verifier_score is not None:
                verifier_threshold = (
                    candidate.verifier_threshold
                    if candidate.verifier_threshold is not None
                    else 0.50
                )
                score += 0.50 * (candidate.verifier_score - verifier_threshold)
            score -= 0.40 * candidate.line_score
            if position is not None:
                marking_distance = self.geometry.distance_to_marking(tuple(position))
                if marking_distance < 0.35:
                    score -= 0.20 * (1.0 - marking_distance / 0.35)
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
            scored_candidates.append((float(score), candidate, position))
        if not scored_candidates:
            return None

        next_hypotheses: list[_BallHypothesis] = []
        for base_score, candidate, position in scored_candidates:
            best_path = _BallHypothesis(candidate, position, base_score, 1)
            best_rank = base_score
            center = np.asarray(candidate.center, dtype=np.float64)
            for previous in self._hypotheses:
                previous_center = np.asarray(previous.candidate.center, dtype=np.float64)
                if previous.position_m is not None and position is not None:
                    continuity = float(np.linalg.norm(position - previous.position_m))
                    normalized_continuity = continuity / max(maximum_metric_distance, 1e-6)
                else:
                    continuity = float(np.linalg.norm(center - previous_center))
                    normalized_continuity = continuity / max(maximum_image_distance, 1e-6)
                if normalized_continuity > 2.0:
                    continue
                cumulative = (
                    previous.cumulative_score
                    + base_score
                    - 0.18 * normalized_continuity
                )
                observations = previous.observations + 1
                rank = cumulative / observations + min(0.08, observations * 0.01)
                if rank > best_rank:
                    best_path = _BallHypothesis(
                        candidate,
                        position,
                        cumulative,
                        observations,
                    )
                    best_rank = rank
            next_hypotheses.append(best_path)

        next_hypotheses.sort(
            key=lambda hypothesis: (
                hypothesis.mean_score
                + min(0.08, hypothesis.observations * 0.01)
            ),
            reverse=True,
        )
        self._hypotheses = next_hypotheses[:3]
        ranked_scores = [
            hypothesis.mean_score
            + min(0.08, hypothesis.observations * 0.01)
            for hypothesis in self._hypotheses
        ]
        best_hypothesis = self._hypotheses[0]
        self._last_hypothesis_count = len(self._hypotheses)
        self._last_ambiguous = (
            len(ranked_scores) > 1
            and ranked_scores[0] - ranked_scores[1] < self.ambiguity_margin
        )
        self.selected_score = float(ranked_scores[0])
        return best_hypothesis.candidate, best_hypothesis.position_m

    def _start_segment(
        self,
        candidate: BallCandidate,
        position_m: Optional[np.ndarray],
        timestamp: float,
        candidate_count: int,
    ) -> dict[int, dict]:
        self.track_segment += 1
        selected_score = self.selected_score
        ambiguous = self._last_ambiguous
        hypothesis_count = self._last_hypothesis_count
        hypotheses = list(self._hypotheses)
        self._reset_motion_state()
        self.selected_score = selected_score
        self._last_ambiguous = ambiguous
        self._last_hypothesis_count = hypothesis_count
        self._hypotheses = hypotheses
        return self._accept_observation(
            candidate,
            position_m,
            timestamp,
            candidate_count=candidate_count,
        )

    def _accept_observation(
        self,
        candidate: BallCandidate,
        position_m: Optional[np.ndarray],
        timestamp: float,
        candidate_count: int,
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
        if self.segment_started_timestamp is None:
            self.segment_started_timestamp = timestamp
        self.track_confidence = (
            candidate.confidence
            if self.track_confidence == 0
            else 0.7 * self.track_confidence + 0.3 * candidate.confidence
        )
        self.confirmation_count += 1
        self.track_length += 1
        elapsed = timestamp - self.segment_started_timestamp
        marking_distance = (
            self.geometry.distance_to_marking(tuple(filtered_position))
            if filtered_position is not None
            else float("inf")
        )
        line_like = candidate.line_score >= 0.55 or marking_distance < 0.20
        evidence_confidence = self.track_confidence >= self.minimum_confirmation_confidence
        verifier_rejected = (
            candidate.verifier_score is not None
            and candidate.verifier_score
            < (
                candidate.verifier_threshold
                if candidate.verifier_threshold is not None
                else 0.50
            )
        )
        if verifier_rejected:
            evidence_confidence = False
        if line_like and self.track_confidence < 0.35:
            evidence_confidence = False
        self.segment_confirmed = (
            self.confirmation_count >= self.confirmations_required
            and elapsed >= self.fixed_lag_seconds
            and evidence_confidence
            and not self._last_ambiguous
        )
        self.track_state = (
            "ambiguous"
            if self._last_ambiguous
            else "confirmed" if self.segment_confirmed else "tentative"
        )
        self.rejection_reasons = []
        if self._last_ambiguous:
            self.rejection_reasons.append("track_ambiguous")
        if not evidence_confidence:
            self.rejection_reasons.append("low_ball_confidence")
        if verifier_rejected:
            self.rejection_reasons.append("appearance_rejected")
        if line_like:
            self.rejection_reasons.append("pitch_marking_overlap")
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
            "track_state": self.track_state,
            "track_length": self.track_length,
            "selected_score": self.selected_score,
            "candidate_count": int(candidate_count),
            "hypothesis_count": self._last_hypothesis_count,
            "appearance_score": candidate.appearance_score,
            "line_score": candidate.line_score,
            "verifier_score": candidate.verifier_score,
            "verifier_threshold": candidate.verifier_threshold,
            "rejection_reasons": list(self.rejection_reasons),
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
            "track_state": "occluded" if confirmed else "tentative",
            "track_length": self.track_length,
            "selected_score": self.selected_score,
            "candidate_count": 0,
            "hypothesis_count": self._last_hypothesis_count,
            "rejection_reasons": ["insufficient_samples"],
        }
        if position_m is not None and calibration.image_to_pitch is not None:
            track["position_m"] = tuple(map(float, position_m))
        return {1: track}

    def reset(self) -> None:
        self._reset_motion_state()
