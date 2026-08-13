from abc import ABC, abstractmethod
from ultralytics import YOLO
import torch
from pathlib import Path
from typing import Any, Dict, List
from ultralytics.engine.results import Results
import numpy as np
import yaml

class AbstractTracker(ABC):

    def __init__(self, model_path: str, conf: float = 0.1, task: str | None = None) -> None:
        """
        Load the model from the given path and set the confidence threshold.

        Args:
            model_path (str): Path to the model.
            conf (float): Confidence threshold for detections.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path, task=task)
        if Path(model_path).is_file():
            self.model.to(device)
        self.conf = conf  # Set confidence threshold
        self.cur_frame = 0  # Initialize current frame counter
        self.fixed_imgsz = self._read_fixed_export_imgsz(Path(model_path))

    @staticmethod
    def _read_fixed_export_imgsz(model_path: Path) -> int | None:
        if not model_path.is_dir():
            return None
        metadata_path = model_path / "metadata.yaml"
        if not metadata_path.is_file():
            return None
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}
            image_size = metadata.get("imgsz")
            if isinstance(image_size, (list, tuple)) and image_size:
                return int(max(image_size))
            if image_size is not None:
                return int(image_size)
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            return None
        return None

    def inference_imgsz(self, requested: int) -> int:
        """Use a fixed exported model size when the backend is not dynamic."""
        return self.fixed_imgsz or requested

    @abstractmethod
    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Abstract method for YOLO detection.

        Args:
            frames (List[np.ndarray]): List of frames for detection.

        Returns:
            List[Results]: List of YOLO detection result objects.
        """
        pass
        
    @abstractmethod
    def track(self, detection: Results) -> dict:
        """
        Abstract method for tracking detections.

        Args:
            detection (Results): YOLO detection results for a single frame.

        Returns:
            dict: Tracking data.
        """
        pass
