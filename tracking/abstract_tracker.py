from abc import ABC, abstractmethod
from ultralytics import YOLO
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple
from ultralytics.engine.results import Results
import numpy as np
import yaml


def resolve_inference_device(
    model_path: str,
    requested: str = "auto",
    priority: Tuple[str, ...] = ("NPU", "GPU", "CPU"),
) -> str:
    """
    Resolve the runtime device for a model.

    For OpenVINO exports (directory weights) the device is probed against the
    devices OpenVINO reports and returned in the ultralytics 'intel:<DEVICE>'
    convention so iGPU/NPU acceleration can be used. For torch weights (.pt)
    only torch devices (cuda/cpu) are returned. An explicit `requested` value
    always wins and is returned untouched.
    """
    if requested != "auto":
        return requested
    if Path(model_path).is_dir():
        try:
            import openvino as ov

            available = {device.upper() for device in ov.Core().available_devices}
        except Exception:
            available = set()
        for name in priority:
            if name.upper() in available:
                return f"intel:{name.upper()}"
        return "intel:CPU"
    return "cuda" if torch.cuda.is_available() else "cpu"


class AbstractTracker(ABC):

    def __init__(
        self,
        model_path: str,
        conf: float = 0.1,
        task: str | None = None,
        device: str = "auto",
    ) -> None:
        """
        Load the model from the given path and set the confidence threshold.

        Args:
            model_path (str): Path to the model.
            conf (float): Confidence threshold for detections.
            task (str | None): YOLO task, e.g. "detect" or "pose".
            device (str): Inference device. "auto" probes OpenVINO iGPU/NPU for
                exported models and falls back to torch cuda/cpu.
        """
        self.device = resolve_inference_device(model_path, requested=device)
        self.model = YOLO(model_path, task=task)
        if Path(model_path).is_file():
            self.model.to(torch.device(self.device))
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
            if bool((metadata.get("args") or {}).get("dynamic")):
                return None
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
