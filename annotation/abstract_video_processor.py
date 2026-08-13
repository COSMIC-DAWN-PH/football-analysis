from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

class AbstractVideoProcessor(ABC):

    def finalize(self) -> None:
        """Flush optional run-level artifacts after the final frame."""
        return None

    @abstractmethod
    def process(
        self,
        frames: List[np.ndarray],
        fps: float = 1e-6,
        timestamps: Optional[List[float]] = None,
        source_frame_numbers: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Abstract method for video processing

        Args:
            frames (List[np.ndarray]): Frame batch to process.
            fps (float): Video FPS.
            timestamps (Optional[List[float]]): Per-frame presentation times in seconds.
            source_frame_numbers (Optional[List[int]]): Original video frame indexes.
        
        Returns:
            List[np.ndarray]: Processed frames.
        """
        pass
