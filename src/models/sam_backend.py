"""
Backend interface for SAM inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class SAMBackend(ABC):
    """Abstract backend for SAM inference."""

    @abstractmethod
    def load_model(self) -> None:
        """Load backend model resources."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Run box-prompt segmentation and normalize outputs."""
