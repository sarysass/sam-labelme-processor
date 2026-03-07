"""
SAM wrapper module.
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np

from .sam_backend import SAMBackend
from .ultralytics_sam_backend import UltralyticsSAMBackend

logger = logging.getLogger(__name__)


class SAMWrapper:
    """
    Thin facade over the configured SAM backend.
    """

    def __init__(
        self,
        weights: str = "weights/sam2.1_t.pt",
        device: Optional[str] = None,
        imgsz: int = 1024,
        iou_threshold: float = 0.3,
        backend: Optional[SAMBackend] = None,
    ):
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.backend = backend or UltralyticsSAMBackend(
            weights=weights,
            device=device,
            imgsz=imgsz,
            iou_threshold=iou_threshold,
        )
        self.predictor: Optional[SAMBackend] = None

    def load_model(self) -> None:
        """Load the configured backend."""
        logger.info(f"Loading SAM model from {self.weights}")
        self.backend.load_model()
        self.predictor = self.backend
        logger.info("SAM model loaded successfully")

    def predict(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Perform SAM inference on an image using the configured backend."""
        if self.predictor is None:
            self.load_model()

        return self.predictor.predict(image, bboxes)

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit."""
        return None
