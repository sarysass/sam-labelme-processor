"""
SAM wrapper module for MicroHunter SAM integration.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Add MicroHunter to path
MICROHUNTER_PATH = Path('/Users/shali/projects/MicroHunter')
if str(MICROHUNTER_PATH) not in sys.path:
    sys.path.insert(0, str(MICROHUNTER_PATH))

try:
    from microhunter.core.sam import UltralyticsSAMPredictor
    MICROHUNTER_AVAILABLE = True
except ImportError:
    logger.warning("MicroHunter not available. SAM functionality will be limited.")
    MICROHUNTER_AVAILABLE = False
    UltralyticsSAMPredictor = None


class SAMWrapper:
    """
    SAM model wrapper.

    Wraps MicroHunter's UltralyticsSAMPredictor, providing a simplified interface.
    """

    def __init__(
        self,
        weights: str = "weights/sam2.1_t.pt",
        device: Optional[str] = None,
        imgsz: int = 1024,
        iou_threshold: float = 0.3
    ):
        """
        Initialize SAM model.

        Args:
            weights: Model weights path.
            device: Inference device ('cuda:0', 'cpu', 'auto').
            imgsz: Input image size.
            iou_threshold: IoU threshold.
        """
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.predictor = None

    def load_model(self) -> None:
        """Load SAM model."""
        if not MICROHUNTER_AVAILABLE:
            raise ImportError("MicroHunter is not available. Please ensure MicroHunter is installed and accessible.")

        logger.info(f"Loading SAM model from {self.weights}")
        self.predictor = UltralyticsSAMPredictor(
            weights=self.weights,
            device=self.device
        )
        logger.info("SAM model loaded successfully")

    def predict(
        self,
        image: np.ndarray,
        bboxes: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """
        Perform SAM inference on image.

        Args:
            image: BGR format numpy array (H, W, 3).
            bboxes: BBox list, format [[x1, y1, x2, y2], ...].

        Returns:
            List of mask information for each bbox:
            [
                {
                    'mask': np.ndarray,      # Binary mask (H, W)
                    'contour': np.ndarray,   # Contour points (N, 2)
                    'bbox': [x1, y1, x2, y2] # Input bbox
                },
                ...
            ]
        """
        if self.predictor is None:
            self.load_model()

        # Call MicroHunter's predict
        result = self.predictor.predict(
            source=image,
            bboxes=bboxes,
            imgsz=self.imgsz,
            iou_threshold=self.iou_threshold
        )

        # Parse result
        masks = result.get('masks', [])      # (N, H, W)
        contours = result.get('contours', []) # list of (M, 2)

        # Assemble output
        outputs = []
        for i, (bbox, mask, contour) in enumerate(zip(bboxes, masks, contours)):
            outputs.append({
                'mask': mask,
                'contour': contour,
                'bbox': bbox
            })

        return outputs

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit."""
        # Cleanup resources if needed
        pass
