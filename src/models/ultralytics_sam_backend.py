"""
Local Ultralytics-backed SAM inference backend.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import importlib

import numpy as np

from .sam_backend import SAMBackend


class UltralyticsSAMBackend(SAMBackend):
    """Run SAM inference through the public Ultralytics API."""

    def __init__(
        self,
        weights: Union[str, Path] = "weights/sam2.1_t.pt",
        device: Optional[str] = None,
        imgsz: int = 1024,
        iou_threshold: float = 0.3,
        half: bool = True,
        verbose: bool = False,
        retina_masks: bool = True,
    ):
        self.weights = str(weights)
        self.device = device
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.half = half
        self.verbose = verbose
        self.retina_masks = retina_masks
        self.model = None

    @staticmethod
    def _import_sam_class():
        """Import the Ultralytics SAM class lazily."""
        try:
            ultralytics_module = importlib.import_module("ultralytics")
        except ImportError as e:
            raise ImportError(
                "Ultralytics is required for SAM inference. Install it with "
                "`pip install ultralytics`."
            ) from e

        try:
            return ultralytics_module.SAM
        except AttributeError as e:
            raise ImportError("Ultralytics SAM class is not available in the installed package.") from e

    def load_model(self) -> None:
        """Load the Ultralytics SAM model."""
        sam_class = self._import_sam_class()
        self.model = sam_class(self.weights)

    @staticmethod
    def _to_numpy(mask_data: Any) -> np.ndarray:
        """Convert tensor-like mask data to numpy."""
        if hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu()
        if hasattr(mask_data, "numpy"):
            mask_data = mask_data.numpy()
        return np.asarray(mask_data, dtype=np.uint8)

    @staticmethod
    def _empty_contour() -> np.ndarray:
        """Return a normalized empty contour array."""
        return np.array([]).reshape(0, 2)

    @staticmethod
    def _prepare_bboxes(
        bboxes: List[List[float]],
        image_height: int,
        image_width: int,
    ) -> np.ndarray:
        """Clip invalid boxes and preserve one slot per input box."""
        if not bboxes:
            return np.empty((0, 4), dtype=np.float32)

        input_bboxes = np.array(bboxes, dtype=np.float32)
        input_bboxes[:, [0, 2]] = np.clip(input_bboxes[:, [0, 2]], 0, image_width)
        input_bboxes[:, [1, 3]] = np.clip(input_bboxes[:, [1, 3]], 0, image_height)

        invalid = (input_bboxes[:, 0] >= input_bboxes[:, 2]) | (input_bboxes[:, 1] >= input_bboxes[:, 3])
        input_bboxes[invalid] = [0, 0, 1, 1]
        return input_bboxes

    def predict(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Run box-prompt inference and normalize one output per input box."""
        if self.model is None:
            self.load_model()

        if image.ndim != 3:
            raise ValueError("image must be a 3D numpy array in BGR format")

        image_height, image_width = image.shape[:2]
        prepared_bboxes = self._prepare_bboxes(bboxes, image_height, image_width)
        if len(prepared_bboxes) == 0:
            return []

        result = self.model.predict(
            source=image,
            bboxes=prepared_bboxes.tolist(),
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            retina_masks=self.retina_masks,
            verbose=self.verbose,
            iou=self.iou_threshold,
        )[0]

        output_masks = [np.zeros((image_height, image_width), dtype=np.uint8) for _ in range(len(prepared_bboxes))]
        output_contours = [self._empty_contour() for _ in range(len(prepared_bboxes))]

        masks_obj = getattr(result, "masks", None)
        if masks_obj is not None and len(masks_obj) > 0:
            masks_np = self._to_numpy(masks_obj.data)
            contours_list = list(getattr(masks_obj, "xy", []))
            for i in range(min(len(masks_np), len(prepared_bboxes))):
                output_masks[i] = masks_np[i]
                if i < len(contours_list):
                    output_contours[i] = np.asarray(contours_list[i])

        outputs: List[Dict[str, Any]] = []
        for bbox, mask, contour in zip(bboxes, output_masks, output_contours):
            outputs.append(
                {
                    "mask": mask,
                    "contour": contour,
                    "bbox": bbox,
                }
            )
        return outputs
