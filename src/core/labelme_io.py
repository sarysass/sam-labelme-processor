"""
Labelme JSON I/O module.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import base64
from dataclasses import dataclass
import numpy as np


@dataclass
class BBoxShape:
    """Bounding box shape in Labelme format."""

    label: str
    points: List[List[float]]  # [[x1, y1], [x2, y2]]
    group_id: Optional[int] = None
    description: str = ""
    flags: Optional[Dict[str, bool]] = None

    def __post_init__(self):
        if self.flags is None:
            object.__setattr__(self, "flags", {})


@dataclass
class MaskShape:
    """Mask shape in Labelme format."""

    label: str
    points: List[List[float]]  # Bounding points [[x1, y1], [x2, y2]]
    group_id: Optional[int] = None
    description: str = ""
    flags: Optional[Dict[str, bool]] = None
    area: float = 0.0
    shape_type: str = "mask"
    mask: Optional[str] = None

    def __post_init__(self):
        if self.flags is None:
            object.__setattr__(self, "flags", {})


@dataclass
class ImageInfo:
    """Image information in Labelme format."""

    image_path: str
    image_height: int
    image_width: int
    version: str = "5.10.1"
    flags: Optional[Dict[str, bool]] = None
    image_data: Optional[str] = None  # Base64 encoded image data

    def __post_init__(self):
        if self.flags is None:
            object.__setattr__(self, "flags", {})


class LabelmeIO:
    """Labelme JSON read/write class."""

    @staticmethod
    def read_bbox_json(json_path: Path) -> Tuple[ImageInfo, List[BBoxShape]]:
        """
        Read BBox JSON file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            Tuple of (image_info, bbox_shapes).
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Parse image info
        image_info = ImageInfo(
            image_path=data.get("imagePath", ""),
            image_height=data.get("imageHeight", 0),
            image_width=data.get("imageWidth", 0),
            version=data.get("version", "5.10.1"),
            image_data=data.get("imageData"),
        )

        # Parse bbox shapes
        bbox_shapes = []
        for shape_data in data.get("shapes", []):
            if shape_data.get("shape_type") == "rectangle":
                bbox_shapes.append(
                    BBoxShape(
                        label=shape_data.get("label", ""),
                        points=shape_data.get("points", []),
                        group_id=shape_data.get("group_id"),
                        description=shape_data.get("description", ""),
                        flags=shape_data.get("flags", {}),
                    )
                )

        return image_info, bbox_shapes

    @staticmethod
    def write_mask_json(
        json_path: Path, mask_shapes: List[MaskShape], image_info: ImageInfo
    ) -> None:
        """
        Write Mask JSON file (separate output).

        Args:
            json_path: Path to write the JSON file.
            mask_shapes: List of mask shapes.
            image_info: Image information.
        """
        # Convert shapes to dict
        shapes_dict = []
        for shape in mask_shapes:
            shapes_dict.append(
                {
                    "label": shape.label,
                    "points": shape.points,
                    "group_id": shape.group_id,
                    "shape_type": "mask",
                    "description": shape.description,
                    "flags": shape.flags,
                    "mask": shape.mask,
                }
            )

        # Create JSON data
        data = {
            "version": image_info.version,
            "flags": image_info.flags or {},
            "shapes": shapes_dict,
            "imagePath": image_info.image_path,
            "imageHeight": image_info.image_height,
            "imageWidth": image_info.image_width,
            "imageData": image_info.image_data,
        }

        # Write to file
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def write_combined_json(
        json_path: Path,
        bbox_shapes: List[BBoxShape],
        mask_shapes: List[MaskShape],
        image_info: ImageInfo,
    ) -> None:
        """
        Write combined JSON file (bbox + mask).

        Args:
            json_path: Path to write the JSON file.
            bbox_shapes: List of bbox shapes.
            mask_shapes: List of mask shapes.
            image_info: Image information.
        """
        # Convert bbox shapes to dict
        shapes_dict = []
        for shape in bbox_shapes:
            shapes_dict.append(
                {
                    "label": shape.label,
                    "points": shape.points,
                    "group_id": shape.group_id,
                    "shape_type": "rectangle",
                    "description": shape.description,
                    "flags": shape.flags,
                }
            )

        # Append mask shapes
        for shape in mask_shapes:
            shapes_dict.append(
                {
                    "label": shape.label,
                    "points": shape.points,
                    "group_id": shape.group_id,
                    "shape_type": "mask",
                    "description": shape.description,
                    "flags": shape.flags,
                    "mask": shape.mask,
                }
            )

        # Create JSON data
        data = {
            "version": image_info.version,
            "flags": image_info.flags or {},
            "shapes": shapes_dict,
            "imagePath": image_info.image_path,
            "imageHeight": image_info.image_height,
            "imageWidth": image_info.image_width,
            "imageData": image_info.image_data,
        }

        # Write to file
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def mask_to_labelme_mask(mask: np.ndarray) -> Tuple[List[List[float]], str]:
        """
        Convert binary mask to Labelme mask payload.

        Args:
            mask: Binary mask array (H, W).

        Returns:
            Tuple of points [[x1, y1], [x2, y2]] and base64 PNG mask.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for mask_to_labelme_mask")

        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D array")

        binary_mask = (mask > 0).astype(np.uint8) * 255
        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0:
            return [], ""

        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())

        cropped = binary_mask[y1 : y2 + 1, x1 : x2 + 1]
        success, encoded = cv2.imencode(".png", cropped)
        if not success:
            raise ValueError("Failed to encode mask to PNG")

        mask_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        points = [[float(x1), float(y1)], [float(x2), float(y2)]]
        return points, mask_b64
