"""
Labelme file adapters for mask refinement.
"""

import base64
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from ..core.labelme_io import LabelmeIO
from .edge_refiner import EdgeOptimizationConfig, refine_mask_with_edges


def decode_labelme_mask(
    mask_b64: str,
    points: List[List[float]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Decode a Labelme mask payload into a full-size binary mask."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if not mask_b64 or len(points) != 2:
        return mask

    png_data = base64.b64decode(mask_b64)
    decoded = cv2.imdecode(
        np.frombuffer(png_data, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if decoded is None:
        raise ValueError("Failed to decode Labelme mask PNG payload")

    x1 = max(0, int(round(points[0][0])))
    y1 = max(0, int(round(points[0][1])))
    x2 = min(image_width - 1, int(round(points[1][0])))
    y2 = min(image_height - 1, int(round(points[1][1])))

    expected_height = max(0, y2 - y1 + 1)
    expected_width = max(0, x2 - x1 + 1)
    if expected_height == 0 or expected_width == 0:
        return mask

    if decoded.shape[:2] != (expected_height, expected_width):
        decoded = cv2.resize(
            decoded,
            (expected_width, expected_height),
            interpolation=cv2.INTER_NEAREST,
        )

    mask[y1 : y2 + 1, x1 : x2 + 1] = (decoded > 0).astype(np.uint8)
    return mask


def optimize_labelme_mask_file(
    input_json_path: Path,
    output_json_path: Path,
    config: EdgeOptimizationConfig,
) -> Dict[str, int]:
    """Optimize all mask shapes in a Labelme JSON file."""
    with open(input_json_path, "r") as f:
        data = json.load(f)

    image_path = (input_json_path.parent / data["imagePath"]).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load source image: {image_path}")

    image_height = int(data["imageHeight"])
    image_width = int(data["imageWidth"])
    updated_shapes = 0

    shapes = data.get("shapes", [])
    for shape in shapes:
        if shape.get("shape_type") != "mask" or not shape.get("mask"):
            continue

        original_mask = decode_labelme_mask(
            shape["mask"],
            shape.get("points", []),
            image_height,
            image_width,
        )
        refined_mask = refine_mask_with_edges(image, original_mask, config)
        if np.array_equal(refined_mask, original_mask):
            continue

        points, mask_b64 = LabelmeIO.mask_to_labelme_mask(refined_mask)
        shape["points"] = points
        shape["mask"] = mask_b64
        updated_shapes += 1

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "total_shapes": len(shapes),
        "updated_shapes": updated_shapes,
    }


def collect_json_files(input_path: Path) -> List[Path]:
    """Collect Labelme JSON files from a file or directory path."""
    if input_path.is_file():
        return [input_path]
    return sorted([path for path in input_path.rglob("*.json") if path.is_file()])


def build_output_path(
    input_json_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Map an input JSON path to an output path while preserving relative structure."""
    relative_path = input_json_path.relative_to(input_root)
    return output_root / relative_path


def default_output_path(input_path: Path) -> Path:
    """Choose a safe default output location."""
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}.edge_optimized.json")
    return input_path.parent / f"{input_path.name}_edge_optimized"
