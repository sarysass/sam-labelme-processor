"""
Edge-assisted optimization for Labelme mask annotations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import base64
import json
import logging

import cv2
import numpy as np

from .labelme_io import LabelmeIO

logger = logging.getLogger(__name__)


@dataclass
class EdgeOptimizationConfig:
    """Configuration for edge-assisted mask refinement."""

    search_radius: int = 6
    foreground_erode: int = 4
    background_dilate: int = 2
    gaussian_kernel_size: int = 3
    canny_low_threshold: int = 25
    canny_high_threshold: int = 80
    min_area_ratio: float = 0.3
    max_area_ratio: float = 1.5
    smoothing_kernel_size: int = 5
    smoothing_morph_radius: int = 1


def _odd_kernel_size(kernel_size: int) -> int:
    """Ensure the Gaussian kernel size is a positive odd integer."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _ellipse_kernel(radius: int) -> np.ndarray:
    """Create an elliptical morphology kernel from a pixel radius."""
    radius = max(0, int(radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def decode_labelme_mask(
    mask_b64: str,
    points: List[List[float]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    Decode a Labelme mask payload into a full-size binary mask.

    Args:
        mask_b64: Base64-encoded PNG bytes for the cropped mask.
        points: Bounding points [[x1, y1], [x2, y2]].
        image_height: Output image height.
        image_width: Output image width.

    Returns:
        Full-size binary mask with values 0 or 1.
    """
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


def smooth_binary_mask(
    mask: np.ndarray,
    blur_kernel_size: int = 5,
    morph_radius: int = 1,
) -> np.ndarray:
    """
    Smooth a binary mask to reduce staircase artifacts on the boundary.

    Args:
        mask: Binary mask with values 0 or 1.
        blur_kernel_size: Gaussian blur kernel size.
        morph_radius: Radius for light close/open morphology.

    Returns:
        Smoothed binary mask with values 0 or 1.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0:
        return binary_mask

    blur_kernel_size = _odd_kernel_size(blur_kernel_size)
    blurred = cv2.GaussianBlur(
        binary_mask.astype(np.float32),
        (blur_kernel_size, blur_kernel_size),
        0,
    )
    smoothed = (blurred >= 0.5).astype(np.uint8)

    if morph_radius > 0:
        kernel = _ellipse_kernel(morph_radius)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

    return smoothed


def refine_mask_with_edges(
    image: np.ndarray,
    mask: np.ndarray,
    config: EdgeOptimizationConfig,
) -> np.ndarray:
    """
    Refine a binary mask using a narrow-band, edge-guided watershed step.

    Args:
        image: Source image in BGR format.
        mask: Input binary mask with values 0 or 1.
        config: Optimization parameters.

    Returns:
        Refined binary mask with values 0 or 1.
    """
    original_mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(original_mask > 0)
    if len(xs) == 0:
        return original_mask

    search_radius = max(1, int(config.search_radius))
    x1 = max(0, int(xs.min()) - search_radius)
    y1 = max(0, int(ys.min()) - search_radius)
    x2 = min(original_mask.shape[1], int(xs.max()) + search_radius + 1)
    y2 = min(original_mask.shape[0], int(ys.max()) + search_radius + 1)

    local_mask = original_mask[y1:y2, x1:x2]
    local_image = image[y1:y2, x1:x2]
    if local_mask.size == 0 or local_image.size == 0:
        return original_mask

    if local_image.ndim == 3:
        gray = cv2.cvtColor(local_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = local_image.copy()

    gray = cv2.GaussianBlur(
        gray,
        (_odd_kernel_size(config.gaussian_kernel_size), _odd_kernel_size(config.gaussian_kernel_size)),
        0,
    )
    edges = cv2.Canny(
        gray,
        threshold1=int(config.canny_low_threshold),
        threshold2=int(config.canny_high_threshold),
    )
    if np.count_nonzero(edges) == 0:
        return original_mask

    # Brighten edge pixels to make the watershed boundary more likely to settle there.
    guided_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    guided_image[edges > 0] = (255, 255, 255)

    sure_foreground = cv2.erode(local_mask, _ellipse_kernel(config.foreground_erode))
    allowed_region = cv2.dilate(local_mask, _ellipse_kernel(config.background_dilate))

    markers = np.zeros(local_mask.shape, dtype=np.int32)
    markers[allowed_region == 0] = 1
    markers[sure_foreground > 0] = 2
    markers[0, :] = 1
    markers[-1, :] = 1
    markers[:, 0] = 1
    markers[:, -1] = 1

    markers = cv2.watershed(guided_image.copy(), markers)
    refined_local = (markers == 2).astype(np.uint8)
    refined_local = cv2.morphologyEx(
        refined_local,
        cv2.MORPH_CLOSE,
        _ellipse_kernel(1),
    )
    refined_local = smooth_binary_mask(
        refined_local,
        blur_kernel_size=config.smoothing_kernel_size,
        morph_radius=config.smoothing_morph_radius,
    )

    original_area = int(np.count_nonzero(local_mask))
    refined_area = int(np.count_nonzero(refined_local))
    if original_area == 0 or refined_area == 0:
        return original_mask

    area_ratio = float(refined_area) / float(original_area)
    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return original_mask

    refined_mask = np.zeros_like(original_mask)
    refined_mask[y1:y2, x1:x2] = refined_local
    return refined_mask


def optimize_labelme_mask_file(
    input_json_path: Path,
    output_json_path: Path,
    config: EdgeOptimizationConfig,
) -> Dict[str, int]:
    """
    Optimize all mask shapes in a Labelme JSON file.

    Args:
        input_json_path: Source Labelme JSON path.
        output_json_path: Destination Labelme JSON path.
        config: Optimization parameters.

    Returns:
        Processing statistics.
    """
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
