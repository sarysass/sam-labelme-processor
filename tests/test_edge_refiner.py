"""
Tests for edge refiner module.
"""

from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.postprocess.edge_refiner import (
    EdgeOptimizationConfig,
    refine_mask_with_edges,
    smooth_binary_mask,
)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a > 0
    mask_b = mask_b > 0
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection) / float(union)


class TestEdgeRefiner:
    """Tests for pure refinement algorithms."""

    def test_smooth_binary_mask_reduces_boundary_complexity(self):
        """Test: Smoothing preserves area roughly while simplifying edges."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        for offset in range(12):
            x_start = 6 + offset
            x_end = 17 + offset + (offset % 2)
            y_start = 6 + offset
            y_end = y_start + 2
            mask[y_start:y_end, x_start:x_end] = 1

        smoothed = smooth_binary_mask(mask, blur_kernel_size=5, morph_radius=1)

        assert np.count_nonzero(smoothed) > 0
        area_ratio = float(np.count_nonzero(smoothed)) / float(np.count_nonzero(mask))
        assert 0.8 <= area_ratio <= 1.2

    def test_refine_mask_with_edges_improves_iou_on_synthetic_image(self):
        """Test: Edge-guided refinement improves a simple synthetic mask."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        target_mask = np.zeros((100, 100), dtype=np.uint8)
        input_mask = np.zeros((100, 100), dtype=np.uint8)

        cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.rectangle(target_mask, (30, 30), (70, 70), 1, -1)
        cv2.rectangle(input_mask, (26, 26), (66, 66), 1, -1)

        config = EdgeOptimizationConfig(
            search_radius=8,
            foreground_erode=2,
            background_dilate=4,
            gaussian_kernel_size=5,
            canny_low_threshold=50,
            canny_high_threshold=150,
            min_area_ratio=0.5,
            max_area_ratio=1.5,
        )

        refined_mask = refine_mask_with_edges(image, input_mask, config)

        assert _iou(refined_mask, target_mask) > _iou(input_mask, target_mask)
