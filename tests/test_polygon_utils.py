"""
Tests for polygon utils module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.polygon_utils import (
    calculate_curvature,
    simplify_polygon_adaptive,
    simplify_contour_to_max_points,
)


class TestCalculateCurvature:
    """Tests for calculate_curvature function."""

    def test_empty_points(self):
        """Test: Empty points returns zero array."""
        points = []
        curvatures = calculate_curvature(points)
        assert len(curvatures) == 0

    def test_insufficient_points(self):
        """Test: Less than 3 points returns zero array."""
        points = [[0, 0], [1, 1]]
        curvatures = calculate_curvature(points)
        assert len(curvatures) == 2
        assert np.allclose(curvatures, 0)

    def test_square_points(self):
        """Test: Square corners have high curvature."""
        points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        curvatures = calculate_curvature(points, window_size=1)

        assert len(curvatures) == 4

    def test_circle_points(self):
        """Test: Circle points have consistent curvature."""
        points = []
        center = (0.5, 0.5)
        radius = 0.5
        for i in range(20):
            angle = 2 * np.pi * i / 20
            points.append(
                [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)]
            )

        curvatures = calculate_curvature(points, window_size=3)

        assert len(curvatures) == 20

        assert np.mean(curvatures) > 0


class TestSimplifyPolygonAdaptive:
    """Tests for simplify_polygon_adaptive function."""

    def test_empty_contour(self):
        """Test: Empty contour returns empty."""
        contour = np.array([]).reshape(0, 2)
        simplified = simplify_polygon_adaptive(contour)
        assert len(simplified) == 0

    def test_few_points(self):
        """Test: Few points (< min_points) are preserved."""
        contour = np.array([[[0, 0]], [[1, 0]], [[2, 0]]])
        simplified = simplify_polygon_adaptive(contour, min_points=5)
        assert len(simplified) <= len(contour)

    def test_rectangle_preserves_corners(self):
        """Test: Rectangle simplification preserves corners."""
        contour = np.array([[[0, 0]], [[100, 0]], [[100, 50]], [[0, 50]]])
        simplified = simplify_polygon_adaptive(
            contour,
            base_epsilon_factor=0.001,
            min_points=3,
            max_points=10,
        )
        assert len(simplified) <= len(contour)
        assert len(simplified) >= 3

    def test_many_points_reduced(self):
        """Test: Dense contour is simplified."""
        points = []
        for i in range(100):
            angle = 2 * np.pi * i / 100
            points.append([50 + 40 * np.cos(angle), 50 + 40 * np.sin(angle)])

        contour = np.array(points).reshape(-1, 1, 2)
        simplified = simplify_polygon_adaptive(
            contour,
            base_epsilon_factor=0.005,
            min_points=8,
            max_points=50,
        )

        assert len(simplified) < len(contour)
        assert len(simplified) >= 8
        assert len(simplified) <= 50

    def test_straight_line_simplified(self):
        """Test: Straight line is heavily simplified."""
        points = [[i, 0] for i in range(100)]
        contour = np.array(points).reshape(-1, 1, 2)
        simplified = simplify_polygon_adaptive(
            contour,
            base_epsilon_factor=0.001,
            min_points=2,
            max_points=50,
        )

        assert len(simplified) < len(contour)
        assert len(simplified) <= 50


class TestSimplifyContourToMaxPoints:
    """Tests for simplify_contour_to_max_points function."""

    def test_empty_contour(self):
        """Test: Empty contour returns empty."""
        contour = np.array([]).reshape(0, 2)
        simplified = simplify_contour_to_max_points(contour)
        assert len(simplified) == 0

    def test_few_points_preserved(self):
        """Test: Few points are preserved."""
        contour = np.array([[[0, 0]], [[1, 0]], [[2, 0]], [[3, 0]]])
        simplified = simplify_contour_to_max_points(contour, max_points=10)
        assert len(simplified) <= len(contour)

    def test_max_points_target(self):
        """Test: Result is close to max_points."""
        points = []
        for i in range(200):
            angle = 2 * np.pi * i / 200
            points.append([50 + 40 * np.cos(angle), 50 + 40 * np.sin(angle)])

        contour = np.array(points).reshape(-1, 1, 2)
        simplified = simplify_contour_to_max_points(contour, max_points=50)

        assert len(simplified) <= 50
        assert len(simplified) > 0

    def test_tolerance_affects_result(self):
        """Test: Different min_tolerance affects result."""
        points = []
        for i in range(100):
            angle = 2 * np.pi * i / 100
            points.append([50 + 40 * np.cos(angle), 50 + 40 * np.sin(angle)])

        contour = np.array(points).reshape(-1, 1, 2)

        simplified1 = simplify_contour_to_max_points(
            contour, max_points=30, min_tolerance=0.1
        )
        simplified2 = simplify_contour_to_max_points(
            contour, max_points=30, min_tolerance=1.0
        )

        assert len(simplified1) > 0
        assert len(simplified2) > 0
        assert len(simplified2) <= len(simplified1)


class TestIntegration:
    """Integration tests for polygon simplification."""

    def test_circle_simplification(self):
        """Test: Circle simplification with adaptive method."""
        points = []
        for i in range(150):
            angle = 2 * np.pi * i / 150
            points.append([50 + 40 * np.cos(angle), 50 + 40 * np.sin(angle)])

        contour = np.array(points).reshape(-1, 1, 2)
        simplified = simplify_polygon_adaptive(
            contour,
            base_epsilon_factor=0.003,
            adaptive_factor=0.6,
            min_points=10,
            max_points=60,
            curvature_window=5,
        )

        reduction_ratio = len(simplified) / len(contour)
        assert 0.05 < reduction_ratio < 0.5
        assert 10 <= len(simplified) <= 60

    def test_irregular_shape_simplification(self):
        """Test: Irregular shape simplification."""
        points = [
            [0, 0],
            [10, 0],
            [20, 0],
            [30, 0],
            [40, 0],
            [50, 0],
            [50, 10],
            [50, 20],
            [50, 30],
            [50, 40],
            [50, 50],
            [40, 50],
            [30, 50],
            [20, 50],
            [10, 50],
            [0, 50],
            [0, 40],
            [0, 30],
            [0, 20],
            [0, 10],
        ]

        contour = np.array(points).reshape(-1, 1, 2)
        simplified = simplify_polygon_adaptive(
            contour,
            base_epsilon_factor=0.005,
            adaptive_factor=0.5,
            min_points=4,
            max_points=20,
        )

        assert len(simplified) < len(contour)
        assert len(simplified) >= 4
        assert len(simplified) <= 20
