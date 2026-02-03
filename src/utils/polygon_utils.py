"""
Polygon simplification utilities with curvature-based adaptive approach.
"""

from typing import List, Tuple, Optional
import numpy as np


def calculate_curvature(points: List[List[float]], window_size: int = 5) -> np.ndarray:
    """
    Calculate curvature for each point in polygon.

    Args:
        points: List of points [[x, y], ...].
        window_size: Window size for curvature calculation.

    Returns:
        Array of curvature values.
    """
    if len(points) < 3:
        return np.zeros(len(points))

    points_np = np.array(points, dtype=np.float64)
    n = len(points_np)
    curvatures = np.zeros(n)

    for i in range(n):
        prev_idx = (i - window_size) % n
        next_idx = (i + window_size) % n

        prev_pt = points_np[prev_idx]
        curr_pt = points_np[i]
        next_pt = points_np[next_idx]

        vec1 = prev_pt - curr_pt
        vec2 = next_pt - curr_pt

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            dot = np.dot(vec1, vec2)
            cos_angle = np.clip(dot / (norm1 * norm2), -1, 1)
            angle = np.arccos(cos_angle)
            curvatures[i] = angle

    return curvatures


def simplify_polygon_adaptive(
    contour: np.ndarray,
    base_epsilon_factor: float = 0.005,
    adaptive_factor: float = 0.5,
    min_points: int = 8,
    max_points: int = 50,
    curvature_window: int = 5,
) -> np.ndarray:
    """
    Simplify polygon using curvature-based adaptive approach.

    Args:
        contour: OpenCV contour array (N, 1, 2) or (N, 2).
        base_epsilon_factor: Base epsilon factor for Douglas-Peucker.
        adaptive_factor: How much curvature affects simplification (0-1).
        min_points: Minimum number of points to preserve.
        max_points: Maximum number of points to preserve.
        curvature_window: Window size for curvature calculation.

    Returns:
        Simplified contour array.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for polygon simplification")

    if len(contour) == 0:
        return contour

    if len(contour.shape) == 3:
        contour = contour.reshape(-1, 2)

    if len(contour) <= min_points:
        return contour

    contour_32 = contour.astype(np.float32)
    perimeter = cv2.arcLength(contour_32, True)

    if len(contour) > max_points:
        current_points = len(contour)
        epsilon = (perimeter * base_epsilon_factor) * (current_points / max_points) ** 2
        simplified = cv2.approxPolyDP(contour_32, epsilon, True)

        if len(simplified) >= min_points:
            return simplified

    points = contour.tolist()
    curvatures = calculate_curvature(points, curvature_window)

    normalized_curvatures = curvatures / (np.max(curvatures) + 1e-6)

    weights = 1.0 + adaptive_factor * normalized_curvatures

    points_np = np.array(points)
    distances = np.zeros(len(points))

    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        distances[i] = np.linalg.norm(points_np[i] - points_np[next_i])

    weighted_distances = distances / weights

    mean_weighted_dist = np.mean(weighted_distances)
    epsilon = (
        base_epsilon_factor
        * perimeter
        * (1 + adaptive_factor * np.std(weights) / np.mean(weights))
    )

    simplified = cv2.approxPolyDP(contour_32, epsilon, True)

    if len(simplified) < min_points:
        epsilon = base_epsilon_factor * perimeter * (min_points / len(simplified)) ** 2
        simplified = cv2.approxPolyDP(contour_32, epsilon, True)
    elif len(simplified) > max_points:
        epsilon = perimeter * base_epsilon_factor * (len(simplified) / max_points)
        simplified = cv2.approxPolyDP(contour_32, epsilon, True)

    return simplified


def simplify_contour_to_max_points(
    contour: np.ndarray,
    max_points: int = 25,
    min_tolerance: float = 0.1,
    max_tolerance: Optional[float] = None,
) -> np.ndarray:
    """
    Simplify contour to achieve approximately max_points.

    Uses binary search to find appropriate tolerance.

    Args:
        contour: OpenCV contour array.
        max_points: Target maximum number of points.
        min_tolerance: Starting minimum tolerance.
        max_tolerance: Maximum tolerance (None = perimeter).

    Returns:
        Simplified contour array.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for polygon simplification")

    if len(contour) == 0:
        return contour

    if len(contour.shape) == 3:
        contour = contour.reshape(-1, 2)

    if len(contour) <= max_points:
        return contour

    contour_32 = contour.astype(np.float32)
    if max_tolerance is None:
        max_tolerance = cv2.arcLength(contour_32, True) * 0.05

    tolerance = min_tolerance

    while tolerance < max_tolerance:
        simplified = cv2.approxPolyDP(contour_32, tolerance, True)
        if len(simplified) <= max_points:
            return simplified

        tolerance *= 2

    simplified = cv2.approxPolyDP(contour_32, max_tolerance, True)
    return simplified
