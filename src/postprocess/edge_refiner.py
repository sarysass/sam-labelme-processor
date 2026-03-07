"""
Pure mask refinement algorithms.
"""

from dataclasses import dataclass

import cv2
import numpy as np


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
    enable_cavity_recovery: bool = False
    cavity_min_area: int = 25
    cavity_min_distance: int = 2
    cavity_intensity_margin: float = 5.0
    enable_shell_removal: bool = False
    shell_max_thickness: int = 5
    shell_background_cost_multiplier: float = 1.1


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


def smooth_binary_mask(
    mask: np.ndarray,
    blur_kernel_size: int = 5,
    morph_radius: int = 1,
) -> np.ndarray:
    """Smooth a binary mask to reduce staircase artifacts on the boundary."""
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


def _connected_component_count(mask: np.ndarray) -> int:
    """Count connected foreground components in a binary mask."""
    binary_mask = (mask > 0).astype(np.uint8)
    component_count, _labels = cv2.connectedComponents(binary_mask)
    return max(0, int(component_count) - 1)


def _find_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Find enclosed background holes inside a binary mask."""
    binary_mask = (mask > 0).astype(np.uint8)
    inverse_mask = (binary_mask == 0).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(inverse_mask)
    if component_count <= 1:
        return np.zeros_like(binary_mask)

    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    holes = np.zeros_like(binary_mask)
    for component_id in range(1, component_count):
        if component_id in border_labels:
            continue
        holes[labels == component_id] = 1
    return holes


def recover_internal_cavities(
    image: np.ndarray,
    mask: np.ndarray,
    config: EdgeOptimizationConfig,
) -> np.ndarray:
    """Reopen enclosed background regions that were incorrectly filled inside a mask."""
    binary_mask = (mask > 0).astype(np.uint8)
    if not config.enable_cavity_recovery or np.count_nonzero(binary_mask) == 0:
        return binary_mask

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = gray.astype(np.float32)

    outside_ring = cv2.dilate(binary_mask, _ellipse_kernel(1)) - binary_mask
    if np.count_nonzero(outside_ring) == 0:
        return binary_mask

    background_reference = float(np.median(gray[outside_ring > 0]))

    foreground_seed = cv2.erode(
        binary_mask,
        _ellipse_kernel(max(1, int(config.cavity_min_distance))),
    )
    if np.count_nonzero(foreground_seed) == 0:
        foreground_seed = binary_mask
    foreground_reference = float(np.median(gray[foreground_seed > 0]))

    if abs(background_reference - foreground_reference) < float(config.cavity_intensity_margin):
        return binary_mask

    distance_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    candidate_region = np.logical_and(
        binary_mask > 0,
        distance_inside >= max(1, int(config.cavity_min_distance)),
    )
    if not np.any(candidate_region):
        return binary_mask

    background_distance = np.abs(gray - background_reference)
    foreground_distance = np.abs(gray - foreground_reference)
    cavity_candidates = np.logical_and(
        candidate_region,
        background_distance + float(config.cavity_intensity_margin) < foreground_distance,
    ).astype(np.uint8)
    if np.count_nonzero(cavity_candidates) == 0:
        return binary_mask

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cavity_candidates,
        connectivity=8,
    )
    refined_mask = binary_mask.copy()
    original_components = _connected_component_count(binary_mask)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(config.cavity_min_area):
            continue

        component_mask = labels == component_id
        expanded_component = cv2.dilate(component_mask.astype(np.uint8), _ellipse_kernel(1))
        if np.any(np.logical_and(expanded_component > 0, binary_mask == 0)):
            continue

        removal_mask = component_mask.astype(np.uint8)
        candidate_mask = refined_mask.copy()
        candidate_mask[removal_mask > 0] = 0
        if _connected_component_count(candidate_mask) != original_components:
            continue

        candidate_mask = _remove_background_like_shells(
            current_mask=candidate_mask,
            cavity_mask=removal_mask,
            background_distance=background_distance,
            foreground_distance=foreground_distance,
            original_components=original_components,
            config=config,
        )

        refined_mask = candidate_mask

    return refined_mask


def _remove_background_like_shells(
    current_mask: np.ndarray,
    cavity_mask: np.ndarray,
    background_distance: np.ndarray,
    foreground_distance: np.ndarray,
    original_components: int,
    config: EdgeOptimizationConfig,
) -> np.ndarray:
    """Remove thin shell components that touch both the cavity and the outer background."""
    if not config.enable_shell_removal:
        return current_mask

    binary_mask = (current_mask > 0).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0 or np.count_nonzero(cavity_mask) == 0:
        return binary_mask

    thickness = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    thin_foreground = np.logical_and(
        binary_mask > 0,
        thickness <= float(max(1, int(config.shell_max_thickness))),
    ).astype(np.uint8)
    if np.count_nonzero(thin_foreground) == 0:
        return binary_mask

    cavity_neighbors = np.logical_and(
        cv2.dilate((cavity_mask > 0).astype(np.uint8), _ellipse_kernel(1)) > 0,
        thin_foreground > 0,
    )
    outside_ring = cv2.dilate(binary_mask, _ellipse_kernel(1)) - binary_mask
    outside_neighbors = np.logical_and(
        cv2.dilate(outside_ring.astype(np.uint8), _ellipse_kernel(1)) > 0,
        thin_foreground > 0,
    )
    if not np.any(cavity_neighbors) or not np.any(outside_neighbors):
        return binary_mask

    background_like_pixels = np.logical_and(
        thin_foreground > 0,
        background_distance * float(config.shell_background_cost_multiplier)
        < foreground_distance + float(config.cavity_intensity_margin),
    ).astype(np.uint8)
    if np.count_nonzero(background_like_pixels) == 0:
        return binary_mask

    component_count, labels, _stats, _ = cv2.connectedComponentsWithStats(
        background_like_pixels,
        connectivity=8,
    )
    refined_mask = binary_mask.copy()
    for component_id in range(1, component_count):
        component_mask = labels == component_id
        expanded_component = cv2.dilate(component_mask.astype(np.uint8), _ellipse_kernel(1))
        if not np.any(np.logical_and(expanded_component > 0, cavity_neighbors)):
            continue
        if not np.any(np.logical_and(expanded_component > 0, outside_neighbors)):
            continue

        component_background_distance = float(np.mean(background_distance[component_mask]))
        component_foreground_distance = float(np.mean(foreground_distance[component_mask]))
        if (
            component_background_distance * float(config.shell_background_cost_multiplier)
            >= component_foreground_distance + float(config.cavity_intensity_margin)
        ):
            continue

        candidate_mask = refined_mask.copy()
        candidate_mask[component_mask] = 0
        if _connected_component_count(candidate_mask) != original_components:
            continue
        refined_mask = candidate_mask

    return refined_mask


def refine_mask_with_edges(
    image: np.ndarray,
    mask: np.ndarray,
    config: EdgeOptimizationConfig,
) -> np.ndarray:
    """Refine a binary mask using narrow-band edge guidance."""
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
    if np.count_nonzero(edges) == 0 and not config.enable_cavity_recovery:
        return original_mask

    refined_local = local_mask.copy()
    if np.count_nonzero(edges) > 0:
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
    refined_local = recover_internal_cavities(local_image, refined_local, config)

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
