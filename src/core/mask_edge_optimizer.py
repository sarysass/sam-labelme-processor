"""
Compatibility re-exports for mask edge optimization.
"""

from ..postprocess.edge_refiner import (
    EdgeOptimizationConfig,
    _connected_component_count,
    _ellipse_kernel,
    _find_enclosed_holes,
    _odd_kernel_size,
    _remove_background_like_shells,
    recover_internal_cavities,
    refine_mask_with_edges,
    smooth_binary_mask,
)
from ..postprocess.labelme_adapter import (
    build_output_path,
    collect_json_files,
    decode_labelme_mask,
    default_output_path,
    optimize_labelme_mask_file,
)

__all__ = [
    "EdgeOptimizationConfig",
    "_connected_component_count",
    "_ellipse_kernel",
    "_find_enclosed_holes",
    "_odd_kernel_size",
    "_remove_background_like_shells",
    "recover_internal_cavities",
    "refine_mask_with_edges",
    "smooth_binary_mask",
    "build_output_path",
    "collect_json_files",
    "decode_labelme_mask",
    "default_output_path",
    "optimize_labelme_mask_file",
]
