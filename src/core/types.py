"""
Shared workflow data types.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .data_manager import DataItem
from .labelme_io import BBoxShape, ImageInfo, MaskShape


@dataclass
class LoadedAnnotations:
    """Loaded annotation payload for a single image."""

    image_info: ImageInfo
    bbox_shapes: List[BBoxShape]


@dataclass
class OutputPaths:
    """Resolved output paths for one processed item."""

    mask_json_path: Optional[Path] = None
    combined_json_path: Optional[Path] = None


@dataclass
class ProcessingResult:
    """Result of processing a single data item."""

    data_item: DataItem
    success: bool
    mask_shapes: List[MaskShape] = field(default_factory=list)
    error_message: Optional[str] = None
