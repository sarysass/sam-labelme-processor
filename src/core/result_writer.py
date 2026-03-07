"""
Writers for processing outputs.
"""

import os
from pathlib import Path
from typing import List, Optional

from .data_manager import DataItem
from .labelme_io import BBoxShape, ImageInfo, LabelmeIO, MaskShape
from .types import OutputPaths


class ResultWriter:
    """Write processing outputs in the configured Labelme formats."""

    def __init__(
        self,
        output_separate: bool = True,
        output_combine: bool = False,
        combined_dir: Optional[Path] = None,
    ):
        self.output_separate = output_separate
        self.output_combine = output_combine
        self.combined_dir = combined_dir

        if output_combine and combined_dir is not None:
            combined_dir.mkdir(parents=True, exist_ok=True)

    def build_output_image_info(
        self,
        base_info: ImageInfo,
        image_path: Path,
        output_json_path: Path,
    ) -> ImageInfo:
        """Build image info with imagePath relative to the output JSON file."""
        relative_image_path = os.path.relpath(image_path, start=output_json_path.parent)
        return ImageInfo(
            image_path=Path(relative_image_path).as_posix(),
            image_height=base_info.image_height,
            image_width=base_info.image_width,
            version=base_info.version,
            flags=base_info.flags,
            image_data=base_info.image_data,
        )

    def write_outputs(
        self,
        data_item: DataItem,
        image_path: Path,
        image_info: ImageInfo,
        bbox_shapes: List[BBoxShape],
        mask_shapes: List[MaskShape],
    ) -> OutputPaths:
        """Write configured outputs for one processed item."""
        output_paths = OutputPaths()

        if self.output_separate and data_item.mask_path is not None:
            mask_json_path = data_item.mask_path
            mask_json_path.parent.mkdir(parents=True, exist_ok=True)
            mask_image_info = self.build_output_image_info(
                base_info=image_info,
                image_path=image_path,
                output_json_path=mask_json_path,
            )
            LabelmeIO.write_mask_json(mask_json_path, mask_shapes, mask_image_info)
            output_paths.mask_json_path = mask_json_path

        if self.output_combine and self.combined_dir is not None:
            combined_json_path = self.combined_dir / data_item.relative_path.with_suffix(".json")
            combined_json_path.parent.mkdir(parents=True, exist_ok=True)
            combined_image_info = self.build_output_image_info(
                base_info=image_info,
                image_path=image_path,
                output_json_path=combined_json_path,
            )
            LabelmeIO.write_combined_json(
                combined_json_path,
                bbox_shapes,
                mask_shapes,
                combined_image_info,
            )
            output_paths.combined_json_path = combined_json_path

        return output_paths
