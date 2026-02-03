"""
Data management module for associating images with bbox/mask files.
"""
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataItem:
    """Data item (one image and its associated files)."""

    image_path: Path  # Image file path (absolute path)
    bbox_path: Optional[Path]  # BBox JSON path (may not exist)
    mask_path: Optional[Path]  # Mask JSON path (may not exist)
    relative_path: Path  # Path relative to images/
    image_id: str  # Unique identifier (filename without extension)

    @property
    def status(self) -> str:
        """Return processing status."""
        if self.mask_path and self.mask_path.exists():
            return "processed"
        elif self.bbox_path and self.bbox_path.exists():
            return "pending"
        else:
            return "no_bbox"


class DataManager:
    """
    Manage data association between images/, bbox/, and mask/ directories.

    Directory structure example:
        data/
        ├── images/
        │   ├── img001.jpg
        │   └── subfolder/
        │       └── img002.jpg
        ├── bbox/
        │   ├── img001.json
        │   └── subfolder/
        │       └── img002.json
        └── mask/
            ├── img001.json
            └── subfolder/
                └── img002.json
    """

    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        data_root: Path,
        images_dir: str = "images",
        bbox_dir: str = "bbox",
        mask_dir: str = "mask"
    ):
        """
        Initialize DataManager.

        Args:
            data_root: Root data directory.
            images_dir: Images subdirectory name.
            bbox_dir: BBox subdirectory name.
            mask_dir: Mask subdirectory name.
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / images_dir
        self.bbox_dir = self.data_root / bbox_dir
        self.mask_dir = self.data_root / mask_dir

        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.bbox_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    def get_image_files(self) -> List[Path]:
        """
        Recursively get all image files.

        Returns:
            List of image file paths (relative to images/).
        """
        images = []
        for file_path in self.images_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                images.append(self._get_relative_image_path(file_path))
        return images

    def get_bbox_path(self, relative_image_path: Path) -> Path:
        """
        Get bbox JSON path based on image relative path.

        Args:
            relative_image_path: Path relative to images/, e.g., Path("subfolder/img001.jpg").

        Returns:
            Absolute path to bbox JSON, e.g., data/bbox/subfolder/img001.json.
        """
        return self.bbox_dir / relative_image_path.with_suffix('.json')

    def get_mask_path(self, relative_image_path: Path) -> Path:
        """
        Get mask JSON path based on image relative path.

        Args:
            relative_image_path: Path relative to images/.

        Returns:
            Absolute path to mask JSON.
        """
        return self.mask_dir / relative_image_path.with_suffix('.json')

    def _get_relative_image_path(self, image_path: Path) -> Path:
        """
        Get relative path from absolute image path.

        Args:
            image_path: Absolute path to image.

        Returns:
            Relative path from images/ directory.
        """
        return image_path.relative_to(self.images_dir)

    def scan_dataset(self) -> List[DataItem]:
        """
        Scan the entire dataset.

        Returns:
            List of all data items.
        """
        items = []
        for relative_image_path in self.get_image_files():
            image_id = relative_image_path.stem
            image_path = self.images_dir / relative_image_path
            bbox_path = self.get_bbox_path(relative_image_path)
            mask_path = self.get_mask_path(relative_image_path)

            items.append(DataItem(
                image_path=image_path,
                bbox_path=bbox_path,
                mask_path=mask_path,
                relative_path=relative_image_path,
                image_id=image_id
            ))

        return items

    def get_pending_items(self) -> List[DataItem]:
        """
        Get pending data items (have bbox but no mask).

        Returns:
            List of pending data items.
        """
        all_items = self.scan_dataset()
        pending = [item for item in all_items if item.status == "pending"]
        return pending

    def get_stats(self) -> dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics:
                'total_images': int,
                'with_bbox': int,
                'processed': int,
                'pending': int
        """
        items = self.scan_dataset()

        stats = {
            'total_images': len(items),
            'with_bbox': sum(1 for item in items if item.bbox_path and item.bbox_path.exists()),
            'processed': sum(1 for item in items if item.status == "processed"),
            'pending': sum(1 for item in items if item.status == "pending")
        }

        return stats
