"""
Tests for DataManager module.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import the module to test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.data_manager import DataManager, DataItem


class TestDataManager:
    """Tests for DataManager class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)

            # Create directory structure
            (data_root / "images").mkdir()
            (data_root / "images" / "subfolder").mkdir()
            (data_root / "bbox").mkdir()
            (data_root / "bbox" / "subfolder").mkdir()
            (data_root / "mask").mkdir()
            (data_root / "mask" / "subfolder").mkdir()

            # Create test files
            (data_root / "images" / "img001.jpg").touch()
            (data_root / "images" / "img002.png").touch()
            (data_root / "images" / "subfolder" / "img003.jpg").touch()
            (data_root / "bbox" / "img001.json").touch()
            (data_root / "bbox" / "subfolder" / "img003.json").touch()
            (data_root / "mask" / "img001.json").touch()

            yield data_root

    def test_scan_all_images(self, temp_data_dir):
        """Test: Scan all images."""
        dm = DataManager(temp_data_dir)

        images = dm.get_image_files()

        assert len(images) == 3
        assert Path("img001.jpg") in images
        assert Path("img002.png") in images
        assert Path("subfolder/img003.jpg") in images

    def test_associate_bbox_files(self, temp_data_dir):
        """Test: Associate bbox files."""
        dm = DataManager(temp_data_dir)

        # img001 has bbox
        bbox_path = dm.get_bbox_path(Path("img001.jpg"))
        assert bbox_path.exists()
        assert bbox_path.name == "img001.json"

        # img002 has no bbox
        bbox_path = dm.get_bbox_path(Path("img002.png"))
        assert not bbox_path.exists()

    def test_check_processing_status(self, temp_data_dir):
        """Test: Check processing status."""
        dm = DataManager(temp_data_dir)
        items = dm.scan_dataset()

        # Find each item
        item_map = {item.image_id: item for item in items}

        # img001: has bbox, has mask -> processed
        assert item_map["img001"].status == "processed"

        # img002: no bbox -> no_bbox
        assert item_map["img002"].status == "no_bbox"

        # img003: has bbox, no mask -> pending
        assert item_map["img003"].status == "pending"

    def test_get_pending_items(self, temp_data_dir):
        """Test: Get pending items."""
        dm = DataManager(temp_data_dir)
        pending = dm.get_pending_items()

        assert len(pending) == 1
        assert pending[0].image_id == "img003"

    def test_get_stats(self, temp_data_dir):
        """Test: Get statistics."""
        dm = DataManager(temp_data_dir)
        stats = dm.get_stats()

        assert stats["total_images"] == 3
        assert stats["with_bbox"] == 2
        assert stats["processed"] == 1
        assert stats["pending"] == 1

    def test_relative_path_calculation(self, temp_data_dir):
        """Test: Relative path calculation."""
        dm = DataManager(temp_data_dir)

        # Test image path
        image_path = temp_data_dir / "images" / "subfolder" / "img003.jpg"
        rel_path = dm._get_relative_image_path(image_path)

        assert rel_path == Path("subfolder/img003.jpg")


class TestDataItem:
    """Tests for DataItem dataclass."""

    def test_data_item_creation(self):
        """Test: Create DataItem."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            bbox_path = Path(f.name)

        try:
            item = DataItem(
                image_path=Path("/data/images/test.jpg"),
                bbox_path=bbox_path,  # Existing bbox file
                mask_path=None,
                relative_path=Path("test.jpg"),
                image_id="test",
            )

            assert item.image_id == "test"
            assert item.status == "pending"  # Has bbox, no mask -> pending
        finally:
            os.unlink(bbox_path)

    def test_data_item_processed_status(self):
        """Test: Processed status."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mask_path = Path(f.name)

        try:
            item = DataItem(
                image_path=Path("/data/images/test.jpg"),
                bbox_path=Path("/data/bbox/test.json"),
                mask_path=mask_path,
                relative_path=Path("test.jpg"),
                image_id="test",
            )

            assert item.status == "processed"
        finally:
            os.unlink(mask_path)
