"""
Tests for SAM processor module.
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import cv2

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.sam_processor import SAMProcessor, ProcessingResult
from src.core.data_manager import DataManager, DataItem
from src.core.labelme_io import MaskShape


class TestSAMProcessor:
    """Tests for SAMProcessor class."""

    @pytest.fixture
    def mock_sam_wrapper(self):
        """Mock SAM Wrapper."""
        mock = MagicMock()
        mock.predict.return_value = [
            {
                'mask': np.random.randint(0, 2, (100, 100), dtype=np.uint8),
                'contour': np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
                'bbox': [10, 10, 50, 50]
            }
        ]
        return mock

    @pytest.fixture
    def temp_data_setup(self):
        """Create temporary data environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)

            # Create directories
            (data_root / "images").mkdir()
            (data_root / "bbox").mkdir()
            (data_root / "mask").mkdir()
            (data_root / "output" / "combined").mkdir(parents=True)

            # Create test image
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(data_root / "images" / "test.jpg"), image)

            # Create test bbox JSON
            bbox_data = {
                "version": "5.10.1",
                "shapes": [
                    {
                        "label": "worm",
                        "points": [[10.0, 10.0], [50.0, 50.0]],
                        "shape_type": "rectangle",
                        "group_id": 1
                    }
                ],
                "imagePath": "../images/test.jpg",
                "imageHeight": 100,
                "imageWidth": 100
            }
            with open(data_root / "bbox" / "test.json", "w") as f:
                json.dump(bbox_data, f)

            yield data_root

    def test_process_single_success(self, temp_data_setup, mock_sam_wrapper):
        """Test: Successfully process single data item."""
        data_root = temp_data_setup

        data_manager = DataManager(data_root)
        items = data_manager.scan_dataset()

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam_wrapper,
            output_separate=True
        )

        result = processor.process_single(items[0])

        assert result.success is True
        assert len(result.mask_shapes) == 1
        assert result.data_item.image_id == "test"

        # Verify mask file was created
        assert (data_root / "mask" / "test.json").exists()

    def test_process_single_no_bbox(self, temp_data_setup, mock_sam_wrapper):
        """Test: Fail when no bbox file."""
        data_root = temp_data_setup

        # Delete bbox file
        (data_root / "bbox" / "test.json").unlink()

        data_manager = DataManager(data_root)
        items = data_manager.scan_dataset()

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam_wrapper
        )

        result = processor.process_single(items[0])

        assert result.success is False
        assert "BBox file not found" in result.error_message

    def test_process_single_image_not_found(self, temp_data_setup, mock_sam_wrapper):
        """Test: Fail when image doesn't exist."""
        data_root = temp_data_setup

        # Delete image
        (data_root / "images" / "test.jpg").unlink()

        data_manager = DataManager(data_root)
        items = data_manager.scan_dataset()

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam_wrapper
        )

        result = processor.process_single(items[0])

        assert result.success is False
        assert "Failed to load image" in result.error_message

    def test_combined_output(self, temp_data_setup, mock_sam_wrapper):
        """Test: Generate combined output."""
        data_root = temp_data_setup

        data_manager = DataManager(data_root)
        items = data_manager.scan_dataset()

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam_wrapper,
            output_separate=False,
            output_combine=True,
            combined_dir=data_root / "output" / "combined"
        )

        result = processor.process_single(items[0])

        assert result.success is True
        # Verify combined file was created
        assert (data_root / "output" / "combined" / "test.json").exists()

    def test_batch_processing(self, temp_data_setup, mock_sam_wrapper):
        """Test: Batch processing."""
        data_root = temp_data_setup

        # Create second test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / "images" / "test2.jpg"), image)

        bbox_data = {
            "version": "5.10.1",
            "shapes": [{"label": "worm", "points": [[0, 0], [50, 50]], "shape_type": "rectangle"}],
            "imagePath": "../images/test2.jpg",
            "imageHeight": 100,
            "imageWidth": 100
        }
        with open(data_root / "bbox" / "test2.json", "w") as f:
            json.dump(bbox_data, f)

        data_manager = DataManager(data_root)
        pending = data_manager.get_pending_items()

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam_wrapper
        )

        results = processor.process_batch(pending, show_progress=False)

        assert len(results) == 2
        assert sum(r.success for r in results) == 2

        # Verify all mask files were created
        assert (data_root / "mask" / "test.json").exists()
        assert (data_root / "mask" / "test2.json").exists()

    def test_sam_error_handling(self, temp_data_setup):
        """Test: SAM inference error handling."""
        data_root = temp_data_setup

        # Create mock that throws exception
        mock_sam = MagicMock()
        mock_sam.predict.side_effect = Exception("CUDA out of memory")

        data_manager = DataManager(data_root)
        items = data_manager.scan_dataset()

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam
        )

        result = processor.process_single(items[0])

        assert result.success is False
        assert "CUDA out of memory" in result.error_message


class TestProcessingResult:
    """Tests for ProcessingResult class."""

    def test_result_creation(self):
        """Test: Create result object."""
        data_item = DataItem(
            image_path=Path("/test.jpg"),
            bbox_path=None,
            mask_path=None,
            relative_path=Path("test.jpg"),
            image_id="test"
        )

        result = ProcessingResult(
            data_item=data_item,
            success=True,
            mask_shapes=[MaskShape(label="worm", points=[[0, 0], [1, 1]])]
        )

        assert result.success is True
        assert len(result.mask_shapes) == 1
        assert result.error_message is None
