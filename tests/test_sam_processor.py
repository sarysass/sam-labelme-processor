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

from src.core.sam_processor import SAMProcessor
from src.core.batch_runner import BatchRunner
from src.core.item_processor import ItemProcessor
from src.core.data_manager import DataManager, DataItem
from src.core.labelme_io import MaskShape
from src.core.types import ProcessingResult


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
        assert result.mask_shapes[0].label == "mask"
        assert result.data_item.image_id == "test"

        # Verify mask file was created
        assert (data_root / "mask" / "test.json").exists()

    def test_process_single_yolo_txt_and_mask_output(self, tmp_path):
        """Test: Read YOLO TXT input and write Labelme mask output."""
        data_root = tmp_path
        images_dir = "select images"
        labels_dir = "select labels"
        masks_dir = "select masks"

        (data_root / images_dir).mkdir(parents=True)
        (data_root / labels_dir).mkdir(parents=True)
        (data_root / masks_dir).mkdir(parents=True)

        image = np.zeros((80, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / images_dir / "sample.png"), image)
        (data_root / labels_dir / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n")

        sam_mask = np.zeros((80, 100), dtype=np.uint8)
        sam_mask[20:50, 30:70] = 1
        mock_sam = MagicMock()
        mock_sam.predict.return_value = [
            {
                "mask": sam_mask,
                "contour": np.array([[30, 20], [69, 20], [69, 49], [30, 49]]),
                "bbox": [30, 20, 70, 50],
            }
        ]

        data_manager = DataManager(
            data_root=data_root,
            images_dir=images_dir,
            bbox_dir=labels_dir,
            mask_dir=masks_dir,
            bbox_extension=".txt",
        )
        items = data_manager.scan_dataset()

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam,
            output_separate=True,
            num_workers=1,
        )

        result = processor.process_single(items[0])
        assert result.success is True

        output_json = data_root / masks_dir / "sample.json"
        assert output_json.exists()

        data = json.loads(output_json.read_text())
        assert len(data["shapes"]) == 1
        assert data["shapes"][0]["label"] == "mask"
        assert data["shapes"][0]["shape_type"] == "mask"
        assert len(data["shapes"][0]["points"]) == 2
        assert isinstance(data["shapes"][0]["mask"], str)
        assert len(data["shapes"][0]["mask"]) > 0
        assert data["imagePath"] == "../select images/sample.png"

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

    def test_process_single_empty_labels_are_skipped_when_enabled(self, tmp_path, mock_sam_wrapper):
        """Test: Empty label files are treated as skipped success when enabled."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / "images" / "empty.jpg"), image)
        (data_root / "bbox" / "empty.json").write_text(
            json.dumps(
                {
                    "version": "5.10.1",
                    "shapes": [],
                    "imagePath": "../images/empty.jpg",
                    "imageHeight": 32,
                    "imageWidth": 32,
                }
            )
        )

        data_manager = DataManager(data_root)
        item = data_manager.scan_dataset()[0]

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam_wrapper,
            skip_empty_labels=True,
        )

        result = processor.process_single(item)

        assert result.success is True
        assert result.mask_shapes == []
        assert not (data_root / "mask" / "empty.json").exists()

    def test_process_single_image_not_found(self, temp_data_setup, mock_sam_wrapper):
        """Test: Fail when image doesn't exist."""
        data_root = temp_data_setup

        # Delete image
        (data_root / "images" / "test.jpg").unlink()

        data_manager = DataManager(data_root)
        data_item = DataItem(
            image_path=data_root / "images" / "test.jpg",
            bbox_path=data_root / "bbox" / "test.json",
            mask_path=data_root / "mask" / "test.json",
            relative_path=Path("test.jpg"),
            image_id="test",
        )

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=mock_sam_wrapper
        )

        result = processor.process_single(data_item)

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


class TestSAMProcessorFacade:
    """Tests for the compatibility facade wiring."""

    def test_process_single_delegates_to_item_processor(self, tmp_path):
        """Test: process_single delegates to the configured item processor."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        data_item = DataItem(
            image_path=data_root / "images" / "sample.jpg",
            bbox_path=data_root / "bbox" / "sample.json",
            mask_path=data_root / "mask" / "sample.json",
            relative_path=Path("sample.jpg"),
            image_id="sample",
        )
        expected = ProcessingResult(data_item=data_item, success=True)

        item_processor = MagicMock(spec=ItemProcessor)
        item_processor.process.return_value = expected

        processor = SAMProcessor(
            data_manager=DataManager(data_root),
            sam_wrapper=MagicMock(),
            item_processor=item_processor,
            batch_runner=MagicMock(spec=BatchRunner),
        )

        result = processor.process_single(data_item)

        assert result is expected
        item_processor.process.assert_called_once_with(data_item)

    def test_process_batch_delegates_to_batch_runner(self, tmp_path):
        """Test: process_batch delegates to the configured batch runner."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        data_item = DataItem(
            image_path=data_root / "images" / "sample.jpg",
            bbox_path=data_root / "bbox" / "sample.json",
            mask_path=data_root / "mask" / "sample.json",
            relative_path=Path("sample.jpg"),
            image_id="sample",
        )
        expected = [ProcessingResult(data_item=data_item, success=True)]

        batch_runner = MagicMock(spec=BatchRunner)
        batch_runner.process_batch.return_value = expected

        processor = SAMProcessor(
            data_manager=DataManager(data_root),
            sam_wrapper=MagicMock(),
            item_processor=MagicMock(spec=ItemProcessor),
            batch_runner=batch_runner,
        )

        result = processor.process_batch([data_item], show_progress=False)

        assert result == expected
        batch_runner.process_batch.assert_called_once_with([data_item], show_progress=False)
