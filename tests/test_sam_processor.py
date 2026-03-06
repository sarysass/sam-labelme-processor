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


class TestSAMProcessorRuntimeControls:
    """Tests for runtime control parameters."""

    def test_checkpoint_interval_control(self, tmp_path):
        """Test: Checkpoint is saved by interval and at completion."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        processor = SAMProcessor(
            data_manager=DataManager(data_root),
            sam_wrapper=MagicMock(),
            enable_checkpoint=True,
            checkpoint_interval=3,
        )

        # Not an interval hit, should skip save.
        processor._save_checkpoint(1, 10)
        assert not (data_root / ".processing_checkpoint.json").exists()

        # Interval hit, should save.
        processor._save_checkpoint(3, 10)
        checkpoint_file = data_root / ".processing_checkpoint.json"
        assert checkpoint_file.exists()
        data = json.loads(checkpoint_file.read_text())
        assert data["processed_count"] == 3

        # Final progress should always save.
        processor._save_checkpoint(10, 10)
        data = json.loads(checkpoint_file.read_text())
        assert data["processed_count"] == 10

    def test_preload_images_populates_cache(self, tmp_path):
        """Test: Preload mode caches images up to configured size."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        # Two images
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / "images" / "a.jpg"), image)
        cv2.imwrite(str(data_root / "images" / "b.jpg"), image)

        # Matching bbox json files
        bbox_data = {
            "version": "5.10.1",
            "shapes": [{"label": "worm", "points": [[0, 0], [10, 10]], "shape_type": "rectangle"}],
            "imagePath": "../images/a.jpg",
            "imageHeight": 32,
            "imageWidth": 32,
        }
        with open(data_root / "bbox" / "a.json", "w") as f:
            json.dump(bbox_data, f)
        bbox_data["imagePath"] = "../images/b.jpg"
        with open(data_root / "bbox" / "b.json", "w") as f:
            json.dump(bbox_data, f)

        data_manager = DataManager(data_root)
        items = sorted(data_manager.scan_dataset(), key=lambda x: x.image_id)

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=MagicMock(),
            preload_images=True,
            image_cache_size=1,
            num_workers=1,
        )

        processor._preload_batch_images(items)
        assert len(processor._image_cache) == 1

    def test_memory_watch_fallback_decision(self, tmp_path):
        """Test: Memory watch forces single worker when above limit."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        processor = SAMProcessor(
            data_manager=DataManager(data_root),
            sam_wrapper=MagicMock(),
            enable_memory_watch=True,
            memory_limit_gb=1.0,
        )

        processor._get_memory_usage_gb = lambda: 2.0
        assert processor._should_fallback_to_single_worker() is True

        processor._get_memory_usage_gb = lambda: 0.5
        assert processor._should_fallback_to_single_worker() is False

    def test_resume_does_not_double_offset_after_slicing(self, tmp_path):
        """Test: Resume mode processes all remaining items without skipping."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        bbox_data = {
            "version": "5.10.1",
            "shapes": [{"label": "worm", "points": [[1, 1], [8, 8]], "shape_type": "rectangle"}],
            "imagePath": "../images/0.jpg",
            "imageHeight": 16,
            "imageWidth": 16,
        }

        for i in range(6):
            name = f"{i}.jpg"
            cv2.imwrite(str(data_root / "images" / name), image)
            bbox_data["imagePath"] = f"../images/{name}"
            with open(data_root / "bbox" / f"{i}.json", "w") as f:
                json.dump(bbox_data, f)

        (data_root / ".processing_checkpoint.json").write_text(
            json.dumps({"processed_count": 2, "total_count": 6})
        )

        data_manager = DataManager(data_root)
        data_items = sorted(data_manager.scan_dataset(), key=lambda x: x.image_id)

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=MagicMock(),
            batch_size=2,
            num_workers=1,
            enable_checkpoint=True,
            enable_resume=True,
        )

        processed_ids = []

        def _fake_process(item):
            processed_ids.append(item.image_id)
            return ProcessingResult(data_item=item, success=True)

        processor._process_single_item = _fake_process

        results = processor.process_batch(data_items, show_progress=False)

        assert len(results) == 4
        assert processed_ids == ["2", "3", "4", "5"]

    def test_resume_ignores_stale_checkpoint_when_item_count_mismatch(self, tmp_path):
        """Test: Stale checkpoint is ignored when current data size differs."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        bbox_data = {
            "version": "5.10.1",
            "shapes": [{"label": "worm", "points": [[1, 1], [8, 8]], "shape_type": "rectangle"}],
            "imagePath": "../images/0.jpg",
            "imageHeight": 16,
            "imageWidth": 16,
        }

        for i in range(3):
            name = f"{i}.jpg"
            cv2.imwrite(str(data_root / "images" / name), image)
            bbox_data["imagePath"] = f"../images/{name}"
            with open(data_root / "bbox" / f"{i}.json", "w") as f:
                json.dump(bbox_data, f)

        # Simulate stale checkpoint from an older run with different dataset size.
        (data_root / ".processing_checkpoint.json").write_text(
            json.dumps({"processed_count": 5, "total_count": 10})
        )

        data_manager = DataManager(data_root)
        data_items = sorted(data_manager.scan_dataset(), key=lambda x: x.image_id)

        processor = SAMProcessor(
            data_manager=data_manager,
            sam_wrapper=MagicMock(),
            batch_size=2,
            num_workers=1,
            enable_checkpoint=True,
            enable_resume=True,
        )

        processed_ids = []

        def _fake_process(item):
            processed_ids.append(item.image_id)
            return ProcessingResult(data_item=item, success=True)

        processor._process_single_item = _fake_process

        results = processor.process_batch(data_items, show_progress=False)

        assert len(results) == 3
        assert processed_ids == ["0", "1", "2"]
