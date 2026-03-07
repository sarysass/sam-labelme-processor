"""
Tests for batch orchestration.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.batch_runner import BatchRunner
from src.core.data_manager import DataManager
from src.core.item_processor import ItemProcessor
from src.core.types import ProcessingResult


class TestBatchRunner:
    """Tests for BatchRunner."""

    def test_checkpoint_interval_control(self, tmp_path):
        """Test: Checkpoint is saved by interval and at completion."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        runner = BatchRunner(
            data_manager=DataManager(data_root),
            item_processor=MagicMock(spec=ItemProcessor),
            enable_checkpoint=True,
            checkpoint_interval=3,
        )

        runner._save_checkpoint(1, 10)
        assert not (data_root / ".processing_checkpoint.json").exists()

        runner._save_checkpoint(3, 10)
        checkpoint_file = data_root / ".processing_checkpoint.json"
        assert checkpoint_file.exists()
        data = json.loads(checkpoint_file.read_text())
        assert data["processed_count"] == 3

        runner._save_checkpoint(10, 10)
        data = json.loads(checkpoint_file.read_text())
        assert data["processed_count"] == 10

    def test_preload_images_populates_cache(self, tmp_path):
        """Test: Preload mode caches images up to configured size."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / "images" / "a.jpg"), image)
        cv2.imwrite(str(data_root / "images" / "b.jpg"), image)

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

        runner = BatchRunner(
            data_manager=data_manager,
            item_processor=MagicMock(spec=ItemProcessor),
            preload_images=True,
            image_cache_size=1,
            num_workers=1,
        )

        runner._preload_batch_images(items)
        assert len(runner._image_cache) == 1

    def test_memory_watch_fallback_decision(self, tmp_path):
        """Test: Memory watch forces single worker when above limit."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        runner = BatchRunner(
            data_manager=DataManager(data_root),
            item_processor=MagicMock(spec=ItemProcessor),
            enable_memory_watch=True,
            memory_limit_gb=1.0,
        )

        runner._get_memory_usage_gb = lambda: 2.0
        assert runner._should_fallback_to_single_worker() is True

        runner._get_memory_usage_gb = lambda: 0.5
        assert runner._should_fallback_to_single_worker() is False

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
        processed_ids = []

        item_processor = MagicMock(spec=ItemProcessor)
        item_processor.process.side_effect = lambda item, image=None: (
            processed_ids.append(item.image_id) or ProcessingResult(data_item=item, success=True)
        )

        runner = BatchRunner(
            data_manager=data_manager,
            item_processor=item_processor,
            batch_size=2,
            num_workers=1,
            enable_checkpoint=True,
            enable_resume=True,
        )

        results = runner.process_batch(data_items, show_progress=False)

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

        (data_root / ".processing_checkpoint.json").write_text(
            json.dumps({"processed_count": 5, "total_count": 10})
        )

        data_manager = DataManager(data_root)
        data_items = sorted(data_manager.scan_dataset(), key=lambda x: x.image_id)
        processed_ids = []

        item_processor = MagicMock(spec=ItemProcessor)
        item_processor.process.side_effect = lambda item, image=None: (
            processed_ids.append(item.image_id) or ProcessingResult(data_item=item, success=True)
        )

        runner = BatchRunner(
            data_manager=data_manager,
            item_processor=item_processor,
            batch_size=2,
            num_workers=1,
            enable_checkpoint=True,
            enable_resume=True,
        )

        results = runner.process_batch(data_items, show_progress=False)

        assert len(results) == 3
        assert processed_ids == ["0", "1", "2"]
