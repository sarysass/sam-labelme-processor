"""
Tests for single-item processing.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.data_manager import DataManager
from src.core.item_processor import ItemProcessor
from src.core.label_reader import LabelReader
from src.core.result_writer import ResultWriter


class TestItemProcessor:
    """Tests for ItemProcessor."""

    def test_process_success(self, tmp_path):
        """Test: Process one item successfully."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / "images" / "test.jpg"), image)
        (data_root / "bbox" / "test.json").write_text(
            json.dumps(
                {
                    "version": "5.10.1",
                    "shapes": [
                        {
                            "label": "worm",
                            "points": [[10.0, 10.0], [50.0, 50.0]],
                            "shape_type": "rectangle",
                            "group_id": 1,
                        }
                    ],
                    "imagePath": "../images/test.jpg",
                    "imageHeight": 100,
                    "imageWidth": 100,
                }
            )
        )

        sam_wrapper = MagicMock()
        sam_wrapper.predict.return_value = [
            {
                "mask": np.random.randint(0, 2, (100, 100), dtype=np.uint8),
                "contour": np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
                "bbox": [10, 10, 50, 50],
            }
        ]

        data_manager = DataManager(data_root)
        item = data_manager.scan_dataset()[0]
        processor = ItemProcessor(
            data_manager=data_manager,
            sam_wrapper=sam_wrapper,
            label_reader=LabelReader(),
            result_writer=ResultWriter(output_separate=True),
        )

        result = processor.process(item)

        assert result.success is True
        assert len(result.mask_shapes) == 1
        assert (data_root / "mask" / "test.json").exists()

    def test_process_empty_labels_are_skipped(self, tmp_path):
        """Test: Empty bbox shapes are treated as skipped success."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        image = np.zeros((20, 20, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / "images" / "empty.jpg"), image)
        (data_root / "bbox" / "empty.json").write_text(
            json.dumps(
                {
                    "version": "5.10.1",
                    "shapes": [],
                    "imagePath": "../images/empty.jpg",
                    "imageHeight": 20,
                    "imageWidth": 20,
                }
            )
        )

        data_manager = DataManager(data_root)
        item = data_manager.scan_dataset()[0]
        processor = ItemProcessor(
            data_manager=data_manager,
            sam_wrapper=MagicMock(),
            label_reader=LabelReader(),
            result_writer=ResultWriter(output_separate=True),
            skip_empty_labels=True,
        )

        result = processor.process(item)

        assert result.success is True
        assert result.mask_shapes == []

    def test_process_missing_bbox_returns_failure(self, tmp_path):
        """Test: Missing bbox file returns a failure result."""
        data_root = tmp_path
        (data_root / "images").mkdir()
        (data_root / "bbox").mkdir()
        (data_root / "mask").mkdir()

        image = np.zeros((20, 20, 3), dtype=np.uint8)
        cv2.imwrite(str(data_root / "images" / "missing.jpg"), image)

        data_manager = DataManager(data_root)
        item = data_manager.scan_dataset()[0]
        processor = ItemProcessor(
            data_manager=data_manager,
            sam_wrapper=MagicMock(),
            label_reader=LabelReader(),
            result_writer=ResultWriter(output_separate=True),
        )

        result = processor.process(item)

        assert result.success is False
        assert "BBox file not found" in result.error_message
