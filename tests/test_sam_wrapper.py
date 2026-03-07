"""
Tests for SAM wrapper module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.sam_wrapper import SAMWrapper


class TestSAMWrapper:
    """Tests for SAMWrapper facade behavior."""

    def test_initialization(self):
        """Test: Initialization stores constructor parameters and lazy state."""
        backend = MagicMock()
        wrapper = SAMWrapper(
            weights="/path/to/sam.pt",
            device="cuda:0",
            imgsz=512,
            iou_threshold=0.5,
            backend=backend,
        )

        assert wrapper.weights == "/path/to/sam.pt"
        assert wrapper.device == "cuda:0"
        assert wrapper.imgsz == 512
        assert wrapper.iou_threshold == 0.5
        assert wrapper.predictor is None

    def test_load_model(self):
        """Test: load_model delegates to the configured backend."""
        backend = MagicMock()
        wrapper = SAMWrapper(weights="weights/sam.pt", backend=backend)

        wrapper.load_model()

        backend.load_model.assert_called_once_with()
        assert wrapper.predictor is backend

    def test_predict_single_bbox(self):
        """Test: Single-bbox inference goes through the backend."""
        backend = MagicMock()
        backend.predict.return_value = [
            {
                "mask": np.random.randint(0, 2, (100, 100), dtype=np.uint8),
                "contour": np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
                "bbox": [10, 10, 50, 50],
            }
        ]
        wrapper = SAMWrapper(backend=backend)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = [[10, 10, 50, 50]]

        results = wrapper.predict(image, bboxes)

        backend.load_model.assert_called_once_with()
        backend.predict.assert_called_once_with(image, bboxes)
        assert len(results) == 1
        assert results[0]["bbox"] == [10, 10, 50, 50]

    def test_predict_multiple_bboxes(self):
        """Test: Multiple bboxes preserve order through the backend facade."""
        backend = MagicMock()
        backend.predict.return_value = [
            {"mask": np.zeros((10, 10), dtype=np.uint8), "contour": np.array([[1, 1]]), "bbox": [1, 1, 2, 2]},
            {"mask": np.ones((10, 10), dtype=np.uint8), "contour": np.array([[2, 2]]), "bbox": [3, 3, 4, 4]},
        ]
        wrapper = SAMWrapper(backend=backend)
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        bboxes = [[1, 1, 2, 2], [3, 3, 4, 4]]

        results = wrapper.predict(image, bboxes)

        assert len(results) == 2
        assert results[0]["bbox"] == [1, 1, 2, 2]
        assert results[1]["bbox"] == [3, 3, 4, 4]

    def test_context_manager(self):
        """Test: Context manager loads model and returns wrapper."""
        backend = MagicMock()

        with SAMWrapper(backend=backend) as wrapper:
            assert wrapper.predictor is backend

        backend.load_model.assert_called_once_with()

    def test_default_backend_is_local_ultralytics_backend(self):
        """Test: Default wrapper backend is the local Ultralytics backend."""
        with patch("src.models.sam_wrapper.UltralyticsSAMBackend") as backend_cls:
            backend = MagicMock()
            backend_cls.return_value = backend

            wrapper = SAMWrapper(weights="weights/model.pt", device="cpu", imgsz=640, iou_threshold=0.2)

            backend_cls.assert_called_once_with(
                weights="weights/model.pt",
                device="cpu",
                imgsz=640,
                iou_threshold=0.2,
            )
            assert wrapper.backend is backend
