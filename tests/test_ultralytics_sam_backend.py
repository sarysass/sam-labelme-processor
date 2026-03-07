"""
Tests for local Ultralytics SAM backend.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.ultralytics_sam_backend import UltralyticsSAMBackend


class _FakeMasks:
    def __init__(self, data=None, contours=None):
        self.data = data
        self.xy = contours or []

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)


class TestUltralyticsSAMBackend:
    """Tests for the local Ultralytics-based backend."""

    def test_load_model_uses_configured_weights(self):
        """Test: Backend constructs the Ultralytics SAM model with configured weights."""
        fake_sam_class = MagicMock()
        with patch("src.models.ultralytics_sam_backend.importlib.import_module") as import_module:
            import_module.return_value = SimpleNamespace(SAM=fake_sam_class)

            backend = UltralyticsSAMBackend(weights="weights/custom.pt", device="cpu")
            backend.load_model()

            fake_sam_class.assert_called_once_with("weights/custom.pt")

    def test_predict_passes_image_and_bboxes_through(self):
        """Test: Predict forwards image and sanitized boxes into Ultralytics."""
        fake_model = MagicMock()
        fake_model.predict.return_value = [
            SimpleNamespace(
                masks=_FakeMasks(
                    data=np.ones((1, 12, 14), dtype=np.uint8),
                    contours=[np.array([[1, 1], [2, 2]])],
                )
            )
        ]

        backend = UltralyticsSAMBackend(weights="weights/custom.pt", device="cpu", imgsz=640, iou_threshold=0.2)
        backend.model = fake_model
        image = np.zeros((12, 14, 3), dtype=np.uint8)
        bboxes = [[1, 2, 10, 11]]

        outputs = backend.predict(image, bboxes)

        fake_model.predict.assert_called_once()
        kwargs = fake_model.predict.call_args.kwargs
        assert kwargs["source"] is image
        assert kwargs["bboxes"] == [[1.0, 2.0, 10.0, 11.0]]
        assert kwargs["imgsz"] == 640
        assert kwargs["device"] == "cpu"
        assert kwargs["iou"] == 0.2
        assert len(outputs) == 1
        assert outputs[0]["bbox"] == [1, 2, 10, 11]

    def test_predict_repairs_invalid_boxes_instead_of_dropping(self):
        """Test: Invalid boxes are converted to tiny valid boxes while preserving order."""
        fake_model = MagicMock()
        fake_model.predict.return_value = [SimpleNamespace(masks=None)]

        backend = UltralyticsSAMBackend()
        backend.model = fake_model
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        bboxes = [[5, 5, 5, 5], [3, 3, 10, 10]]

        outputs = backend.predict(image, bboxes)

        kwargs = fake_model.predict.call_args.kwargs
        assert kwargs["bboxes"][0] == [0.0, 0.0, 1.0, 1.0]
        assert kwargs["bboxes"][1] == [3.0, 3.0, 10.0, 10.0]
        assert [output["bbox"] for output in outputs] == bboxes

    def test_predict_returns_empty_masks_when_no_result_masks_exist(self):
        """Test: Empty mask predictions still return one zero mask per input box."""
        fake_model = MagicMock()
        fake_model.predict.return_value = [SimpleNamespace(masks=None)]

        backend = UltralyticsSAMBackend()
        backend.model = fake_model
        image = np.zeros((9, 11, 3), dtype=np.uint8)
        bboxes = [[1, 1, 2, 2], [3, 3, 4, 4]]

        outputs = backend.predict(image, bboxes)

        assert len(outputs) == 2
        assert outputs[0]["mask"].shape == (9, 11)
        assert outputs[1]["mask"].shape == (9, 11)
        assert np.count_nonzero(outputs[0]["mask"]) == 0
        assert outputs[0]["contour"].shape == (0, 2)

    def test_predict_returns_empty_list_when_no_boxes(self):
        """Test: No input boxes returns no outputs."""
        backend = UltralyticsSAMBackend()
        backend.model = MagicMock()
        image = np.zeros((9, 11, 3), dtype=np.uint8)

        assert backend.predict(image, []) == []
        backend.model.predict.assert_not_called()

    def test_import_failure_raises_clear_error(self):
        """Test: Missing ultralytics dependency raises a helpful ImportError."""
        backend = UltralyticsSAMBackend()
        with patch("src.models.ultralytics_sam_backend.importlib.import_module", side_effect=ImportError("missing")):
            with pytest.raises(ImportError, match="Ultralytics is required for SAM inference"):
                backend.load_model()
