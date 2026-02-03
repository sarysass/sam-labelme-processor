"""
Tests for SAM wrapper module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import the module to test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.models.sam_wrapper import SAMWrapper


class TestSAMWrapper:
    """Tests for SAMWrapper class."""

    @pytest.fixture
    def mock_sam_predictor(self):
        """Mock MicroHunter's SAM Predictor."""
        with patch('src.models.sam_wrapper.UltralyticsSAMPredictor') as MockPredictor:
            # Create mock instance
            mock_instance = MagicMock()

            # Set mock return values
            mock_result = Mock()
            mock_result.masks = Mock()
            mock_result.masks.data = np.array([
                np.random.randint(0, 2, (100, 100), dtype=np.uint8)
            ])
            mock_result.masks.xy = [
                np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
            ]

            mock_instance.predict.return_value = {
                'masks': mock_result.masks.data,
                'contours': mock_result.masks.xy
            }

            MockPredictor.return_value = mock_instance
            yield MockPredictor, mock_instance

    def test_initialization(self, mock_sam_predictor):
        """Test: Initialization parameters."""
        wrapper = SAMWrapper(
            weights="/path/to/sam.pt",
            device="cuda:0",
            imgsz=512,
            iou_threshold=0.5
        )

        assert wrapper.weights == "/path/to/sam.pt"
        assert wrapper.device == "cuda:0"
        assert wrapper.imgsz == 512
        assert wrapper.iou_threshold == 0.5
        assert wrapper.predictor is None  # Lazy loading

    def test_load_model(self, mock_sam_predictor):
        """Test: Load model."""
        MockPredictor, mock_instance = mock_sam_predictor

        wrapper = SAMWrapper(weights="weights/sam.pt")
        wrapper.load_model()

        # Verify UltralyticsSAMPredictor was called
        MockPredictor.assert_called_once_with(
            weights="weights/sam.pt",
            device=None
        )
        assert wrapper.predictor is mock_instance

    def test_predict_single_bbox(self, mock_sam_predictor):
        """Test: Single bbox inference."""
        MockPredictor, mock_instance = mock_sam_predictor

        wrapper = SAMWrapper()

        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = [[10, 10, 50, 50]]

        results = wrapper.predict(image, bboxes)

        # Verify results
        assert len(results) == 1
        assert 'mask' in results[0]
        assert 'contour' in results[0]
        assert 'bbox' in results[0]
        assert results[0]['bbox'] == [10, 10, 50, 50]

        # Verify predict was called
        mock_instance.predict.assert_called_once()
        call_args = mock_instance.predict.call_args
        assert call_args[1]['source'] is image
        assert call_args[1]['bboxes'] == bboxes

    def test_predict_multiple_bboxes(self, mock_sam_predictor):
        """Test: Multiple bbox inference."""
        MockPredictor, mock_instance = mock_sam_predictor

        # Set multiple results mock
        mock_result = Mock()
        mock_result.masks = Mock()
        mock_result.masks.data = np.array([
            np.random.randint(0, 2, (100, 100), dtype=np.uint8),
            np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        ])
        mock_result.masks.xy = [
            np.array([[10, 10], [20, 10], [20, 20]]),
            np.array([[30, 30], [40, 30], [40, 40]])
        ]

        mock_instance.predict.return_value = {
            'masks': mock_result.masks.data,
            'contours': mock_result.masks.xy
        }

        wrapper = SAMWrapper()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = [[10, 10, 30, 30], [40, 40, 60, 60]]

        results = wrapper.predict(image, bboxes)

        assert len(results) == 2
        assert results[0]['bbox'] == [10, 10, 30, 30]
        assert results[1]['bbox'] == [40, 40, 60, 60]

    def test_context_manager(self, mock_sam_predictor):
        """Test: Context manager."""
        MockPredictor, mock_instance = mock_sam_predictor

        with SAMWrapper() as wrapper:
            assert wrapper.predictor is mock_instance

        # Verify model was loaded
        MockPredictor.assert_called_once()

    def test_predict_without_loading(self, mock_sam_predictor):
        """Test: Auto-load model when not loaded."""
        MockPredictor, mock_instance = mock_sam_predictor

        wrapper = SAMWrapper()
        assert wrapper.predictor is None

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bboxes = [[10, 10, 50, 50]]

        # Should auto-load model
        results = wrapper.predict(image, bboxes)

        MockPredictor.assert_called_once()
        assert len(results) == 1
