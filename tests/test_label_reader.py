"""
Tests for label reader module.
"""

import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.label_reader import LabelReader


class TestLabelReader:
    """Tests for reading supported bbox label formats."""

    def test_read_yolo_txt_labels(self, tmp_path):
        """Test: Read YOLO TXT labels and convert to bbox shapes."""
        txt_path = tmp_path / "sample.txt"
        txt_path.write_text("0 0.5 0.5 0.4 0.4\n")

        reader = LabelReader()
        loaded = reader.read_yolo_txt_labels(
            txt_path=txt_path,
            image_path=tmp_path / "sample.png",
            image_height=80,
            image_width=100,
        )

        assert loaded.image_info.image_path == "sample.png"
        assert len(loaded.bbox_shapes) == 1
        assert loaded.bbox_shapes[0].label == "class_0"
        assert loaded.bbox_shapes[0].points == [[30.0, 24.0], [70.0, 56.0]]

    def test_read_yolo_txt_labels_ignores_empty_and_invalid_lines(self, tmp_path):
        """Test: Ignore empty lines and malformed YOLO rows."""
        txt_path = tmp_path / "sample.txt"
        txt_path.write_text("\ninvalid\n1 0.5 x 0.4 0.4\n2 0.4 0.6 0.2 0.2\n")

        reader = LabelReader()
        loaded = reader.read_yolo_txt_labels(
            txt_path=txt_path,
            image_path=tmp_path / "sample.png",
            image_height=100,
            image_width=100,
        )

        assert len(loaded.bbox_shapes) == 1
        assert loaded.bbox_shapes[0].label == "class_2"
        assert loaded.bbox_shapes[0].points[0] == pytest.approx([30.0, 50.0])
        assert loaded.bbox_shapes[0].points[1] == pytest.approx([50.0, 70.0])

    def test_read_yolo_txt_labels_strips_prefix_before_pipe(self, tmp_path):
        """Test: Support prefixed YOLO rows containing a pipe delimiter."""
        txt_path = tmp_path / "sample.txt"
        txt_path.write_text("frame|meta 3 0.5 0.5 0.2 0.2\n")

        reader = LabelReader()
        loaded = reader.read_yolo_txt_labels(
            txt_path=txt_path,
            image_path=tmp_path / "sample.png",
            image_height=50,
            image_width=50,
        )

        assert len(loaded.bbox_shapes) == 1
        assert loaded.bbox_shapes[0].label == "class_3"
        assert loaded.bbox_shapes[0].points == [[20.0, 20.0], [30.0, 30.0]]

    def test_read_label_file_reads_labelme_bbox_json(self, tmp_path):
        """Test: Read Labelme rectangle JSON files through the shared interface."""
        json_path = tmp_path / "sample.json"
        json_path.write_text(
            json.dumps(
                {
                    "version": "5.10.1",
                    "shapes": [
                        {
                            "label": "worm",
                            "points": [[10.0, 20.0], [30.0, 40.0]],
                            "shape_type": "rectangle",
                            "group_id": 7,
                        }
                    ],
                    "imagePath": "../images/sample.png",
                    "imageHeight": 60,
                    "imageWidth": 80,
                }
            )
        )

        reader = LabelReader()
        loaded = reader.read_label_file(
            label_path=json_path,
            image_path=tmp_path / "sample.png",
            image_height=60,
            image_width=80,
        )

        assert loaded.image_info.image_path == "../images/sample.png"
        assert len(loaded.bbox_shapes) == 1
        assert loaded.bbox_shapes[0].group_id == 7
