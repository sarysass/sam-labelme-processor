"""
Label file readers for supported bbox formats.
"""

from pathlib import Path
from typing import List

from .labelme_io import BBoxShape, ImageInfo, LabelmeIO
from .types import LoadedAnnotations


class LabelReader:
    """Read supported label file formats into shared bbox objects."""

    def read_yolo_txt_labels(
        self,
        txt_path: Path,
        image_path: Path,
        image_height: int,
        image_width: int,
    ) -> LoadedAnnotations:
        """Read YOLO TXT labels and convert them to bbox shapes."""
        bbox_shapes: List[BBoxShape] = []

        with open(txt_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                parts = stripped.split()
                if "|" in parts[0]:
                    parts = parts[1:]

                if len(parts) < 5:
                    continue

                class_id = parts[0]
                try:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    continue

                x1 = max(0.0, (x_center - width / 2.0) * image_width)
                y1 = max(0.0, (y_center - height / 2.0) * image_height)
                x2 = min(float(image_width - 1), (x_center + width / 2.0) * image_width)
                y2 = min(float(image_height - 1), (y_center + height / 2.0) * image_height)

                class_label = class_id
                try:
                    class_label = str(int(float(class_id)))
                except ValueError:
                    pass

                bbox_shapes.append(
                    BBoxShape(
                        label=f"class_{class_label}",
                        points=[[x1, y1], [x2, y2]],
                    )
                )

        image_info = ImageInfo(
            image_path=image_path.name,
            image_height=image_height,
            image_width=image_width,
        )
        return LoadedAnnotations(image_info=image_info, bbox_shapes=bbox_shapes)

    def read_label_file(
        self,
        label_path: Path,
        image_path: Path,
        image_height: int,
        image_width: int,
    ) -> LoadedAnnotations:
        """Read a supported label file into a shared annotation payload."""
        suffix = label_path.suffix.lower()
        if suffix == ".txt":
            return self.read_yolo_txt_labels(
                txt_path=label_path,
                image_path=image_path,
                image_height=image_height,
                image_width=image_width,
            )

        image_info, bbox_shapes = LabelmeIO.read_bbox_json(label_path)
        return LoadedAnnotations(image_info=image_info, bbox_shapes=bbox_shapes)
