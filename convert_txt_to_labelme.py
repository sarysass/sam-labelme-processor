#!/usr/bin/env python3
"""
Convert YOLO format TXT files to Labelme JSON format.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import base64

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
from src.core.labelme_io import ImageInfo, BBoxShape


class YOLOToLabelmeConverter:
    """Convert YOLO TXT to Labelme JSON."""

    def __init__(self, labels_dir: Path, images_dir: Path, output_dir: Path):
        """
        Initialize converter.

        Args:
            labels_dir: Directory containing YOLO TXT files.
            images_dir: Directory containing corresponding images.
            output_dir: Directory to save Labelme JSON files.
        """
        self.labels_dir = Path(labels_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """
        Get image dimensions.

        Args:
            image_path: Path to image file.

        Returns:
            Tuple of (width, height).
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path)
        return img.width, img.height

    def encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to image file.

        Returns:
            Base64 encoded string.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def parse_yolo_txt(self, txt_path: Path) -> List[Dict[str, Any]]:
        """
        Parse YOLO TXT file.

        Args:
            txt_path: Path to YOLO TXT file.

        Returns:
            List of bounding box dicts.
        """
        bboxes = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                # Skip number prefix (e.g., "00001|")
                if "|" in parts[0]:
                    parts = parts[1:]

                # Format: class_id x_center y_center width height (normalized)
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                bboxes.append(
                    {
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                    }
                )
        return bboxes

    def normalize_bbox_to_points(
        self, bbox: Dict[str, Any], img_width: int, img_height: int
    ) -> List[List[float]]:
        """
        Convert normalized YOLO bbox to Labelme rectangle points.

        Args:
            bbox: Bounding box dict.
            img_width: Image width.
            img_height: Image height.

        Returns:
            List of points [[x1, y1], [x2, y2]].
        """
        x_center = bbox["x_center"]
        y_center = bbox["y_center"]
        w = bbox["width"]
        h = bbox["height"]

        x1 = (x_center - w / 2) * img_width
        y1 = (y_center - h / 2) * img_height
        x2 = (x_center + w / 2) * img_width
        y2 = (y_center + h / 2) * img_height

        return [[x1, y1], [x2, y2]]

    def convert_single_file(self, txt_path: Path) -> bool:
        """
        Convert single TXT file to Labelme JSON.

        Args:
            txt_path: Path to TXT file.

        Returns:
            True if conversion succeeded.
        """
        # Find corresponding image
        image_name = txt_path.stem
        image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        image_path = None

        for ext in image_extensions:
            potential_path = self.images_dir / (image_name + ext)
            if potential_path.exists():
                image_path = potential_path
                break

        if not image_path:
            print(f"Warning: No image found for {txt_path.stem}")
            return False

        try:
            # Get image dimensions
            img_width, img_height = self.get_image_dimensions(image_path)

            # Parse YOLO TXT
            bboxes = self.parse_yolo_txt(txt_path)

            # Convert to Labelme bbox shapes
            bbox_shapes = []
            for bbox in bboxes:
                points = self.normalize_bbox_to_points(bbox, img_width, img_height)
                bbox_shapes.append(
                    BBoxShape(
                        label=f"class_{bbox['class_id']}",
                        points=points,
                    )
                )

            # Create image info
            image_info = ImageInfo(
                image_path=image_path.name,
                image_height=img_height,
                image_width=img_width,
            )

            # Encode image to base64
            image_data = self.encode_image_to_base64(image_path)

            # Write bbox shapes (manual write for rectangle type)
            shapes_dict = []
            for shape in bbox_shapes:
                shapes_dict.append(
                    {
                        "label": shape.label,
                        "points": shape.points,
                        "group_id": shape.group_id,
                        "shape_type": "rectangle",
                        "description": shape.description,
                        "flags": shape.flags,
                    }
                )

            # Create Labelme JSON with imageData
            output_path = self.output_dir / f"{image_name}.json"
            data = {
                "version": image_info.version,
                "flags": image_info.flags or {},
                "shapes": shapes_dict,
                "imagePath": image_info.image_path,
                "imageData": image_data,
                "imageHeight": image_info.image_height,
                "imageWidth": image_info.image_width,
            }

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error converting {txt_path.name}: {e}")
            return False

    def convert_all(self) -> Dict[str, int]:
        """
        Convert all TXT files.

        Returns:
            Statistics dict.
        """
        txt_files = list(self.labels_dir.glob("*.txt"))
        stats = {"total": len(txt_files), "success": 0, "failed": 0}

        print(f"Found {stats['total']} TXT files")

        for i, txt_path in enumerate(txt_files, 1):
            if self.convert_single_file(txt_path):
                stats["success"] += 1
            else:
                stats["failed"] += 1

            if i % 100 == 0:
                print(f"Processed {i}/{stats['total']} files")

        return stats


if __name__ == "__main__":
    labels_dir = Path("data/Benchmark_Dataset/labels")
    images_dir = Path("data/Benchmark_Dataset/images")
    output_dir = Path("data/Benchmark_Dataset/labels_json")

    converter = YOLOToLabelmeConverter(labels_dir, images_dir, output_dir)
    stats = converter.convert_all()

    print("\nConversion complete:")
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
