"""
SAM processor module for batch processing.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import logging
import cv2
from multiprocessing import Pool
import json
import time
import os
import resource
import numpy as np

from ..models.sam_wrapper import SAMWrapper
from ..core.labelme_io import LabelmeIO, MaskShape, BBoxShape, ImageInfo
from ..core.data_manager import DataManager, DataItem

logger = logging.getLogger(__name__)


class ProcessingResult:
    """Processing result."""

    def __init__(
        self,
        data_item: DataItem,
        success: bool,
        mask_shapes: Optional[List[MaskShape]] = None,
        error_message: Optional[str] = None,
    ):
        self.data_item = data_item
        self.success = success
        self.mask_shapes = mask_shapes or []
        self.error_message = error_message


class SAMProcessor:
    """
    SAM batch processor.

    Coordinates entire workflow: load data -> SAM inference -> save results.
    Supports batch parallel processing with checkpoints for large datasets.
    """

    def __init__(
        self,
        data_manager: DataManager,
        sam_wrapper: SAMWrapper,
        output_separate: bool = True,
        output_combine: bool = False,
        combined_dir: Optional[Path] = None,
        batch_size: int = 20,
        num_workers: int = 2,
        enable_checkpoint: bool = True,
        checkpoint_interval: int = 100,
        enable_resume: bool = True,
        max_retries: int = 3,
        retry_delay: int = 5,
        skip_empty_labels: bool = True,
        memory_limit_gb: float = 12,
        enable_memory_watch: bool = True,
        preload_images: bool = False,
        image_cache_size: int = 10,
    ):
        """
        Initialize SAM processor.

        Args:
            data_manager: Data manager instance.
            sam_wrapper: SAM wrapper instance.
            output_separate: Output separate mask JSON files.
            output_combine: Output combined bbox+mask JSON files.
            combined_dir: Directory for combined output.
            batch_size: Number of images to process per batch.
            num_workers: Number of parallel processes.
            enable_checkpoint: Enable checkpoint saving.
            checkpoint_interval: Save checkpoint every N images.
            enable_resume: Support resuming from checkpoint.
            max_retries: Max retries per batch.
            retry_delay: Retry delay in seconds.
            skip_empty_labels: Skip files with shapes=0.
            memory_limit_gb: Memory limit in GB.
            enable_memory_watch: Enable memory usage check.
            preload_images: Preload images into cache.
            image_cache_size: Maximum number of cached images.
        """
        self.data_manager = data_manager
        self.sam_wrapper = sam_wrapper
        self.output_separate = output_separate
        self.output_combine = output_combine
        self.combined_dir = combined_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.enable_checkpoint = enable_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.enable_resume = enable_resume
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.skip_empty_labels = skip_empty_labels
        self.memory_limit_gb = memory_limit_gb
        self.enable_memory_watch = enable_memory_watch
        self.preload_images = preload_images
        self.image_cache_size = image_cache_size
        self._image_cache = {}
        self._image_cache_order = []

        self.checkpoint_file = None
        if data_manager and data_manager.data_root:
            self.checkpoint_file = (
                data_manager.data_root / ".processing_checkpoint.json"
            )

        if output_combine and combined_dir:
            combined_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, processed_count: int, total_count: int) -> None:
        """Save checkpoint to JSON file."""
        if not self.enable_checkpoint or self.checkpoint_file is None:
            return

        checkpoint_interval = max(1, int(self.checkpoint_interval))
        if processed_count < total_count and processed_count % checkpoint_interval != 0:
            return

        checkpoint_data = {
            "processed_count": processed_count,
            "total_count": total_count,
            "timestamp": time.time(),
        }

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved: {processed_count}/{total_count}")

    def _load_checkpoint(self) -> dict:
        """Load checkpoint from JSON file."""
        if (
            not self.enable_checkpoint
            or not self.enable_resume
            or self.checkpoint_file is None
            or not self.checkpoint_file.exists()
        ):
            return {"processed_count": 0, "total_count": 0}

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            logger.info(
                f"Checkpoint loaded: {checkpoint['processed_count']}/{checkpoint['total_count']}"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {"processed_count": 0, "total_count": 0}

    def _process_single_item(self, data_item: DataItem) -> ProcessingResult:
        """Process single item with retry logic."""
        retry_count = 0
        last_error = None

        while retry_count < self.max_retries:
            try:
                return self._process_single_item_internal(data_item)
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(
                        f"Retry {retry_count}/{self.max_retries} "
                        f"for {data_item.image_id}: {last_error}. "
                        f"Waiting {self.retry_delay}s before retry..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Failed after {self.max_retries} retries: "
                        f"{data_item.image_id}: {last_error}"
                    )
                    return ProcessingResult(
                        data_item=data_item, success=False, error_message=last_error
                    )

        return ProcessingResult(
            data_item=data_item,
            success=False,
            error_message=last_error or "Unknown error",
        )

    def _get_memory_usage_gb(self) -> float:
        """
        Get current process max RSS memory usage in GB.

        On macOS, ru_maxrss is bytes. On Linux, it's KB.
        """
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return usage / (1024**3)
        return usage / (1024**2)

    def _should_fallback_to_single_worker(self) -> bool:
        """Check if memory usage exceeds configured limit."""
        if not self.enable_memory_watch:
            return False
        if self.memory_limit_gb is None or self.memory_limit_gb <= 0:
            return False
        current_gb = self._get_memory_usage_gb()
        if current_gb > self.memory_limit_gb:
            logger.warning(
                f"Memory usage {current_gb:.2f}GB exceeds limit "
                f"{self.memory_limit_gb:.2f}GB, fallback to single-process mode."
            )
            return True
        return False

    def _update_image_cache(self, image_path: Path, image: np.ndarray) -> None:
        """Insert image into simple FIFO cache."""
        if not self.preload_images or self.image_cache_size <= 0:
            return

        cache_key = str(image_path)
        if cache_key in self._image_cache:
            return

        self._image_cache[cache_key] = image
        self._image_cache_order.append(cache_key)
        while len(self._image_cache_order) > self.image_cache_size:
            old_key = self._image_cache_order.pop(0)
            self._image_cache.pop(old_key, None)

    def _get_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Get image from cache or disk."""
        cache_key = str(image_path)
        if self.preload_images and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        image = cv2.imread(str(image_path))
        if image is not None:
            self._update_image_cache(image_path, image)
        return image

    def _preload_batch_images(self, batch_data: List[DataItem]) -> None:
        """Preload batch images up to cache size."""
        if not self.preload_images or self.image_cache_size <= 0:
            return
        if self.num_workers > 1:
            return

        for item in batch_data[: self.image_cache_size]:
            image_path = self.data_manager.images_dir / item.relative_path
            if str(image_path) not in self._image_cache:
                image = cv2.imread(str(image_path))
                if image is not None:
                    self._update_image_cache(image_path, image)

    def _read_yolo_txt_labels(
        self,
        txt_path: Path,
        image_path: Path,
        image_height: int,
        image_width: int,
    ) -> Tuple[ImageInfo, List[BBoxShape]]:
        """Read YOLO TXT labels and convert to bbox shapes."""
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
        return image_info, bbox_shapes

    def _read_label_file(
        self,
        label_path: Path,
        image_path: Path,
        image_height: int,
        image_width: int,
    ) -> Tuple[ImageInfo, List[BBoxShape]]:
        """Read label file based on extension (.json or .txt)."""
        suffix = label_path.suffix.lower()
        if suffix == ".txt":
            return self._read_yolo_txt_labels(
                label_path, image_path, image_height, image_width
            )
        return LabelmeIO.read_bbox_json(label_path)

    def _build_output_image_info(
        self,
        base_info: ImageInfo,
        image_path: Path,
        output_json_path: Path,
    ) -> ImageInfo:
        """Build output image info with imagePath relative to output JSON path."""
        relative_image_path = os.path.relpath(image_path, start=output_json_path.parent)
        return ImageInfo(
            image_path=Path(relative_image_path).as_posix(),
            image_height=base_info.image_height,
            image_width=base_info.image_width,
            version=base_info.version,
            flags=base_info.flags,
            image_data=base_info.image_data,
        )

    def _process_single_item_internal(self, data_item: DataItem) -> ProcessingResult:
        """Internal method to process single item without retry logic."""
        image_path = self.data_manager.images_dir / data_item.relative_path
        label_path = data_item.bbox_path

        if label_path is None or not label_path.exists():
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"BBox file not found: {label_path}",
            )

        try:
            image = self._get_image(image_path)
            if image is None:
                return ProcessingResult(
                    data_item=data_item,
                    success=False,
                    error_message=f"Failed to load image: {image_path}",
                )
            image_height, image_width = image.shape[:2]
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"Failed to load image: {e}",
            )

        try:
            image_info, bbox_shapes = self._read_label_file(
                label_path, image_path, image_height, image_width
            )
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"Failed to read bbox: {e}",
            )

        if not bbox_shapes:
            if self.skip_empty_labels:
                logger.info(f"Skipping {data_item.image_id}: No bbox shapes found")
                return ProcessingResult(
                    data_item=data_item, success=True, mask_shapes=[]
                )
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message="No bbox shapes found",
            )

        try:
            bboxes = [shape.points for shape in bbox_shapes]
            bboxes = [[p[0][0], p[0][1], p[1][0], p[1][1]] for p in bboxes]
            sam_results = self.sam_wrapper.predict(image, bboxes)
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"SAM inference failed: {e}",
            )

        try:
            mask_shapes = []
            for i, (bbox_shape, sam_result) in enumerate(zip(bbox_shapes, sam_results)):
                mask_array = np.array(sam_result.get("mask", []), dtype=np.uint8)
                points, encoded_mask = LabelmeIO.mask_to_labelme_mask(mask_array)

                mask_shapes.append(
                    MaskShape(
                        label="mask",
                        points=points,
                        group_id=bbox_shape.group_id or i,
                        description="Generated by SAM",
                        flags={"sam_generated": True},
                        shape_type="mask",
                        mask=encoded_mask,
                    )
                )
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"Failed to create mask shapes: {e}",
            )

        try:
            if self.output_separate:
                mask_json_path = data_item.mask_path
                mask_json_path.parent.mkdir(parents=True, exist_ok=True)
                mask_image_info = self._build_output_image_info(
                    image_info, image_path, mask_json_path
                )
                LabelmeIO.write_mask_json(mask_json_path, mask_shapes, mask_image_info)
                logger.info(f"Saved mask to {mask_json_path}")

            if self.output_combine and self.combined_dir is not None:
                combined_json_path = (
                    self.combined_dir / data_item.relative_path.with_suffix(".json")
                )
                combined_json_path.parent.mkdir(parents=True, exist_ok=True)
                combined_image_info = self._build_output_image_info(
                    image_info, image_path, combined_json_path
                )
                LabelmeIO.write_combined_json(
                    combined_json_path,
                    bbox_shapes,
                    mask_shapes,
                    combined_image_info,
                )
                logger.info(f"Saved combined output to {combined_json_path}")
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"Failed to save mask: {e}",
            )

        return ProcessingResult(data_item=data_item, success=True, mask_shapes=mask_shapes)

    def _process_batch_with_worker(
        self, batch_data: List[DataItem]
    ) -> List[ProcessingResult]:
        """Process a batch of items (worker function)."""
        results = []
        for item in batch_data:
            result = self._process_single_item(item)
            results.append(result)
        return results

    def _get_batch_results(self, batch_data: List[DataItem]) -> List[ProcessingResult]:
        """Process a batch using multiprocessing pool."""
        try:
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(
                    self._process_batch_with_worker,
                    [batch_data[i :: self.num_workers] for i in range(self.num_workers)],
                )
            return [result for batch_results in results for result in batch_results]
        except Exception as e:
            logger.warning(
                f"Multiprocessing failed ({e}), fallback to single-process batch."
            )
            return [self._process_single_item(item) for item in batch_data]

    def process_single(self, data_item: DataItem) -> ProcessingResult:
        """
        Process single data item.

        Args:
            data_item: Data item to process.

        Returns:
            ProcessingResult with processing status.
        """
        return self._process_single_item(data_item)

    def process_batch(
        self, data_items: Optional[List[DataItem]] = None, show_progress: bool = True
    ) -> List[ProcessingResult]:
        """
        Batch process data items with parallel processing and checkpoint support.

        Args:
            data_items: List of data items to process, None means process all pending.
            show_progress: Show progress bar.

        Returns:
            List of processing results.
        """
        if data_items is None:
            all_items = self.data_manager.scan_dataset()
            data_items = [
                item
                for item in all_items
                if item.bbox_path
                and item.bbox_path.exists()
                and (not self.skip_empty_labels or item.status != "no_bbox")
            ]

        if not data_items:
            logger.info("No items to process")
            return []

        checkpoint = self._load_checkpoint()
        start_index = int(checkpoint.get("processed_count", 0) or 0)
        checkpoint_total = int(checkpoint.get("total_count", 0) or 0)

        if start_index < 0:
            start_index = 0

        # If input size differs from checkpoint total, treat checkpoint as stale.
        # This happens when caller passes a filtered list (e.g., pending items only).
        if checkpoint_total > 0 and checkpoint_total != len(data_items):
            logger.warning(
                f"Checkpoint total_count={checkpoint_total} does not match current "
                f"items={len(data_items)}. Ignoring checkpoint progress."
            )
            start_index = 0

        if start_index > len(data_items):
            logger.warning(
                f"Checkpoint processed_count={start_index} exceeds current "
                f"items={len(data_items)}. Resetting to 0."
            )
            start_index = 0

        if start_index == len(data_items):
            logger.info(f"All items already processed. Total: {start_index}")
            return []

        total_count = len(data_items)
        data_items = data_items[start_index:]

        logger.info(
            f"Processing {len(data_items)} items (resuming from {start_index})"
        )
        logger.info(f"Total dataset size: {total_count} items")

        results = []
        total_batches = (len(data_items) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(data_items))
            batch_data = data_items[batch_start:batch_end]
            global_batch_start = batch_start + start_index
            global_batch_end = batch_end + start_index

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches}: "
                f"items {global_batch_start + 1}-{global_batch_end}"
            )

            self._preload_batch_images(batch_data)
            use_single_worker = self.num_workers == 1 or self._should_fallback_to_single_worker()

            if use_single_worker:
                batch_results = [self._process_single_item(item) for item in batch_data]
            else:
                batch_results = self._get_batch_results(batch_data)

            results.extend(batch_results)

            processed_count = start_index + len(results)
            self._save_checkpoint(processed_count, total_count)

            if show_progress:
                print(
                    f"Progress: {processed_count}/{total_count} ({processed_count / total_count * 100:.1f}%)"
                )

        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        logger.info(f"Completed: {success_count} succeeded, {failed_count} failed")

        return results
