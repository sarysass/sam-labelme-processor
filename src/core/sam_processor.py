"""
SAM processor module for batch processing.
"""

from pathlib import Path
from typing import List, Optional
import logging
import cv2
from multiprocessing import Pool
import json
import time
import numpy as np

from ..models.sam_wrapper import SAMWrapper
from ..core.labelme_io import LabelmeIO, MaskShape
from ..core.data_manager import DataManager, DataItem
from ..utils.polygon_utils import (
    simplify_polygon_adaptive,
    simplify_contour_to_max_points,
)

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
        simplification_config: Optional[dict] = None,
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
            simplification_config: Polygon simplification config.
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

        self.simplification_enabled = (
            simplification_config.get("enabled", True)
            if simplification_config
            else True
        )
        self.simplification_method = (
            simplification_config.get("method", "adaptive")
            if simplification_config
            else "adaptive"
        )
        self.simplification_params = (
            simplification_config if simplification_config else {}
        )

        self.checkpoint_file = None
        if data_manager and data_manager.data_root:
            self.checkpoint_file = (
                data_manager.data_root / ".processing_checkpoint.json"
            )

        if output_combine and combined_dir:
            combined_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, processed_count: int, total_count: int) -> None:
        """Save checkpoint to JSON file."""
        if not self.enable_checkpoint:
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
        if not self.enable_resume or not self.checkpoint_file.exists():
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

    def _process_single_item_internal(self, data_item: DataItem) -> ProcessingResult:
        """Internal method to process single item without retry logic."""
        if self.skip_empty_labels:
            try:
                image_info, bbox_shapes = LabelmeIO.read_bbox_json(data_item.bbox_path)
                if not bbox_shapes:
                    logger.info(f"Skipping {data_item.image_id}: No bbox shapes found")
                    return ProcessingResult(
                        data_item=data_item, success=True, mask_shapes=[]
                    )
            except Exception as e:
                logger.warning(f"Error reading bbox for {data_item.image_id}: {e}")

        try:
            image_info, bbox_shapes = LabelmeIO.read_bbox_json(data_item.bbox_path)
            if not bbox_shapes:
                return ProcessingResult(
                    data_item=data_item,
                    success=False,
                    error_message="No bbox shapes found",
                )
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"Failed to read bbox: {e}",
            )

        try:
            image_path = self.data_manager.images_dir / data_item.relative_path
            image = cv2.imread(str(image_path))
            if image is None:
                return ProcessingResult(
                    data_item=data_item,
                    success=False,
                    error_message=f"Failed to load image: {image_path}",
                )
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"Failed to load image: {e}",
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
                contour = sam_result["contour"]

                if self.simplification_enabled:
                    contour_np = np.array(contour, dtype=np.float32)
                    if len(contour_np.shape) == 2 and contour_np.shape[1] == 2:
                        contour_np = contour_np.reshape(-1, 1, 2)

                    if self.simplification_method == "adaptive":
                        simplified_contour = simplify_polygon_adaptive(
                            contour_np,
                            base_epsilon_factor=self.simplification_params.get(
                                "base_epsilon_factor", 0.005
                            ),
                            adaptive_factor=self.simplification_params.get(
                                "adaptive_factor", 0.5
                            ),
                            min_points=self.simplification_params.get("min_points", 8),
                            max_points=self.simplification_params.get("max_points", 50),
                            curvature_window=self.simplification_params.get(
                                "curvature_window", 5
                            ),
                        )
                    elif self.simplification_method == "max_points":
                        simplified_contour = simplify_contour_to_max_points(
                            contour_np,
                            max_points=self.simplification_params.get("max_points", 50),
                        )
                    else:
                        simplified_contour = contour_np

                    contour = simplified_contour

                if hasattr(contour, "reshape"):
                    contour = contour.reshape(-1, 2)
                    points = contour.tolist()
                elif hasattr(contour, "tolist"):
                    points = contour.tolist()
                    if (
                        points
                        and isinstance(points[0], list)
                        and len(points[0]) > 0
                        and isinstance(points[0][0], list)
                    ):
                        points = [[float(pt[0]), float(pt[1])] for pt in points]
                else:
                    points = contour

                if not isinstance(points, list) or not all(
                    isinstance(p, list) and len(p) == 2 for p in points
                ):
                    points = []

                mask_shapes.append(
                    MaskShape(
                        label=bbox_shape.label,
                        points=points,
                        group_id=bbox_shape.group_id or i,
                        description="Generated by SAM",
                        flags={"sam_generated": True},
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
                LabelmeIO.write_mask_json(mask_json_path, mask_shapes, image_info)
                logger.info(f"Saved mask to {mask_json_path}")
        except Exception as e:
            return ProcessingResult(
                data_item=data_item,
                success=False,
                error_message=f"Failed to save mask: {e}",
            )

        return ProcessingResult(
            data_item=data_item, success=True, mask_shapes=mask_shapes
        )

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
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(
                self._process_batch_with_worker,
                [batch_data[i :: self.num_workers] for i in range(self.num_workers)],
            )

        return [result for batch_results in results for result in batch_results]

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
        start_index = checkpoint.get("processed_count", 0)

        if start_index >= len(data_items):
            logger.info(f"All items already processed. Total: {start_index}")
            return []

        data_items = data_items[start_index:]
        total_count = start_index + len(data_items)

        logger.info(f"Processing {len(data_items)} items (resuming from {start_index})")
        logger.info(f"Total dataset size: {total_count} items")

        results = []
        total_batches = (len(data_items) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size + start_index
            batch_end = min(
                batch_start + self.batch_size, len(data_items) + start_index
            )
            batch_data = data_items[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches}: items {batch_start + 1}-{batch_end}"
            )

            if self.num_workers == 1:
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
