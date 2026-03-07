"""
Batch orchestration for SAM processing.
"""

from multiprocessing import Pool
import json
import logging
import os
from pathlib import Path
import resource
import time
from typing import List, Optional

import cv2
import numpy as np

from .data_manager import DataItem, DataManager
from .item_processor import ItemProcessor
from .types import ProcessingResult

logger = logging.getLogger(__name__)


class BatchRunner:
    """Run item processing in batches with checkpointing and retries."""

    def __init__(
        self,
        data_manager: DataManager,
        item_processor: ItemProcessor,
        batch_size: int = 20,
        num_workers: int = 2,
        enable_checkpoint: bool = True,
        checkpoint_interval: int = 100,
        enable_resume: bool = True,
        max_retries: int = 3,
        retry_delay: int = 5,
        memory_limit_gb: float = 12,
        enable_memory_watch: bool = True,
        preload_images: bool = False,
        image_cache_size: int = 10,
    ):
        self.data_manager = data_manager
        self.item_processor = item_processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.enable_checkpoint = enable_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.enable_resume = enable_resume
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.memory_limit_gb = memory_limit_gb
        self.enable_memory_watch = enable_memory_watch
        self.preload_images = preload_images
        self.image_cache_size = image_cache_size
        self._image_cache = {}
        self._image_cache_order = []

        self.checkpoint_file: Optional[Path] = None
        if data_manager and data_manager.data_root:
            self.checkpoint_file = data_manager.data_root / ".processing_checkpoint.json"

    def _save_checkpoint(self, processed_count: int, total_count: int) -> None:
        """Save checkpoint to disk."""
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
        """Load checkpoint from disk."""
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

    @staticmethod
    def _get_memory_usage_gb() -> float:
        """Get current process max RSS memory usage in GB."""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return usage / (1024**3)
        return usage / (1024**2)

    def _should_fallback_to_single_worker(self) -> bool:
        """Check if memory usage exceeds the configured limit."""
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
        """Insert image into a simple FIFO cache."""
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
        """Preload images up to the configured cache size."""
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

    def _process_single_item(self, data_item: DataItem) -> ProcessingResult:
        """Process one item with retry semantics."""
        retry_count = 0
        last_error = None
        image_path = self.data_manager.images_dir / data_item.relative_path

        while retry_count < self.max_retries:
            try:
                image = self._get_image(image_path)
                return self.item_processor.process(data_item, image=image)
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
                        data_item=data_item,
                        success=False,
                        error_message=last_error,
                    )

        return ProcessingResult(
            data_item=data_item,
            success=False,
            error_message=last_error or "Unknown error",
        )

    def _process_batch_with_worker(
        self,
        batch_data: List[DataItem],
    ) -> List[ProcessingResult]:
        """Worker function for multiprocessing batches."""
        return [self._process_single_item(item) for item in batch_data]

    def _get_batch_results(self, batch_data: List[DataItem]) -> List[ProcessingResult]:
        """Process a batch with multiprocessing, falling back if needed."""
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

    def process_batch(
        self,
        data_items: List[DataItem],
        show_progress: bool = True,
    ) -> List[ProcessingResult]:
        """Process dataset items in batches."""
        if not data_items:
            logger.info("No items to process")
            return []

        checkpoint = self._load_checkpoint()
        start_index = int(checkpoint.get("processed_count", 0) or 0)
        checkpoint_total = int(checkpoint.get("total_count", 0) or 0)

        if start_index < 0:
            start_index = 0

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
        remaining_items = data_items[start_index:]

        logger.info(
            f"Processing {len(remaining_items)} items (resuming from {start_index})"
        )
        logger.info(f"Total dataset size: {total_count} items")

        results: List[ProcessingResult] = []
        total_batches = (len(remaining_items) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(remaining_items))
            batch_data = remaining_items[batch_start:batch_end]
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
