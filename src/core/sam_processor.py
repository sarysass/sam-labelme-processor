"""
Compatibility facade for the SAM processing workflow.
"""

from pathlib import Path
from typing import List, Optional

from ..models.sam_wrapper import SAMWrapper
from .batch_runner import BatchRunner
from .data_manager import DataItem, DataManager
from .item_processor import ItemProcessor
from .label_reader import LabelReader
from .result_writer import ResultWriter
from .types import ProcessingResult


class SAMProcessor:
    """
    Compatibility facade preserving the original constructor and public methods.
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
        label_reader: Optional[LabelReader] = None,
        result_writer: Optional[ResultWriter] = None,
        item_processor: Optional[ItemProcessor] = None,
        batch_runner: Optional[BatchRunner] = None,
    ):
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
        self.label_reader = label_reader or LabelReader()
        self.result_writer = result_writer or ResultWriter(
            output_separate=output_separate,
            output_combine=output_combine,
            combined_dir=combined_dir,
        )
        self.item_processor = item_processor or ItemProcessor(
            data_manager=data_manager,
            sam_wrapper=sam_wrapper,
            label_reader=self.label_reader,
            result_writer=self.result_writer,
            skip_empty_labels=skip_empty_labels,
        )
        self.batch_runner = batch_runner or BatchRunner(
            data_manager=data_manager,
            item_processor=self.item_processor,
            batch_size=batch_size,
            num_workers=num_workers,
            enable_checkpoint=enable_checkpoint,
            checkpoint_interval=checkpoint_interval,
            enable_resume=enable_resume,
            max_retries=max_retries,
            retry_delay=retry_delay,
            memory_limit_gb=memory_limit_gb,
            enable_memory_watch=enable_memory_watch,
            preload_images=preload_images,
            image_cache_size=image_cache_size,
        )

    def process_single(self, data_item: DataItem) -> ProcessingResult:
        """Process one item through the item processor."""
        return self.item_processor.process(data_item)

    def process_batch(
        self,
        data_items: Optional[List[DataItem]] = None,
        show_progress: bool = True,
    ) -> List[ProcessingResult]:
        """Process a list of items, defaulting to all current pending items."""
        if data_items is None:
            all_items = self.data_manager.scan_dataset()
            data_items = [
                item
                for item in all_items
                if item.bbox_path
                and item.bbox_path.exists()
                and (not self.skip_empty_labels or item.status != "no_bbox")
            ]

        return self.batch_runner.process_batch(data_items, show_progress=show_progress)
