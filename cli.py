#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM Labelme Processor - CLI entry point.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config
from src.core.data_manager import DataManager
from src.core.sam_processor import SAMProcessor
from src.models.sam_wrapper import SAMWrapper


def setup_logging(level: str = "INFO", log_file=None) -> None:
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def load_runtime_config(args) -> Config:
    """Load runtime configuration from the selected file."""
    return Config(args.config)


def build_data_manager(config: Config, args) -> DataManager:
    """Build the dataset manager from config and CLI overrides."""
    data_root = Path(args.data_dir) if getattr(args, "data_dir", None) else Path(config.get("data.root"))
    return DataManager(
        data_root=data_root,
        images_dir=config.get("data.images_dir", "images"),
        bbox_dir=config.get("data.bbox_dir", "bbox"),
        mask_dir=config.get("data.mask_dir", "mask"),
        bbox_extension=config.get("data.bbox_extension", ".json"),
    )


def build_processor(config: Config, data_manager: DataManager) -> SAMProcessor:
    """Build the SAM processor and its runtime dependencies."""
    sam_wrapper = SAMWrapper(
        weights=config.get("sam.weights", "weights/sam2.1_t.pt"),
        device=config.get("sam.device", "auto"),
        imgsz=config.get("sam.imgsz", 1024),
        iou_threshold=config.get("sam.iou_threshold", 0.3),
    )
    return SAMProcessor(
        data_manager=data_manager,
        sam_wrapper=sam_wrapper,
        output_separate=config.get("output.separate", True),
        output_combine=config.get("output.combine", False),
        combined_dir=Path(config.get("data.combined_dir", "output/combined")),
        batch_size=config.get("processing.batch_size", 20),
        num_workers=config.get("processing.num_workers", 2),
        enable_checkpoint=config.get("processing.enable_checkpoint", True),
        checkpoint_interval=config.get("processing.checkpoint_interval", 100),
        enable_resume=config.get("processing.enable_resume", True),
        max_retries=config.get("processing.max_retries", 3),
        retry_delay=config.get("processing.retry_delay", 5),
        skip_empty_labels=config.get("processing.skip_empty_labels", True),
        memory_limit_gb=config.get("processing.memory_limit_gb", 12),
        enable_memory_watch=config.get("processing.enable_memory_watch", True),
        preload_images=config.get("processing.preload_images", False),
        image_cache_size=config.get("processing.image_cache_size", 10),
    )


def cmd_process(args):
    """process command."""
    # Load config
    config = load_runtime_config(args)

    # Setup logging
    setup_logging(config.get("logging.level", "INFO"), config.get("logging.file"))

    data_manager = build_data_manager(config, args)
    processor = build_processor(config, data_manager)

    # Get pending items
    if args.resume:
        data_items = data_manager.get_pending_items()
    else:
        data_items = data_manager.scan_dataset()

        data_items = [
            item for item in data_items if item.bbox_path and item.bbox_path.exists()
        ]

    if not data_items:
        print("No items to process")
        return

    print(f"Processing {len(data_items)} items...")

    if not data_items:
        return

    results = processor.process_batch(data_items, show_progress=True)

    success = sum(1 for r in results if r.success)
    failed = len(results) - success
    empty_skipped = sum(1 for r in results if r.success and r.mask_shapes == [])

    print(f"\nCompleted: {success} succeeded, {failed} failed")
    if empty_skipped > 0:
        print(f"Empty labels skipped: {empty_skipped}")


def cmd_validate(args):
    """validate command."""
    config = load_runtime_config(args)
    data_manager = build_data_manager(config, args)

    items = data_manager.scan_dataset()
    stats = data_manager.get_stats()

    print("\nDataset validation:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  With bbox: {stats['with_bbox']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Pending: {stats['pending']}")

    # Check invalid bboxes
    invalid = [item for item in items if item.status == "no_bbox"]
    if invalid:
        print(f"\nWarning: {len(invalid)} images without bbox:")
        for item in invalid[:10]:
            print(f"  - {item.relative_path}")


def cmd_stats(args):
    """stats command."""
    config = load_runtime_config(args)
    data_manager = build_data_manager(config, args)

    stats = data_manager.get_stats()

    print("\nDataset statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(
        f"  With bbox: {stats['with_bbox']} ({stats['with_bbox'] / stats['total_images'] * 100:.1f}%)"
    )
    print(
        f"  Processed: {stats['processed']} ({stats['processed'] / stats['total_images'] * 100:.1f}%)"
    )
    print(f"  Pending: {stats['pending']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAM Labelme Processor - Batch generate masks from bboxes"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # process command
    process_parser = subparsers.add_parser(
        "process", help="Process images to generate masks"
    )
    process_parser.add_argument(
        "--data-dir", "-d", type=str, help="Data directory (overrides config)"
    )
    process_parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume processing (skip already processed)",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate dataset structure"
    )
    validate_parser.add_argument(
        "--data-dir", "-d", type=str, help="Data directory (overrides config)"
    )

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument(
        "--data-dir", "-d", type=str, help="Data directory (overrides config)"
    )

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
