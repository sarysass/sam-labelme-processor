#!/usr/bin/env python
"""
Edge-assisted optimizer for existing Labelme mask annotations.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """Fallback iterator when tqdm is unavailable."""
        return iterable

from src.postprocess.edge_refiner import EdgeOptimizationConfig
from src.postprocess.labelme_adapter import (
    build_output_path,
    collect_json_files,
    default_output_path,
    optimize_labelme_mask_file,
)


logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    """Configure command-line logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize existing Labelme mask JSON files with edge-assisted refinement",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input Labelme JSON file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file or directory. Defaults to a sibling optimized path.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input JSON files instead of writing to a separate location",
    )
    parser.add_argument("--search-radius", type=int, default=6)
    parser.add_argument("--foreground-erode", type=int, default=4)
    parser.add_argument("--background-dilate", type=int, default=2)
    parser.add_argument("--gaussian-kernel-size", type=int, default=3)
    parser.add_argument("--canny-low-threshold", type=int, default=25)
    parser.add_argument("--canny-high-threshold", type=int, default=80)
    parser.add_argument("--min-area-ratio", type=float, default=0.3)
    parser.add_argument("--max-area-ratio", type=float, default=1.5)
    parser.add_argument("--smoothing-kernel-size", type=int, default=5)
    parser.add_argument("--smoothing-morph-radius", type=int, default=1)
    parser.add_argument("--enable-cavity-recovery", action="store_true")
    parser.add_argument("--cavity-min-area", type=int, default=25)
    parser.add_argument("--cavity-min-distance", type=int, default=2)
    parser.add_argument("--cavity-intensity-margin", type=float, default=5.0)
    parser.add_argument("--enable-shell-removal", action="store_true")
    parser.add_argument("--shell-max-thickness", type=int, default=5)
    parser.add_argument("--shell-background-cost-multiplier", type=float, default=1.1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Run the edge-assisted mask optimizer."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.in_place and args.output is not None:
        raise ValueError("--in-place and --output cannot be used together")

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_path = input_path if args.in_place else (args.output.resolve() if args.output else default_output_path(input_path))
    config = EdgeOptimizationConfig(
        search_radius=args.search_radius,
        foreground_erode=args.foreground_erode,
        background_dilate=args.background_dilate,
        gaussian_kernel_size=args.gaussian_kernel_size,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        smoothing_kernel_size=args.smoothing_kernel_size,
        smoothing_morph_radius=args.smoothing_morph_radius,
        enable_cavity_recovery=args.enable_cavity_recovery,
        cavity_min_area=args.cavity_min_area,
        cavity_min_distance=args.cavity_min_distance,
        cavity_intensity_margin=args.cavity_intensity_margin,
        enable_shell_removal=args.enable_shell_removal,
        shell_max_thickness=args.shell_max_thickness,
        shell_background_cost_multiplier=args.shell_background_cost_multiplier,
    )

    json_files = collect_json_files(input_path)
    if not json_files:
        logger.warning("No JSON files found under %s", input_path)
        return 0

    total_updated_shapes = 0
    for input_json_path in tqdm(json_files, desc="Optimizing masks"):
        if input_path.is_file():
            current_output_path = output_path
        elif args.in_place:
            current_output_path = input_json_path
        else:
            current_output_path = build_output_path(
                input_json_path=input_json_path,
                input_root=input_path,
                output_root=output_path,
            )

        stats = optimize_labelme_mask_file(
            input_json_path=input_json_path,
            output_json_path=current_output_path,
            config=config,
        )
        total_updated_shapes += stats["updated_shapes"]

    logger.info(
        "Processed %s JSON files, updated %s mask shapes",
        len(json_files),
        total_updated_shapes,
    )
    print(f"Processed {len(json_files)} JSON files")
    print(f"Updated {total_updated_shapes} mask shapes")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
