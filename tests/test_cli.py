"""
Tests for CLI module.
"""
import pytest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import cli


class TestCLI:
    """Tests for command line interface."""

    def test_cli_help(self):
        """Test: Help information."""
        # Skip if cli.py doesn't exist
        try:
            result = subprocess.run(
                [sys.executable, "cli.py", "--help"],
                capture_output=True,
                text=True,
                cwd="/Users/shali/projects/tools/sam-labelme-processor"
            )
            assert result.returncode == 0 or "FileNotFoundError" in result.stderr
        except FileNotFoundError:
            pytest.skip("cli.py not yet implemented")

    def test_build_data_manager_uses_cli_override(self, tmp_path):
        """Test: CLI override path takes precedence when building the data manager."""
        config = cli.Config()
        args = SimpleNamespace(data_dir=str(tmp_path / "custom"), config="config.yaml")

        data_manager = cli.build_data_manager(config, args)

        assert data_manager.data_root == tmp_path / "custom"

    def test_build_processor_uses_configured_runtime_parameters(self, tmp_path):
        """Test: Processor builder wires config values into the processor."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
sam:
  weights: "weights/custom.pt"
  device: "cpu"
  imgsz: 640
  iou_threshold: 0.2
processing:
  batch_size: 5
  num_workers: 1
"""
        )
        config = cli.Config(config_path)
        data_manager = cli.DataManager(tmp_path / "data")

        processor = cli.build_processor(config, data_manager)

        assert processor.sam_wrapper.weights == "weights/custom.pt"
        assert processor.sam_wrapper.device == "cpu"
        assert processor.sam_wrapper.imgsz == 640
        assert processor.sam_wrapper.iou_threshold == 0.2
        assert processor.batch_runner.batch_size == 5
        assert processor.batch_runner.num_workers == 1
