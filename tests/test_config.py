"""
Tests for Config module.
"""

import pytest
from pathlib import Path
import yaml
import tempfile
import os

# Import the module to test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_default_config_values(self):
        """Test: Use default values when no config file is provided."""
        config = Config()

        assert config.get("sam.weights") == "weights/sam2.1_t.pt"
        assert config.get("sam.device") == "auto"
        assert config.get("sam.imgsz") == 1024
        assert config.get("output.separate") is True
        assert config.get("output.combine") is False
        assert config.get("data.root") == "./data"

    def test_load_yaml_config(self):
        """Test: Load configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "sam": {"weights": "/custom/path/sam.pt", "device": "cuda:0"},
                    "data": {"root": "/custom/data"},
                },
                f,
            )
            config_path = f.name

        try:
            config = Config(config_path)

            # Verify overridden values
            assert config.get("sam.weights") == "/custom/path/sam.pt"
            assert config.get("sam.device") == "cuda:0"
            assert config.get("data.root") == "/custom/data"

            # Verify default values still work
            assert config.get("sam.imgsz") == 1024
            assert config.get("output.separate") is True
        finally:
            os.unlink(config_path)

    def test_nested_key_access(self):
        """Test: Support nested key access (e.g., 'sam.weights')."""
        config = Config()

        # Test nested access
        assert config.get("sam.weights") is not None
        assert config.get("data.images_dir") == "images"

        # Test non-existent keys
        assert config.get("nonexistent.key", "default") == "default"
        assert config.get("sam.nonexistent") is None

    def test_invalid_config_file(self):
        """Test: Handle invalid config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:[{")
            config_path = f.name

        try:
            # Should raise exception or use default values
            with pytest.raises((yaml.YAMLError, ValueError)):
                Config(config_path)
        finally:
            os.unlink(config_path)

    def test_config_file_not_exists(self):
        """Test: Use default values when config file doesn't exist."""
        config = Config("/nonexistent/path/config.yaml")

        # Should use default values, not raise exception
        assert config.get("sam.weights") == "weights/sam2.1_t.pt"
