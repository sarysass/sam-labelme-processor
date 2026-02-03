"""
Configuration management module.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import copy
import yaml


class Config:
    """Configuration management class."""

    DEFAULT_CONFIG = {
        "sam": {
            "weights": "weights/sam2.1_t.pt",
            "device": "auto",
            "imgsz": 1024,
            "iou_threshold": 0.3,
        },
        "output": {
            "separate": True,
            "combine": False,
            "format": "polygon",
            "polygon_simplification": {
                "enabled": True,
                "method": "adaptive",
                "base_epsilon_factor": 0.005,
                "adaptive_factor": 0.5,
                "min_points": 8,
                "max_points": 50,
                "curvature_window": 5,
            },
        },
        "data": {
            "root": "./data",
            "images_dir": "images",
            "bbox_dir": "bbox",
            "mask_dir": "mask",
            "combined_dir": "output/combined",
        },
        "processing": {
            "batch_size": 20,
            "num_workers": 2,
            "enable_checkpoint": True,
            "checkpoint_interval": 100,
            "enable_resume": True,
            "max_retries": 3,
            "retry_delay": 5,
            "memory_limit_gb": 12,
            "enable_memory_watch": True,
            "preload_images": False,
            "image_cache_size": 10,
            "skip_empty_labels": True,
        },
        "logging": {"level": "INFO", "file": "logs/processor.log"},
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self._config: Dict[str, Any] = copy.deepcopy(self.DEFAULT_CONFIG)

        if config_path and Path(config_path).exists():
            self._load_from_file(Path(config_path))

    def _load_from_file(self, config_path: Path) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Raises:
            yaml.YAMLError: If the YAML file is invalid.
            ValueError: If the configuration format is invalid.
        """
        try:
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)

            if not isinstance(loaded_config, dict):
                raise ValueError("Configuration file must contain a dictionary")

            # Merge with default config (deep merge for nested dicts)
            self._merge_config(self._config, loaded_config)

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Deep merge override config into base config.

        Args:
            base: Base configuration dictionary (modified in place).
            override: Override configuration dictionary.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested key access using dot notation (e.g., 'sam.weights').

        Args:
            key: Configuration key (supports nested keys with dot notation).
            default: Default value if key is not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value
