# SAM Labelme Processor - Agent Development Guide

This guide is for agentic coding assistants working on this project. Follow these conventions to maintain code quality and consistency.

## Project Overview

Batch processing tool that generates masks from bounding boxes using SAM (Segment Anything Model) and outputs Labelme JSON format. Built with strict TDD methodology.

**Core Workflow**: Load images → Read bboxes → SAM inference → Save masks (separate or combined)

## Build & Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests with verbose output
pytest tests/ -v

# Run specific test module
pytest tests/test_config.py -v

# Run single test function
pytest tests/test_data_manager.py::TestDataManager::test_scan_all_images -v

# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# Run CLI commands
python cli.py validate
python cli.py stats
python cli.py process --resume
```

## Code Style Guidelines

### File Structure
```
sam-labelme-processor/
├── cli.py                 # CLI entry point (no package import)
├── config.yaml            # Configuration file
├── src/
│   ├── core/             # Core logic modules
│   └── models/           # Model wrappers (SAM)
└── tests/               # Test files mirror src structure
```

### Imports

**Order**: Standard library → External packages → Local modules

```python
# Standard library
from pathlib import Path
from typing import List, Dict, Any, Optional

# External packages
import yaml
import cv2
import numpy as np
import logging

# Local modules (from src/)
from src.core.config import Config
```

**CLI entry point pattern** (cli.py only):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.core.config import Config
```

**Relative imports within src/**:
```python
from ..core.config import Config
from ..models.sam_wrapper import SAMWrapper
```

### Type Annotations

**Mandatory** for all function parameters and return values. Use `Optional[T]` for nullable types, not `T | None` (Python 3.10+).

**Common types from typing module**: `List`, `Dict`, `Any`, `Optional`, `Tuple`, `Callable`

**Dataclasses with type hints**:
```python
from dataclasses import dataclass

@dataclass
class DataItem:
    image_path: Path
    bbox_path: Optional[Path] = None
```

### Naming Conventions

- **Classes**: `PascalCase` → `Config`, `DataManager`, `SAMProcessor`
- **Functions/Methods**: `snake_case` → `process_single()`, `get_pending_items()`
- **Constants**: `UPPER_SNAKE_CASE` → `IMAGE_EXTENSIONS`
- **Private methods**: `_snake_case` → `_load_from_file()`
- **Instance variables**: `self.variable_name`
- **Test classes**: `Test{ClassName}` → `TestConfig`, `TestDataManager`
- **Test methods**: `test_{description}` → `test_default_config_values()`

### Documentation

**Module docstring** (at top of every .py file):
```python
"""
Configuration management module.
"""
```

**Class docstring** with purpose description:
```python
class SAMWrapper:
    """
    SAM model wrapper.

    Wraps MicroHunter's UltralyticsSAMPredictor, providing a simplified interface.
    """
```

**Function docstring format**:
```python
def predict(self, image: np.ndarray, bboxes: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Perform SAM inference on image.

    Args:
        image: BGR format numpy array (H, W, 3).
        bboxes: BBox list, format [[x1, y1, x2, y2], ...].

    Returns:
        List of mask information for each bbox.
    """
```

### Error Handling

**Logging pattern**:
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Processing {len(data_items)} items")
logger.warning(f"Failed: {error_message}")
logger.error(f"Error processing: {e}")
```

**Return result objects for errors** (don't raise unless critical):
```python
@dataclass
class ProcessingResult:
    data_item: DataItem
    success: bool
    error_message: Optional[str] = None

if not data_item.bbox_path.exists():
    return ProcessingResult(data_item=data_item, success=False, error_message="BBox file not found")
```

### Testing Style

**Test file structure**:
```python
"""
Tests for Config module.
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.core.config import Config

class TestConfig:
    """Tests for Config class."""

    def test_default_config_values(self):
        """Test: Use default values when no config file is provided."""
        config = Config()
        assert config.get("sam.weights") == "weights/sam2.1_t.pt"

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
```

**Test naming**:
- Test methods: `test_{feature}_{scenario}` or `test_{description}`
- Fixture names: descriptive nouns (`temp_data_dir`, `mock_sam_wrapper`)
- Test descriptions in docstrings: `"Test: {what is being tested}"`

### Path Handling

**Always use pathlib.Path**:
```python
from pathlib import Path

data_root = Path("./data")
data_root.mkdir(parents=True, exist_ok=True)
image_path = self.images_dir / relative_image_path
```

## Development Workflow

1. **RED**: Write failing test first
2. **GREEN**: Implement minimal code to pass test
3. **REFACTOR**: Clean up while keeping tests green

Run tests after every change: `pytest tests/ -v`

## Important Notes

- No build step required (pure Python)
- No linting/formatting config found - follow PEP 8
- Python 3.10+ required
- Virtual environment recommended
- CLI entry point (cli.py) uses sys.path manipulation
- All other code uses relative imports from src/

## Output Format

### Current Implementation

**Shape Type**: `polygon`
**Fields**: 
- `shape_type`: "polygon"
- `points`: List of polygon contour points [[x, y], ...]
- `label`: Class label
- `group_id`: Optional integer
- `description`: "Generated by SAM"
- `flags`: `{"sam_generated": true}`

**Example**:
```json
{
  "version": "5.10.1",
  "flags": {},
  "shapes": [
    {
      "label": "class_0",
      "points": [[666.0, 92.0], [665.0, 93.0], ...],
      "group_id": 0,
      "shape_type": "polygon",
      "description": "Generated by SAM",
      "flags": {"sam_generated": true}
    }
  ],
  "imagePath": "frame_0000.png",
  "imageHeight": 1440,
  "imageWidth": 1168
}
```

**Characteristics**:
- Multi-gon points: 50-93 points per shape (simplified via OpenCV APPROX_SIMPLE)
- No `mask` field (reduced storage size)
- Supports manual editing: Add/remove/move polygon vertices in Labelme

### Comparison with Labelme AI Mask

| Feature | Current Implementation | Labelme AI Mask |
|---------|----------------------|------------------|
| **shape_type** | "polygon" | "mask" or "polygon" |
| **mask field** | ❌ No | ✅ Yes (Base64 PNG) |
| **points** | 50-93 (simplified) | Varies (from SAM output) |
| **Storage overhead** | ~593 KB (9 shapes) | ~669 KB (with mask field, +13%) |
| **Rendering in Labelme** | Green outline + semi-transparent fill | White fill + green outline (mask field) |
| **Manual editing** | Add/remove/move vertices | Click include/exclude points (mask) or edit vertices (polygon) |

### Training Data Conversion

To convert polygon output to binary masks for SAM training:

```python
import json
import cv2
import numpy as np
from pathlib import Path

# Read mask JSON
with open('mask/frame_0000.json') as f:
    data = json.load(f)

# Create blank mask
mask = np.zeros((data['imageHeight'], data['imageWidth']), dtype=np.uint8)

# Fill polygons
for shape in data['shapes']:
    if shape['shape_type'] == 'polygon':
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

# Save training mask
cv2.imwrite('training_mask.png', mask)
```

### Future Considerations

If full compatibility with Labelme AI Mask is needed, consider:
- Adding `mask` field with Base64-encoded PNG images
- Supporting `shape_type="mask"` option
- Preserving original SAM output before polygon simplification
- Estimated storage overhead: ~13% increase for typical datasets
