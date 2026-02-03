# SAM Labelme Processor

Batch process images with bounding boxes using SAM (Segment Anything Model) to generate masks, outputting them in Labelme JSON format.

## Features

- **Batch Processing**: Process multiple images at once with progress tracking
- **Incremental Processing**: Skip already processed files with `--resume` flag
- **Flexible Output**: Separate mask files or combined bbox+mask files
- **Dataset Validation**: Validate dataset structure before processing
- **Statistics**: View dataset statistics (total images, processed, pending)
- **TDD Development**: Built with strict TDD methodology

## Installation

### Prerequisites

- Python 3.10+
- MicroHunter (for SAM model)
  - Install from: `/Users/shali/projects/MicroHunter`

### Setup

1. Clone or navigate to the project directory:
   ```bash
   cd /Users/shali/projects/tools/sam-labelme-processor
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Directory Structure

Prepare your data directory as follows:
```
data/
├── images/
│   ├── frame_001.jpg
│   └── frame_002.jpg
└── bbox/
    ├── frame_001.json
    └── frame_002.json
```

### Commands

#### Validate Dataset
Check if dataset structure is valid:
```bash
python cli.py validate
```

#### View Statistics
Show dataset statistics:
```bash
python cli.py stats
```

#### Process Images
Generate masks for all pending images:
```bash
python cli.py process
```

#### Resume Processing
Skip already processed images:
```bash
python cli.py process --resume
```

#### Custom Data Directory
Use a custom data directory:
```bash
python cli.py process --data-dir /path/to/data
```

## Configuration

Edit `config.yaml` to customize:

```yaml
sam:
  weights: "/path/to/sam/weights.pt"
  device: "auto"              # auto, cuda:0, cpu
  imgsz: 1024
  iou_threshold: 0.3

output:
  separate: true              # Output separate mask JSON
  combine: false              # Output combined bbox+mask JSON

data:
  root: "./data"
  images_dir: "images"
  bbox_dir: "bbox"
  mask_dir: "mask"

logging:
  level: "INFO"
  file: "logs/processor.log"
```

## Output

After processing, masks will be generated in:
```
data/
├── images/
├── bbox/
└── mask/                    # Generated mask JSON files
    ├── frame_001.json
    └── frame_002.json
```

## Development

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_config.py -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

### TDD Workflow

This project follows strict TDD:
1. **RED**: Write failing tests
2. **GREEN**: Implement to pass tests
3. **REFACTOR**: Optimize while keeping tests green

## Project Structure

```
sam-labelme-processor/
├── cli.py                      # CLI entry point
├── config.yaml                 # Configuration file
├── requirements.txt             # Dependencies
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── data_manager.py    # Dataset management
│   │   ├── labelme_io.py     # Labelme JSON I/O
│   │   └── sam_processor.py   # Main processing logic
│   └── models/
│       ├── __init__.py
│       └── sam_wrapper.py    # SAM model wrapper
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data_manager.py
│   ├── test_labelme_io.py
│   ├── test_sam_processor.py
│   └── test_sam_wrapper.py
└── examples/
    └── sample_config.yaml
```

## License

MIT License

## Author

Generated with TDD methodology following Context7 best practices.
