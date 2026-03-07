# SAM Labelme Processor Refactor Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Reorganize the project into a clearer, testable architecture without changing CLI behavior, data compatibility, output format, or current mask optimization results.

**Architecture:** Split the current "god modules" into smaller services around four boundaries: dataset access, single-item processing, batch orchestration, and postprocessing. Keep behavior stable by adding regression tests first, then migrate code behind compatibility facades so the public entry points remain unchanged throughout the refactor.

**Tech Stack:** Python 3.10+, pytest, OpenCV, NumPy, YAML, Ultralytics SAM, Labelme JSON.

---

## Scope

This refactor covers:

- `cli.py`
- `optimize_mask_edges.py`
- `src/core/`
- `src/models/`
- `tests/`
- `requirements.txt`
- `config.yaml.example`
- `README.md`

This refactor does **not** change:

- Existing CLI command names
- Existing config keys unless the key is additive
- Existing Labelme output schema
- Existing optimization algorithm behavior
- Existing dataset directory layout

## Baseline Constraints

Before making any production change, preserve these current invariants:

1. `python cli.py process`
2. `python cli.py validate`
3. `python cli.py stats`
4. `python optimize_mask_edges.py --input ... --output ...`
5. `pytest tests/ -q`

Current expected baseline:

```bash
pytest tests/ -q
```

Expected:

```text
45 passed
```

## Target File Layout

```text
sam-labelme-processor/
├── cli.py
├── optimize_mask_edges.py
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── data_manager.py
│   │   ├── types.py
│   │   ├── label_reader.py
│   │   ├── result_writer.py
│   │   ├── item_processor.py
│   │   ├── batch_runner.py
│   │   ├── sam_processor.py
│   │   └── labelme_io.py
│   ├── models/
│   │   ├── sam_backend.py
│   │   ├── sam_wrapper.py
│   │   └── ultralytics_sam_backend.py
│   └── postprocess/
│       ├── __init__.py
│       ├── edge_refiner.py
│       └── labelme_adapter.py
├── tests/
│   ├── test_label_reader.py
│   ├── test_result_writer.py
│   ├── test_item_processor.py
│   ├── test_batch_runner.py
│   ├── test_edge_refiner.py
│   ├── test_labelme_adapter.py
│   └── test_ultralytics_sam_backend.py
└── docs/
    └── plans/
```

## Migration Strategy

Use a strangler pattern:

1. Freeze behavior with tests.
2. Extract read/write helpers first.
3. Extract single-item processing next.
4. Extract batch orchestration after single-item logic is stable.
5. Split postprocessing into algorithm and adapter layers.
6. Introduce a local Ultralytics SAM backend behind the existing wrapper contract.
7. Remove all runtime dependency on `MicroHunter`.
8. Keep `sam_processor.py` as a compatibility facade until the end.

## Rollback Strategy

After each task:

1. Run focused tests for that task.
2. Run `pytest tests/ -q`.
3. If failures appear outside the touched area, revert only the last task.
4. Do not continue to the next extraction until the full suite is green.

## Commit Strategy

Use one commit per task group:

1. `test: freeze refactor baseline behavior`
2. `refactor(core): extract label reader`
3. `refactor(core): extract result writer`
4. `refactor(core): extract item processor`
5. `refactor(core): extract batch runner`
6. `refactor(postprocess): split edge refiner and labelme adapter`
7. `feat(models): add local ultralytics SAM backend`
8. `refactor(models): remove MicroHunter dependency from wrapper`
9. `refactor(cli): centralize runtime composition`
10. `docs: update architecture and refactor notes`

---

### Task 1: Freeze Current Behavior With Regression Tests

**Files:**
- Create: `tests/test_label_reader.py`
- Create: `tests/test_result_writer.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_sam_processor.py`
- Modify: `tests/test_mask_edge_optimizer.py`

**Goal:** Add coverage for the behaviors that must not change during refactor.

**Step 1: Add label parsing tests**

Add tests for:

- YOLO `.txt` parsing to bbox coordinates
- Labelme rectangle JSON loading
- Empty-line handling in YOLO files
- `class_id` conversion to `class_<id>`

**Step 2: Add result writing tests**

Add tests for:

- `imagePath` stored relative to output JSON path
- separate mask JSON output shape structure
- combined output preserving bbox + mask entries

**Step 3: Expand `SAMProcessor` behavior tests**

Add tests for:

- empty labels skipped when `skip_empty_labels=True`
- no-bbox returns failure
- retry returns failure after max retries
- checkpoint ignored when stale

**Step 4: Expand optimizer tests**

Add tests for:

- `--enable-cavity-recovery` without shell removal
- single-file `--output` behavior
- file-to-file output preserves image loading correctness

**Step 5: Run focused tests**

Run:

```bash
pytest tests/test_sam_processor.py tests/test_mask_edge_optimizer.py tests/test_cli.py -v
```

Expected: all pass

**Step 6: Run full suite**

Run:

```bash
pytest tests/ -q
```

Expected: full suite green

**Step 7: Commit**

```bash
git add tests/test_label_reader.py tests/test_result_writer.py tests/test_cli.py tests/test_sam_processor.py tests/test_mask_edge_optimizer.py
git commit -m "test: freeze refactor baseline behavior"
```

---

### Task 2: Introduce Shared Workflow Types

**Files:**
- Create: `src/core/types.py`
- Modify: `src/core/sam_processor.py`
- Modify: `tests/test_sam_processor.py`

**Goal:** Move shared workflow models out of `SAMProcessor` so they can be reused by new services.

**Create these types in `src/core/types.py`:**

- `ProcessingResult`
- `LoadedAnnotations`
- `OutputPaths`

Suggested shape:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .labelme_io import BBoxShape, ImageInfo, MaskShape
from .data_manager import DataItem


@dataclass
class LoadedAnnotations:
    image_info: ImageInfo
    bbox_shapes: List[BBoxShape]


@dataclass
class OutputPaths:
    mask_json_path: Optional[Path]
    combined_json_path: Optional[Path]


@dataclass
class ProcessingResult:
    data_item: DataItem
    success: bool
    mask_shapes: List[MaskShape]
    error_message: Optional[str] = None
```

**Step 1: Write failing import test**

Move `ProcessingResult` import expectations from `SAMProcessor` tests to `src.core.types`.

**Step 2: Implement `types.py`**

Keep the fields identical to current behavior. Do not change result semantics.

**Step 3: Update `SAMProcessor` to import `ProcessingResult`**

Do not change any logic yet.

**Step 4: Run tests**

```bash
pytest tests/test_sam_processor.py -v
pytest tests/ -q
```

**Step 5: Commit**

```bash
git add src/core/types.py src/core/sam_processor.py tests/test_sam_processor.py
git commit -m "refactor(core): extract shared workflow types"
```

---

### Task 3: Extract Label Reading

**Files:**
- Create: `src/core/label_reader.py`
- Modify: `src/core/sam_processor.py`
- Modify: `tests/test_label_reader.py`
- Modify: `tests/test_sam_processor.py`

**Goal:** Move label-format parsing out of `SAMProcessor`.

**Create `src/core/label_reader.py` with:**

- `class LabelReader`
- `read_yolo_txt_labels(...)`
- `read_label_file(...)`

Suggested API:

```python
class LabelReader:
    def read_label_file(
        self,
        label_path: Path,
        image_path: Path,
        image_height: int,
        image_width: int,
    ) -> LoadedAnnotations:
        ...
```

**Step 1: Copy parsing logic from `SAMProcessor`**

Move:

- `_read_yolo_txt_labels`
- `_read_label_file`

Keep all edge-case behavior identical.

**Step 2: Convert return values to `LoadedAnnotations`**

Use `ImageInfo` and `BBoxShape` exactly as before.

**Step 3: Update `SAMProcessor` constructor**

Allow:

```python
label_reader: Optional[LabelReader] = None
```

Default to `LabelReader()` when not provided.

**Step 4: Replace internal calls**

Route all label parsing through `self.label_reader`.

**Step 5: Run focused tests**

```bash
pytest tests/test_label_reader.py tests/test_sam_processor.py -v
```

**Step 6: Run full suite**

```bash
pytest tests/ -q
```

**Step 7: Commit**

```bash
git add src/core/label_reader.py src/core/sam_processor.py tests/test_label_reader.py tests/test_sam_processor.py
git commit -m "refactor(core): extract label reader"
```

---

### Task 4: Extract Result Writing

**Files:**
- Create: `src/core/result_writer.py`
- Modify: `src/core/sam_processor.py`
- Modify: `tests/test_result_writer.py`
- Modify: `tests/test_sam_processor.py`

**Goal:** Move output path construction and JSON writing out of `SAMProcessor`.

**Create `src/core/result_writer.py` with:**

- `class ResultWriter`
- `_build_output_image_info(...)`
- `write_outputs(...)`

Suggested API:

```python
class ResultWriter:
    def write_outputs(
        self,
        data_item: DataItem,
        image_path: Path,
        image_info: ImageInfo,
        bbox_shapes: List[BBoxShape],
        mask_shapes: List[MaskShape],
    ) -> OutputPaths:
        ...
```

Constructor should accept:

- `output_separate`
- `output_combine`
- `combined_dir`

**Step 1: Write failing tests**

Test:

- separate-only
- combined-only
- both enabled
- relative `imagePath` generation

**Step 2: Move write logic from `SAMProcessor`**

Move:

- `_build_output_image_info`
- separate write block
- combined write block

Do not change file names or output structure.

**Step 3: Inject `ResultWriter` into `SAMProcessor`**

Default to a concrete instance when not supplied.

**Step 4: Run tests**

```bash
pytest tests/test_result_writer.py tests/test_sam_processor.py -v
pytest tests/ -q
```

**Step 5: Commit**

```bash
git add src/core/result_writer.py src/core/sam_processor.py tests/test_result_writer.py tests/test_sam_processor.py
git commit -m "refactor(core): extract result writer"
```

---

### Task 5: Extract Single-Item Processing

**Files:**
- Create: `src/core/item_processor.py`
- Modify: `src/core/sam_processor.py`
- Modify: `tests/test_item_processor.py`
- Modify: `tests/test_sam_processor.py`

**Goal:** Move per-image workflow into a dedicated service.

**Create `src/core/item_processor.py` with:**

- `class ItemProcessor`
- `process(data_item: DataItem) -> ProcessingResult`

Constructor should accept:

- `data_manager`
- `sam_wrapper`
- `label_reader`
- `result_writer`
- `skip_empty_labels`

This class should own:

- image loading
- label loading
- SAM invocation
- mask object creation
- output writing

This class should **not** own:

- checkpointing
- multiprocessing
- global batch progress

**Step 1: Move `_process_single_item_internal` into `ItemProcessor`**

Split into private helpers if needed:

- `_load_image`
- `_build_mask_shapes`

**Step 2: Keep return values identical**

Important:

- empty-label skip still returns `success=True, mask_shapes=[]`
- label missing still returns `success=False`
- error messages should remain semantically equivalent

**Step 3: Keep retry outside this class**

Do not move retry here. That belongs in the next task.

**Step 4: Update `SAMProcessor` to delegate single-item work**

It should hold an `ItemProcessor` instance and call it.

**Step 5: Run tests**

```bash
pytest tests/test_item_processor.py tests/test_sam_processor.py -v
pytest tests/ -q
```

**Step 6: Commit**

```bash
git add src/core/item_processor.py src/core/sam_processor.py tests/test_item_processor.py tests/test_sam_processor.py
git commit -m "refactor(core): extract item processor"
```

---

### Task 6: Extract Batch Orchestration

**Files:**
- Create: `src/core/batch_runner.py`
- Modify: `src/core/sam_processor.py`
- Modify: `tests/test_batch_runner.py`
- Modify: `tests/test_sam_processor.py`

**Goal:** Move retry, checkpoint, cache, preload, and worker orchestration out of `SAMProcessor`.

**Create `src/core/batch_runner.py` with:**

- `class BatchRunner`
- `process_batch(...)`
- `_load_checkpoint(...)`
- `_save_checkpoint(...)`
- `_get_batch_results(...)`
- `_should_fallback_to_single_worker(...)`

Constructor should accept:

- `data_manager`
- `item_processor`
- `batch_size`
- `num_workers`
- `enable_checkpoint`
- `checkpoint_interval`
- `enable_resume`
- `max_retries`
- `retry_delay`
- `memory_limit_gb`
- `enable_memory_watch`
- `preload_images`
- `image_cache_size`

**Important design choice**

Keep retry at the batch-runner layer by wrapping calls to `item_processor.process`.

**Step 1: Add batch-runner tests**

Cover:

- checkpoint stale mismatch resets to zero
- fallback to single worker when memory exceeds limit
- retry stops after max retries
- progress result count matches input

**Step 2: Move orchestration logic**

Move:

- checkpoint management
- batch slicing
- multiprocessing pool dispatch
- retry wrapper

**Step 3: Keep `SAMProcessor.process_batch` public API**

`SAMProcessor` should delegate to `BatchRunner.process_batch`.

**Step 4: Run tests**

```bash
pytest tests/test_batch_runner.py tests/test_sam_processor.py -v
pytest tests/ -q
```

**Step 5: Commit**

```bash
git add src/core/batch_runner.py src/core/sam_processor.py tests/test_batch_runner.py tests/test_sam_processor.py
git commit -m "refactor(core): extract batch runner"
```

---

### Task 7: Reduce `SAMProcessor` To Compatibility Facade

**Files:**
- Modify: `src/core/sam_processor.py`
- Modify: `tests/test_sam_processor.py`

**Goal:** Keep old imports and public methods valid while stripping `SAMProcessor` down to orchestration only.

**End-state for `SAMProcessor`:**

- Constructor wires:
  - `LabelReader`
  - `ResultWriter`
  - `ItemProcessor`
  - `BatchRunner`
- `process_single(...)` delegates to `item_processor.process(...)`
- `process_batch(...)` delegates to `batch_runner.process_batch(...)`

**Step 1: Delete migrated private methods from `SAMProcessor`**

After previous tasks pass, remove duplicated implementations:

- label parsing helpers
- output writing helpers
- retry helpers
- checkpoint helpers

**Step 2: Keep constructor arguments backward compatible**

Do not change `cli.py` call sites.

**Step 3: Run tests**

```bash
pytest tests/test_sam_processor.py -v
pytest tests/ -q
```

**Step 4: Commit**

```bash
git add src/core/sam_processor.py tests/test_sam_processor.py
git commit -m "refactor(core): shrink sam processor to facade"
```

---

### Task 8: Split Postprocessing Into Algorithm And Labelme Adapter

**Files:**
- Create: `src/postprocess/__init__.py`
- Create: `src/postprocess/edge_refiner.py`
- Create: `src/postprocess/labelme_adapter.py`
- Modify: `src/core/mask_edge_optimizer.py`
- Modify: `optimize_mask_edges.py`
- Modify: `tests/test_mask_edge_optimizer.py`
- Create: `tests/test_edge_refiner.py`
- Create: `tests/test_labelme_adapter.py`

**Goal:** Separate pure optimization logic from Labelme file adaptation.

**Move to `src/postprocess/edge_refiner.py`:**

- `EdgeOptimizationConfig`
- `_odd_kernel_size`
- `_ellipse_kernel`
- `smooth_binary_mask`
- `_connected_component_count`
- `_find_enclosed_holes`
- `recover_internal_cavities`
- `_remove_background_like_shells`
- `refine_mask_with_edges`

**Move to `src/postprocess/labelme_adapter.py`:**

- `decode_labelme_mask`
- `optimize_labelme_mask_file`
- `collect_json_files`
- `build_output_path`
- `default_output_path`

**Compatibility rule**

Keep `src/core/mask_edge_optimizer.py` as a thin re-export shim in this refactor:

```python
from ..postprocess.edge_refiner import ...
from ..postprocess.labelme_adapter import ...
```

This preserves existing imports while tests and scripts move over.

**Step 1: Write new tests first**

Move algorithm-focused tests to `tests/test_edge_refiner.py`.

Move file adaptation tests to `tests/test_labelme_adapter.py`.

**Step 2: Extract code without behavior changes**

Do not rename config flags or alter thresholds.

**Step 3: Update `optimize_mask_edges.py` imports**

Import from `src.postprocess...` instead of `src.core.mask_edge_optimizer`.

**Step 4: Keep old module passing**

Add one test that importing from `src.core.mask_edge_optimizer` still works during the compatibility window.

**Step 5: Run tests**

```bash
pytest tests/test_mask_edge_optimizer.py tests/test_edge_refiner.py tests/test_labelme_adapter.py -v
pytest tests/ -q
```

**Step 6: Commit**

```bash
git add src/postprocess/__init__.py src/postprocess/edge_refiner.py src/postprocess/labelme_adapter.py src/core/mask_edge_optimizer.py optimize_mask_edges.py tests/test_mask_edge_optimizer.py tests/test_edge_refiner.py tests/test_labelme_adapter.py
git commit -m "refactor(postprocess): split edge refiner and labelme adapter"
```

---

### Task 9: Add Local Ultralytics SAM Backend

**Files:**
- Create: `src/models/sam_backend.py`
- Create: `src/models/ultralytics_sam_backend.py`
- Modify: `src/models/sam_wrapper.py`
- Modify: `requirements.txt`
- Create: `tests/test_ultralytics_sam_backend.py`
- Modify: `tests/test_sam_wrapper.py`

**Goal:** Reimplement the currently used subset of SAM inference inside this repo so the project no longer depends on `MicroHunter` for segmentation.

**Important rule**

Do **not** copy code from the `MicroHunter` repository. Recreate the required behavior using the public `ultralytics` API and this repo's existing wrapper contract.

**Required functional scope**

The local backend only needs to support the behavior this project actually uses today:

- loading SAM weights
- box-prompt inference
- preserving one-result-per-input-box order
- returning:
  - `masks`: `(N, H, W)` uint8 array
  - `contours`: `List[np.ndarray]`

Point prompts and full-image mode are optional and out of scope for this refactor unless already trivial.

**Create `src/models/sam_backend.py`**

Define a minimal interface or protocol:

```python
from typing import Any, Dict, List
import numpy as np


class SAMBackend:
    def predict(self, image: np.ndarray, bboxes: List[List[float]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
```

**Create `src/models/ultralytics_sam_backend.py`**

Implement:

- lazy import of `from ultralytics import SAM`
- model load
- bbox clipping
- invalid-box replacement with tiny valid box to preserve index order
- empty-result handling
- output normalization into the same structure used by `SAMWrapper`

**Step 1: Add failing tests first**

Cover:

- model constructor called with configured weights
- `predict` passes image and bboxes through Ultralytics
- invalid boxes are repaired instead of dropped
- output order matches input bbox order
- no masks returns empty masks in stable shape

Use monkeypatch/fake Ultralytics objects. Do not require a real model file.

**Step 2: Add `ultralytics` dependency**

Update `requirements.txt`:

```text
ultralytics>=8.3.0
```

Use the actual version range you validate locally.

**Step 3: Implement backend**

Preserve current project contract:

```python
[
    {
        "mask": np.ndarray,
        "contour": np.ndarray,
        "bbox": bbox,
    }
]
```

**Step 4: Refactor `SAMWrapper` into a facade**

During this task, keep `SAMWrapper` public constructor intact.

Add constructor option:

```python
backend: Optional[SAMBackend] = None
```

Default behavior for this task:

- if backend is explicitly supplied, use it
- otherwise instantiate `UltralyticsSAMBackend`

**Step 5: Run tests**

```bash
pytest tests/test_ultralytics_sam_backend.py tests/test_sam_wrapper.py -v
pytest tests/ -q
```

**Step 6: Commit**

```bash
git add src/models/sam_backend.py src/models/ultralytics_sam_backend.py src/models/sam_wrapper.py requirements.txt tests/test_ultralytics_sam_backend.py tests/test_sam_wrapper.py
git commit -m "feat(models): add local ultralytics SAM backend"
```

---

### Task 10: Remove `MicroHunter` Dependency From The Wrapper

**Files:**
- Modify: `src/models/sam_wrapper.py`
- Modify: `src/core/config.py`
- Modify: `cli.py`
- Modify: `tests/test_sam_wrapper.py`
- Modify: `tests/test_config.py`
- Modify: `README.md`
- Modify: `config.yaml.example`

**Goal:** Make the project fully independent from `MicroHunter` and remove local-machine path assumptions.

**Required behavior**

- `SAMWrapper` must not import `microhunter`
- `SAMWrapper` must not mutate `sys.path`
- no config field should reference `microhunter_path`
- all SAM inference should route through the local Ultralytics backend

**Step 1: Remove old path/import logic**

Delete:

- `MICROHUNTER_PATH`
- import-time `sys.path.insert(...)`
- `MICROHUNTER_AVAILABLE`
- legacy `UltralyticsSAMPredictor` import fallback logic

**Step 2: Simplify config**

Do not add `sam.microhunter_path`.

If needed, add only additive config related to the local backend, for example:

```python
"backend": "ultralytics"
```

Only do this if the backend selection is still useful after Task 9. Otherwise skip it and keep config unchanged.

**Step 3: Update CLI wiring**

Ensure `cli.py` constructs `SAMWrapper` without any `MicroHunter` dependency.

**Step 4: Update tests**

Cover:

- wrapper initializes the local backend
- import failure from `ultralytics` raises a clear message
- old `MicroHunter` path behavior no longer exists

**Step 5: Run tests**

```bash
pytest tests/test_sam_wrapper.py tests/test_config.py -v
pytest tests/ -q
```

**Step 6: Commit**

```bash
git add src/models/sam_wrapper.py src/core/config.py cli.py tests/test_sam_wrapper.py tests/test_config.py README.md config.yaml.example
git commit -m "refactor(models): remove MicroHunter dependency from wrapper"
```

---

### Task 11: Tighten CLI Composition

**Files:**
- Modify: `cli.py`
- Modify: `tests/test_cli.py`

**Goal:** Remove repeated object-construction code from CLI commands.

**Add helper functions in `cli.py`:**

- `load_runtime_config(args) -> Config`
- `build_data_manager(config, args) -> DataManager`
- `build_processor(config, data_manager) -> SAMProcessor`

**Behavior rules**

- `process`, `validate`, and `stats` must produce the same user-visible output
- `process --resume` semantics must not change

**Step 1: Refactor only composition**

Do not move business logic back into CLI.

**Step 2: Re-run CLI tests**

```bash
pytest tests/test_cli.py -v
pytest tests/ -q
```

**Step 3: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "refactor(cli): centralize runtime composition"
```

---

### Task 12: Documentation And Repository Hygiene

**Files:**
- Modify: `README.md`
- Create: `docs/architecture.md`
- Modify: `.gitignore`

**Goal:** Align docs and repo hygiene with the new structure.

**Step 1: Update README**

Document:

- new module responsibilities
- main pipeline
- postprocessing pipeline
- local Ultralytics SAM backend configuration

**Step 2: Add `docs/architecture.md`**

Include:

- module map
- data flow
- four-step mask optimization chain

**Step 3: Update `.gitignore`**

Ignore:

- `__pycache__/`
- `.DS_Store`
- generated compare images if desired

**Step 4: Remove tracked trash files only if safe**

If tracked:

- `src/.DS_Store`
- stale `__pycache__`

Only delete after confirming they are generated artifacts.

**Step 5: Run final verification**

```bash
pytest tests/ -q
python cli.py --help
python optimize_mask_edges.py --help
```

Expected:

- tests green
- both CLIs print usage

**Step 6: Commit**

```bash
git add README.md docs/architecture.md .gitignore
git commit -m "docs: update architecture and repository hygiene"
```

---

## End-of-Refactor Acceptance Checklist

All of the following must be true before calling the refactor complete:

- `pytest tests/ -q` passes
- `cli.py` commands behave the same from the user perspective
- `optimize_mask_edges.py` behaves the same from the user perspective
- `SAMProcessor` public constructor still works with the current CLI
- `src/core/sam_processor.py` no longer contains:
  - label parsing
  - output writing
  - checkpoint persistence
  - multiprocessing dispatch
  - direct retry loops
- `src/postprocess/edge_refiner.py` contains pure mask algorithms only
- `src/postprocess/labelme_adapter.py` contains file adaptation logic only
- `src/models/ultralytics_sam_backend.py` exists and is covered by tests
- `SAMWrapper` no longer imports or references `MicroHunter`
- `requirements.txt` declares the direct SAM runtime dependency used by this repo
- docs describe the new architecture accurately

## Final Verification Script

Run from repo root:

```bash
pytest tests/ -q && \
python cli.py --help >/tmp/cli_help.txt && \
python optimize_mask_edges.py --help >/tmp/opt_help.txt
```

Optional live-data verification on a safe sample:

```bash
python optimize_mask_edges.py \
  --input "data/select/select_masks_opt/0002/0988-5.json" \
  --output "data/select/tmp_refactor_check/0002/0988-5.json" \
  --enable-cavity-recovery
```

Expected:

- command exits `0`
- output JSON created
- image path remains readable from the output JSON location

Optional local inference smoke check if weights are available:

```bash
python cli.py process --config config.yaml --resume
```

Expected:

- command starts without `MicroHunter` import errors
- SAM model loads through the local Ultralytics backend
- outputs remain valid Labelme mask JSON files

## Execution Notes

- Do not merge tasks together during implementation.
- Do not rename public CLI files during this refactor.
- Do not change the optimization thresholds as part of the structural work.
- Prefer keeping compatibility shims for one iteration rather than breaking imports.
- If a task exposes an unexpected hidden behavior change, stop and add a regression test before proceeding.
- Do not copy source from the `MicroHunter` repository into this project; reimplement behavior against the public Ultralytics API.
