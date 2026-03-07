# SAM Labelme 处理器

使用本地 `Ultralytics SAM` 根据 bbox 批量生成 Labelme `mask` JSON，并支持对已有 mask 做边缘辅助后处理。

## 当前能力

- 批量读取图片和 bbox 标签
- 支持 Labelme `rectangle` JSON 和 YOLO `.txt` 输入
- 调用本项目内置 SAM backend 生成初始 mask
- 输出独立 mask JSON 或 bbox+mask 合并 JSON
- 支持 checkpoint、resume、重试、基础内存保护
- 支持已有 mask 的后处理优化：
  - 边缘贴合
  - 内部空腔恢复
  - 薄壳残留去除

## 依赖

- Python 3.10+
- `ultralytics`
- `opencv-python`
- `numpy`

安装：

```bash
pip install -r requirements.txt
```

## 配置

复制模板：

```bash
cp config.yaml.example config.yaml
```

关键配置：

```yaml
sam:
  weights: "weights/sam2.1_t.pt"
  device: "auto"
  imgsz: 1024
  iou_threshold: 0.3

output:
  separate: true
  combine: false

data:
  root: "./data"
  images_dir: "images"
  bbox_dir: "bbox"
  bbox_extension: ".json"
  mask_dir: "mask"
  combined_dir: "output/combined"
```

YOLO TXT 示例：

```yaml
data:
  root: "./data/select"
  images_dir: "select images"
  bbox_dir: "select labels"
  bbox_extension: ".txt"
  mask_dir: "select masks"
```

## CLI

验证数据集：

```bash
python cli.py validate
```

查看统计：

```bash
python cli.py stats
```

批量生成 mask：

```bash
python cli.py process
```

恢复处理：

```bash
python cli.py process --resume
```

## 后处理 CLI

对已有 Labelme mask JSON 做优化：

```bash
python optimize_mask_edges.py \
  --input data/select/select_masks_opt/0002/0988-5.json \
  --output data/select/tmp_refined/0002/0988-5.json \
  --enable-cavity-recovery
```

常用开关：

- `--enable-cavity-recovery`
- `--enable-shell-removal`
- `--shell-max-thickness`
- `--shell-background-cost-multiplier`

## 4 步算法链

1. SAM 初始分割
2. 边缘辅助贴合
3. 内部空腔恢复
4. 薄壳残留去除

## 项目结构

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
│       ├── edge_refiner.py
│       └── labelme_adapter.py
└── tests/
```

## 测试

```bash
pytest tests/ -q
```

## 架构说明

更详细的模块关系见：

- [docs/architecture.md](/Users/shali/projects/tools/sam-labelme-processor/docs/architecture.md)
