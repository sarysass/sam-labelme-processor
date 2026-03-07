# 项目架构

## 主处理链

```text
cli.py
  -> Config
  -> DataManager
  -> SAMProcessor
       -> BatchRunner
            -> ItemProcessor
                 -> LabelReader
                 -> SAMWrapper
                      -> UltralyticsSAMBackend
                 -> ResultWriter
                      -> LabelmeIO
```

## 后处理链

```text
optimize_mask_edges.py
  -> EdgeOptimizationConfig
  -> labelme_adapter
       -> decode Labelme mask
       -> edge_refiner
            -> 边缘贴合
            -> 空腔恢复
            -> 薄壳去除
       -> encode Labelme mask
```

## 模块职责

### `src/core`

- `config.py`: 读取配置和默认值
- `data_manager.py`: 扫描数据集并映射图片、bbox、mask 路径
- `types.py`: 共享 workflow 数据结构
- `label_reader.py`: 读取 Labelme bbox JSON / YOLO TXT
- `result_writer.py`: 输出 separate / combined Labelme JSON
- `item_processor.py`: 单张图处理
- `batch_runner.py`: checkpoint、resume、retry、batch 调度
- `sam_processor.py`: 兼容门面，对外保留原接口
- `labelme_io.py`: Labelme 底层格式读写

### `src/models`

- `sam_backend.py`: SAM backend 抽象接口
- `ultralytics_sam_backend.py`: 本地 Ultralytics SAM 实现
- `sam_wrapper.py`: 对上层暴露统一 SAM 调用接口

### `src/postprocess`

- `edge_refiner.py`: 纯算法逻辑
- `labelme_adapter.py`: Labelme mask 文件适配与路径处理

## 4 步算法链

1. 使用 SAM 根据 bbox 生成初始 mask
2. 使用边缘辅助优化贴合真实轮廓
3. 恢复被错误填充的内部空腔
4. 去除空腔周围与外背景相连的薄壳残留

## 兼容策略

- `src/core/sam_processor.py` 保留旧的 public API
- `src/core/mask_edge_optimizer.py` 保留旧的 import 路径，但内部已经改为 re-export `src/postprocess/*`

这样做的目的：

- 先完成结构重组
- 不打断现有 CLI 和测试
- 给后续继续删兼容层留出空间
