# SAM Labelme 处理器

使用 SAM（Segment Anything Model）批量处理带边框的图像，生成 mask 并以 Labelme JSON 格式输出。

## 功能特性

- **批处理**：一次处理多张图像，支持进度跟踪
- **增量处理**：使用 `--resume` 标志跳过已处理的文件
- **灵活输出**：输出独立的 mask 文件或合并的 bbox+mask 文件
- **数据集验证**：处理前验证数据集结构
- **统计信息**：查看数据集统计（总图像数、已处理、待处理）
- **TDD 开发**：采用严格的 TDD 方法论构建

## 安装

### 前置条件

- Python 3.10+
- MicroHunter（用于 SAM 模型）
  - 从本地安装 MicroHunter 项目

### 设置步骤

1. 克隆或导航到项目目录：
   ```bash
   cd /path/to/sam-labelme-processor
   ```

2. 创建并激活虚拟环境：
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. 配置项目：
   ```bash
   # 复制配置文件模板
   cp config.yaml.example config.yaml

   # 编辑 config.yaml，设置正确的路径和参数
   # 重点关注：
   # - sam.weights: SAM 模型权重文件路径
   # - sam.device: 根据你的硬件设置 (mps/cuda/cpu)
   # - data.root: 数据集根目录
   ```

## 使用方法

### 数据目录结构

按照以下结构准备数据目录：
```
data/
├── images/
│   ├── frame_001.jpg
│   └── frame_002.jpg
└── bbox/
    ├── frame_001.json
    └── frame_002.json
```

### 命令

#### 验证数据集
检查数据集结构是否有效：
```bash
python cli.py validate
```

#### 查看统计信息
显示数据集统计：
```bash
python cli.py stats
```

#### 处理图像
为所有待处理的图像生成 mask：
```bash
python cli.py process
```

#### 恢复处理
跳过已处理的图像：
```bash
python cli.py process --resume
```

#### 自定义数据目录
使用自定义数据目录：
```bash
python cli.py process --data-dir /path/to/data
```

## 配置说明

编辑 `config.yaml` 进行自定义配置：

```yaml
sam:
  weights: "/path/to/sam/weights.pt"  # SAM 模型权重路径
  device: "auto"                        # auto, cuda:0, cpu, mps
  imgsz: 1024
  iou_threshold: 0.3

output:
  separate: true                          # 输出独立的 mask JSON
  combine: false                          # 输出合并的 bbox+mask JSON

data:
  root: "./data"                          # 数据根目录
  images_dir: "images"                     # 图像文件夹
  bbox_dir: "bbox"                        # 边框文件夹
  mask_dir: "mask"                        # mask 输出文件夹

logging:
  level: "INFO"
  file: "logs/processor.log"
```

## 输出说明

处理后，mask 将生成在：
```
data/
├── images/
├── bbox/
└── mask/                    # 生成的 mask JSON 文件
    ├── frame_001.json
    └── frame_002.json
```

## 开发

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_config.py -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### TDD 工作流程

本项目严格遵循 TDD 开发：
1. **RED**：编写失败的测试
2. **GREEN**：实现代码使测试通过
3. **REFACTOR**：重构优化，保持测试通过

## 项目结构

```
sam-labelme-processor/
├── cli.py                      # CLI 入口
├── config.yaml                 # 配置文件
├── config.yaml.example          # 配置文件模板
├── requirements.txt             # 依赖列表
├── README.md                  # 项目说明
├── AGENTS.md                  # Agent 开发指南
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # 配置管理
│   │   ├── data_manager.py    # 数据集管理
│   │   ├── labelme_io.py     # Labelme JSON I/O
│   │   └── sam_processor.py   # 主处理逻辑
│   └── models/
│       ├── __init__.py
│       └── sam_wrapper.py    # SAM 模型封装
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

## 许可证

MIT License

## 作者

采用 TDD 方法论和 Context7 最佳实践开发。
