# CLAUDE.md

This file provides guidance to Claude (claude.ai/code) when working with code in this repository.

## 项目概述

ContextGen 是一个基于 FLUX.1-Kontext 的多实例身份一致性图像生成框架（ICLR 2026），通过用户提供的参考图像，在布局控制下生成保持身份一致性的多实例图像。

## 环境配置

```bash
conda create contextgen python=3.12 -y
conda activate contextgen
pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件（参考 `.env_template`）：

```bash
KONTEXT_MODEL_PATH="path_to_kontext_model"   # FLUX.1-Kontext-dev 模型路径
ADAPTER_PATH="path_to_contextgen_adapter"     # ContextGen adapter 路径
WANDB_API_KEY="your_wandb_api_key"            # 训练时需要
# GUI 额外需要：
BEN_CKPT_PATH="path_to_ben2_model"            # BEN2 抠图模型
FLUX_MODEL_PATH="path_to_flux_model"          # FLUX.1-dev（可选，文本生成资产）
```

## 常用命令

### 推理

```bash
# 使用默认 demo（images/input/segment_info.json）
python inference.py

# 自定义参数
python inference.py --input_dir ./images/input --output_dir ./images/output --num_samples 4 --image_size 768
```

输入目录需包含 `segment_info.json`（描述实例 bbox 和参考图像路径）及对应图像文件。

### 训练

```bash
# 先配置 train/config/config.yaml，然后：
python src/model/train.py
```

### GUI

```bash
# 终端 1：前端
cd gui/frontend
npm install  # 首次运行
npm run dev

# 终端 2：后端
python gui/backend/app.py
```

访问 `http://localhost:5173`（远程服务器需转发端口 5173 和 5000）。

## 代码架构

### 核心数据流（推理）

```
inference.py
  └─ src/model/generate.py: load_model() + generate()
       ├─ load_model(): 加载 FluxKontextPipeline + LoRA adapter (safetensors)
       │    └─ src/flux/attention_processor.py: FluxAttnProcessor2_0_Attention 替换注意力处理器
       └─ generate():
            ├─ src/model/pipeline_preprocess.py: 准备参考图潜变量、位置编码、注意力掩码
            ├─ src/flux/pipeline_forward.py: fluxkontext_pipeline_forward（自定义前向传播）
            └─ src/flux/transformer_flux.py: FluxTransformer2DModel_forward（transformer 前向）
```

### 核心数据流（训练）

```
src/model/train.py
  └─ ContextGenModel (Lightning Module, src/model/model.py)
       ├─ FluxKontextPipeline（冻结 VAE、text encoder）
       ├─ LoRA 注入 transformer 的 attention/MLP 层（PEFT）
       ├─ MigDataset (src/model/data.py)：多实例图像数据集
       └─ TrainingCallback (src/model/callbacks.py)：日志/保存检查点
```

### 模块说明

- **`src/flux/`**: FLUX.1-Kontext pipeline 的定制扩展
  - `pipeline_flux_kontext.py`: 继承自 diffusers 的 FluxKontextPipeline
  - `attention_processor.py`: 自定义注意力处理器，支持多实例注意力掩码
  - `pipeline_forward.py`: 带参考图条件的自定义 pipeline 前向
  - `transformer_flux.py`: 带布局锚定的 transformer 前向
  - `embeddings.py`: 位置编码扩展（支持参考图 PE）

- **`src/model/`**: 训练与推理逻辑
  - `model.py`: `ContextGenModel`（Lightning Module），管理 LoRA 训练
  - `generate.py`: 推理入口，`load_model()` 和 `generate()` 函数
  - `pipeline_preprocess.py`: 多实例预处理（参考图潜变量、位置编码、注意力掩码）
  - `data.py`: `MigDataset`，加载多实例图像数据集
  - `train.py`: 训练脚本入口（DDP 多 GPU 支持）

- **`src/utils/`**: 工具函数
  - `image_process.py`: 图像尺寸对齐（`uniform_resize_16x`）等
  - `text_process.py`: Kontext 编辑模板生成
  - `file_utils.py`: 配置文件读取、检查点查找

- **`gui/backend/app.py`**: Flask 后端，集成 BEN2 抠图 + ContextGen 推理
- **`gui/frontend/`**: Vite + React 前端
- **`bench/`**: LaMIC-Bench+ 评测脚本

### 训练配置

训练参数在 `train/config/config.yaml` 中配置，关键项：
- `flux_path`: FLUX.1-Kontext 模型路径
- `train.lora_config.r`: LoRA rank（默认 512）
- `train.train_method`: `"sft"` 或 `"dpo"`
- `train.dataset.path`: IMIG-Dataset 路径
- `train.save_path`: 检查点保存路径

### input 数据格式

推理输入目录需包含 `segment_info.json`：
```json
[{
  "caption": "描述文本",
  "width": 768,
  "layout_image": "layout.png",  // 可选布局图
  "instances": [
    {"image": "ref1.png", "bbox": [x1, y1, x2, y2], "mask": "mask1.png"}
  ]
}]
```

bbox 格式为 `[x1, y1, x2, y2]`（相对于原始 `width` 的像素坐标，推理时会缩放至 `image_size`）。

## GPU 内存需求

- 推理：~35-40GB（FLUX.1-Kontext + ContextGen adapter）
- GUI（含文本生成资产功能）：额外 ~30GB

## 推理质量建议

- 推荐分辨率：768×768 或 512×512
- 使用包含实例间交互关系的丰富 prompt
- 结果质量不佳时尝试不同随机种子
