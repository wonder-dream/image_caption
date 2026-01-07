# Image Caption - 图像描述生成项目

基于 DeepFashion-MultiModal 数据集的图像描述生成模型，实现了 ViT-Transformer 和 Grid-Transformer 两种架构。

## 📁 项目结构

```
image_caption/
├── models/                          # 模型定义
│   ├── vit_transformer_model.py     # ViT + Transformer 模型
│   └── grid_transformer_model.py    # Grid (CNN) + Transformer 模型
│
├── scripts/                         # 训练和推理脚本
│   ├── train_vit_transformer.py              # ViT 基础训练 (交叉熵)
│   ├── train_grid_transformer.py             # Grid 基础训练 (交叉熵)
│   ├── train_vit_transformer_optimized.py    # ViT 优化版训练
│   ├── train_grid_transformer_optimized.py   # Grid 优化版训练
│   ├── train_vit_transformer_scst_optimized.py   # ViT SCST 强化学习训练
│   ├── train_grid_transformer_scst_optimized.py  # Grid SCST 强化学习训练
│   ├── evaluate_model.py            # 模型评估脚本 (CIDEr/METEOR/ROUGE-L)
│   └── inference.py                 # 单图推理脚本
│
├── utils/                           # 工具模块
│   ├── deepfashion_dataset.py       # 数据集类
│   ├── prepare_data.py              # 数据预处理
│   ├── unzip_dataset.py             # 数据集解压
│   ├── eval_metrics.py              # 评测指标
│   ├── scst_loss.py                 # SCST 损失函数
│   └── optimizations.py             # 优化工具 (Label Smoothing, EMA等)
│
├── data/                            # 数据目录
│   ├── images/                      # 图像文件夹
│   ├── vocab.json                   # 词典
│   ├── train_data.json              # 训练集
│   ├── val_data.json                # 验证集
│   └── test_data.json               # 测试集
│
├── checkpoints/                     # 模型检查点
│   ├── vit_transformer/             # ViT 基础模型
│   ├── grid_transformer/            # Grid 基础模型
│   ├── vit_transformer_optimized/   # ViT 优化版
│   └── grid_transformer_optimized/  # Grid 优化版
│
├── requirements.txt                 # 依赖
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 解压数据集
python utils/unzip_dataset.py

# 生成词典和数据集划分
python utils/prepare_data.py
```

### 3. 训练模型

#### 基础训练（交叉熵）
```bash
# ViT + Transformer
python scripts/train_vit_transformer.py

# Grid + Transformer
python scripts/train_grid_transformer.py
```

#### 优化版训练（推荐）
```bash
# ViT 优化版
python scripts/train_vit_transformer_optimized.py

# Grid 优化版
python scripts/train_grid_transformer_optimized.py
```

#### SCST 强化学习微调
需要先完成优化版训练，再进行 SCST 微调：
```bash
# ViT SCST
python scripts/train_vit_transformer_scst_optimized.py

# Grid SCST  
python scripts/train_grid_transformer_scst_optimized.py
```

### 4. 模型评估

在测试集上评估模型，输出 CIDEr、METEOR、ROUGE-L 指标：

```bash
# 评估 Grid 模型
python scripts/evaluate_model.py --checkpoint checkpoints/grid_transformer/best_model.pth --model_type grid

# 评估 ViT 模型
python scripts/evaluate_model.py --checkpoint checkpoints/vit_transformer/best_model.pth --model_type vit

# 评估优化版模型
python scripts/evaluate_model.py --checkpoint checkpoints/grid_transformer_optimized/best_model.pth --model_type grid
```

### 5. 单图推理

对单张图片生成描述：

```bash
# 使用 Grid 模型
python scripts/inference.py --image path/to/image.jpg --checkpoint checkpoints/grid_transformer/best_model.pth --model_type grid

# 使用 Beam Search
python scripts/inference.py --image path/to/image.jpg --checkpoint checkpoints/vit_transformer/best_model.pth --model_type vit --method beam_search --beam_size 5

# 保存可视化结果
python scripts/inference.py --image path/to/image.jpg --checkpoint checkpoints/grid_transformer/best_model.pth --model_type grid --save output.png
```

## 🧠 模型架构

### ViT + Transformer
- **编码器**: Vision Transformer (ViT-B/16)，将图像分割为 16x16 patches
- **解码器**: 6 层 Transformer Decoder
- **特点**: 全局注意力，适合捕捉图像整体语义

### Grid + Transformer  
- **编码器**: ResNet-101 CNN，提取 7x7 网格特征
- **解码器**: 6 层 Transformer Decoder
- **特点**: 保留空间信息，渐进式 CNN 解冻

## 🔧 优化技术

| 技术 | 说明 | 配置 |
|------|------|------|
| Label Smoothing | 软标签，防止过拟合 | `label_smoothing=0.1` |
| Warmup + Cosine LR | 学习率预热 + 余弦退火 | `warmup_steps=300` |
| 梯度裁剪 | 防止梯度爆炸 | `gradient_clip=5.0` |
| EMA | 指数移动平均 | `ema_decay=0.999` |
| R-Drop | 一致性正则化 | `use_r_drop=True` |
| 早停 | 防止过拟合 | `patience=7` |

## 📊 训练流程

```
┌─────────────────────────────────────────────────────────┐
│                    三阶段训练                            │
├─────────────────────────────────────────────────────────┤
│  阶段一: 基础训练 (XE)                                   │
│    └─ 交叉熵损失，快速收敛                               │
│                    ↓                                     │
│  阶段二: 优化版训练 (XE + Optimizations)                 │
│    └─ Label Smoothing + Warmup + EMA + ...              │
│                    ↓                                     │
│  阶段三: SCST 微调 (RL)                                  │
│    └─ 直接优化 CIDEr，进一步提升                         │
└─────────────────────────────────────────────────────────┘
```

## 📈 SCST 原理

Self-Critical Sequence Training 使用强化学习直接优化评测指标：

$$L_{RL} = -\mathbb{E}_{w^s \sim p_\theta}\left[(r(w^s) - r(\hat{w}))\log p_\theta(w^s)\right]$$

- $w^s$: 采样生成的序列
- $\hat{w}$: Greedy 解码的序列（baseline）
- $r(\cdot)$: CIDEr 奖励

## ⚙️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 训练)

## 📦 依赖安装

```bash
# 基础依赖
pip install torch torchvision
pip install Pillow numpy tqdm matplotlib tensorboard

# 评测指标
pip install pycocoevalcap
```

## 📝 License

MIT License
