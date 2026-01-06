# Image Caption - 图像描述生成项目

基于 DeepFashion-MultiModal 数据集的图像描述生成模型。

## 📁 项目结构

```text
image_caption/
├── models/                          # 模型定义
│   ├── __init__.py
│   ├── vit_transformer_model.py     # ViT + Transformer 模型
│   └── grid_transformer_model.py    # Grid + Transformer 模型
│
├── scripts/                         # 训练和推理脚本
│   ├── __init__.py
│   ├── train_vit_transformer.py              # ViT 模型基础训练 (交叉熵)
│   ├── train_grid_transformer.py             # Grid 模型基础训练 (交叉熵)
│   ├── train_vit_transformer_optimized.py    # ViT 模型优化版训练
│   ├── train_grid_transformer_optimized.py   # Grid 模型优化版训练
│   ├── train_vit_transformer_scst_optimized.py   # ViT SCST+优化 训练
│   ├── train_grid_transformer_scst_optimized.py  # Grid SCST+优化 训练
│   ├── inference_vit_transformer.py          # ViT 模型推理脚本
│   ├── inference_grid_transformer.py         # Grid 模型推理脚本
│   ├── inference.py                          # 通用推理脚本
│   └── test_model.py                         # 模型测试脚本
│
├── utils/                           # 工具模块
│   ├── __init__.py
│   ├── deepfashion_dataset.py       # 数据集类和数据加载器
│   ├── prepare_data.py              # 数据预处理脚本
│   ├── unzip_dataset.py             # 数据集解压脚本
│   ├── eval_metrics.py              # 评测指标 (CIDEr, METEOR, BLEU)
│   ├── scst_loss.py                 # SCST 强化学习损失函数
│   └── optimizations.py             # 优化工具 (Label Smoothing, EMA, etc.)
│
├── data/                            # 数据目录
│   ├── images/                      # 图像文件夹
│   ├── vocab.json                   # 词典文件
│   ├── train_data.json              # 训练集
│   ├── val_data.json                # 验证集
│   └── test_data.json               # 测试集
│
├── checkpoints/                     # 模型检查点
│   ├── vit_transformer/             # ViT 基础模型
│   ├── vit_transformer_optimized/   # ViT 优化版模型
│   ├── vit_transformer_scst_opt/    # ViT SCST+优化模型
│   ├── grid_transformer/            # Grid 基础模型
│   ├── grid_transformer_optimized/  # Grid 优化版模型
│   └── grid_transformer_scst_opt/   # Grid SCST+优化模型
│
├── 模型完成总结.md                   # 项目总结文档
└── README.md                        # 本文件
```

## 🚀 快速开始

### 1. 数据准备

```bash
# 解压数据集
python utils/unzip_dataset.py

# 生成词典和数据集划分
python utils/prepare_data.py
```

### 2. 训练模型

#### 基础训练（交叉熵）
```bash
# 训练 ViT + Transformer 模型
python scripts/train_vit_transformer.py

# 训练 Grid + Transformer 模型
python scripts/train_grid_transformer.py
```

#### 优化版训练（推荐）
```bash
# ViT 模型优化版训练
python scripts/train_vit_transformer_optimized.py

# Grid 模型优化版训练
python scripts/train_grid_transformer_optimized.py
```

### 3. 强化学习微调（SCST + 优化）

SCST (Self-Critical Sequence Training) 直接优化评测指标，需要先完成优化版训练。

```bash
# ViT 模型 SCST+优化 训练
python scripts/train_vit_transformer_scst_optimized.py

# Grid 模型 SCST+优化 训练
python scripts/train_grid_transformer_scst_optimized.py
```

### 4. 推理测试

```bash
# ViT 模型推理
python scripts/inference_vit_transformer.py

# Grid 模型推理
python scripts/inference_grid_transformer.py
```

## 🧠 训练策略

### 三阶段训练流程

```
阶段一: 交叉熵预训练 (XE)
         ↓
阶段二: 优化版训练 (XE + 各种优化技术)
         ↓
阶段三: 强化学习微调 (SCST)
```

### 优化技术详解

| 优化技术 | 作用 | 配置参数 |
|---------|------|----------|
| **Label Smoothing** | 防止过拟合，提高泛化 | `label_smoothing=0.1` |
| **Warmup + Cosine LR** | 稳定训练，更好收敛 | `warmup_steps=2000` |
| **数据增强** | 增加数据多样性 | `use_data_augmentation=True` |
| **EMA** | 参数平滑，提升泛化 | `use_ema=True, ema_decay=0.999` |
| **梯度裁剪** | 防止梯度爆炸 | `gradient_clip=1.0` |
| **早停机制** | 防止过拟合 | `patience=7` |
| **R-Drop** | 一致性正则化 | `use_r_drop=True` |
| **Weight Decay** | L2 正则化 | `weight_decay=0.01` |

### SCST 原理

$$L_{RL} = -\mathbb{E}_{w^s \sim p_\theta}[(r(w^s) - r(\hat{w}))\log p_\theta(w^s)]$$

- $w^s$: 采样生成的序列
- $\hat{w}$: Greedy 解码的序列（作为 baseline）
- $r(\cdot)$: 奖励函数（如 CIDEr 分数）

## 📊 模型性能

| 模型 | 训练方式 | CIDEr | METEOR | ROUGE-L |
|------|----------|-------|--------|---------|
| ViT + Transformer | XE | ~1.2 | - | - |
| ViT + Transformer | XE + 优化 | ~1.3 | - | - |
| ViT + Transformer | XE + SCST | ~1.5+ | - | - |
| Grid + Transformer | XE | - | - | - |
| Grid + Transformer | XE + 优化 | - | - | - |
| Grid + Transformer | XE + SCST | - | - | - |

## ⚙️ 优化配置示例

```python
config = {
    # 优化策略
    "label_smoothing": 0.1,        # Label Smoothing
    "warmup_steps": 2000,          # Warmup 步数
    "gradient_clip": 1.0,          # 梯度裁剪
    "use_data_augmentation": True, # 数据增强
    "use_ema": True,               # EMA
    "ema_decay": 0.999,
    "weight_decay": 0.01,          # L2 正则化
    "patience": 7,                 # 早停 patience
}
```

## 🔧 依赖

- Python 3.8+
- PyTorch 2.0+
- torchvision
- pycocotools
- pycocoevalcap
- tqdm
- matplotlib
- Pillow
