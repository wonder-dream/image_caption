# Image Caption - 图像描述生成项目

基于 DeepFashion-MultiModal 数据集的图像描述生成模型。

## 📁 项目结构

```
image_caption/
├── models/                          # 模型定义
│   ├── __init__.py
│   ├── vit_transformer_model.py     # ViT + Transformer 模型
│   └── grid_transformer_model.py    # Grid + Transformer 模型
│
├── scripts/                         # 训练和推理脚本
│   ├── __init__.py
│   ├── train_vit_transformer.py     # ViT 模型训练脚本
│   ├── train_grid_transformer.py    # Grid 模型训练脚本
│   ├── inference_vit_transformer.py # ViT 模型推理脚本
│   ├── inference_grid_transformer.py# Grid 模型推理脚本
│   ├── inference.py                 # 通用推理脚本
│   └── test_model.py                # 模型测试脚本
│
├── utils/                           # 工具模块
│   ├── __init__.py
│   ├── deepfashion_dataset.py       # 数据集类和数据加载器
│   ├── prepare_data.py              # 数据预处理脚本
│   ├── unzip_dataset.py             # 数据集解压脚本
│   └── eval_metrics.py              # 评测指标 (CIDEr, METEOR, BLEU)
│
├── data/                            # 数据目录
│   ├── images/                      # 图像文件夹
│   ├── images.zip                   # 原始图像压缩包
│   ├── captions.json                # 原始标注文件
│   ├── vocab.json                   # 词典文件
│   ├── train_data.json              # 训练集
│   ├── val_data.json                # 验证集
│   └── test_data.json               # 测试集
│
├── checkpoints/                     # 模型检查点
│   ├── vit_transformer/
│   │   └── best_model.pth
│   └── grid_transformer/
│       └── best_model.pth
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

```bash
# 训练 ViT + Transformer 模型
python scripts/train_vit_transformer.py

# 训练 Grid + Transformer 模型
python scripts/train_grid_transformer.py
```

### 3. 推理测试

```bash
# ViT 模型推理
python scripts/inference_vit_transformer.py

# Grid 模型推理
python scripts/inference_grid_transformer.py

# 通用推理（指定图片）
python scripts/inference.py --image test.jpg
```

## 📊 模型性能

| 模型 | CIDEr | BLEU-4 | METEOR |
|------|-------|--------|--------|
| ViT + Transformer | 1.5+ | - | - |
| Grid + Transformer | - | - | - |

## 🔧 依赖

- Python 3.8+
- PyTorch 2.0+
- torchvision
- pycocotools
- pycocoevalcap
- tqdm
- matplotlib
- Pillow
