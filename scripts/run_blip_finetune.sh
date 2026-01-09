#!/bin/bash
# ==============================================================
# BLIP 模型微调完整流程脚本 (Linux 服务器)
# ==============================================================

set -e

echo "=============================================="
echo "BLIP Image Captioning 微调流程"
echo "=============================================="

# 配置变量 (根据需要修改)
MODEL_TYPE="blip-base"          # 可选: blip-base, blip-large, blip2
USE_MIRROR=true                  # 国内服务器设为 true
BATCH_SIZE=8                     # 根据显存调整
EPOCHS=10
LEARNING_RATE=5e-5
NUM_WORKERS=4
USE_FP16=true                    # 混合精度训练

# 路径配置
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA_DIR="${PROJECT_ROOT}/data"
MODEL_DIR="${PROJECT_ROOT}/pretrained_models"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/blip_finetuned"

echo "项目目录: ${PROJECT_ROOT}"
echo "数据目录: ${DATA_DIR}"
echo "模型目录: ${MODEL_DIR}"
echo "输出目录: ${OUTPUT_DIR}"

# ==============================================================
# Step 1: 检查环境
# ==============================================================
echo ""
echo "=============================================="
echo "Step 1: 检查环境"
echo "=============================================="

# 检查 Python
python3 --version || { echo "错误: 未找到 Python3"; exit 1; }

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "警告: 未检测到 NVIDIA GPU"
fi

# 检查必要的包
echo ""
echo "检查 Python 包..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import PIL; print('Pillow: OK')"

# ==============================================================
# Step 2: 下载预训练模型
# ==============================================================
echo ""
echo "=============================================="
echo "Step 2: 下载预训练模型"
echo "=============================================="

DOWNLOAD_ARGS="--model ${MODEL_TYPE} --save_dir ${MODEL_DIR} --verify"
if [ "$USE_MIRROR" = true ]; then
    DOWNLOAD_ARGS="${DOWNLOAD_ARGS} --mirror"
fi

echo "下载参数: ${DOWNLOAD_ARGS}"

# 检查模型是否已存在
MODEL_NAME_MAP=(
    "blip-base:Salesforce_blip-image-captioning-base"
    "blip-large:Salesforce_blip-image-captioning-large"
    "blip2:Salesforce_blip2-opt-2.7b"
)

for mapping in "${MODEL_NAME_MAP[@]}"; do
    key="${mapping%%:*}"
    value="${mapping#*:}"
    if [ "$key" = "$MODEL_TYPE" ]; then
        MODEL_PATH="${MODEL_DIR}/${value}"
        break
    fi
done

if [ -d "$MODEL_PATH" ] && [ -f "${MODEL_PATH}/config.json" ]; then
    echo "模型已存在: ${MODEL_PATH}"
    echo "跳过下载"
else
    echo "开始下载模型..."
    python3 "${PROJECT_ROOT}/scripts/download_blip.py" ${DOWNLOAD_ARGS}
fi

# ==============================================================
# Step 3: 准备数据
# ==============================================================
echo ""
echo "=============================================="
echo "Step 3: 检查数据"
echo "=============================================="

# 检查数据文件
for file in "train_data.json" "val_data.json" "test_data.json" "vocab.json"; do
    if [ -f "${DATA_DIR}/${file}" ]; then
        echo "✓ ${file}"
    else
        echo "✗ ${file} 不存在"
        echo "请先运行数据预处理: python utils/prepare_data.py"
        exit 1
    fi
done

# 检查图像目录
if [ -d "${DATA_DIR}/images" ]; then
    IMAGE_COUNT=$(find "${DATA_DIR}/images" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo "✓ images/ (${IMAGE_COUNT} 张图片)"
else
    echo "✗ images/ 目录不存在"
    exit 1
fi

# ==============================================================
# Step 4: 微调模型
# ==============================================================
echo ""
echo "=============================================="
echo "Step 4: 开始微调"
echo "=============================================="

FINETUNE_ARGS="--model_path ${MODEL_PATH}"
FINETUNE_ARGS="${FINETUNE_ARGS} --data_dir ${DATA_DIR}"
FINETUNE_ARGS="${FINETUNE_ARGS} --output_dir ${OUTPUT_DIR}"
FINETUNE_ARGS="${FINETUNE_ARGS} --epochs ${EPOCHS}"
FINETUNE_ARGS="${FINETUNE_ARGS} --batch_size ${BATCH_SIZE}"
FINETUNE_ARGS="${FINETUNE_ARGS} --learning_rate ${LEARNING_RATE}"
FINETUNE_ARGS="${FINETUNE_ARGS} --num_workers ${NUM_WORKERS}"

if [ "$USE_FP16" = true ]; then
    FINETUNE_ARGS="${FINETUNE_ARGS} --fp16"
fi

echo "微调参数:"
echo "  模型路径: ${MODEL_PATH}"
echo "  数据目录: ${DATA_DIR}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  批量大小: ${BATCH_SIZE}"
echo "  学习率: ${LEARNING_RATE}"
echo "  训练轮数: ${EPOCHS}"
echo "  混合精度: ${USE_FP16}"
echo ""

python3 "${PROJECT_ROOT}/scripts/finetune_blip.py" ${FINETUNE_ARGS}

# ==============================================================
# Step 5: 测试模型
# ==============================================================
echo ""
echo "=============================================="
echo "Step 5: 测试模型性能"
echo "=============================================="

BEST_MODEL="${OUTPUT_DIR}/best_model"
RESULT_FILE="${OUTPUT_DIR}/test_results.json"

if [ -d "$BEST_MODEL" ]; then
    echo "测试最佳模型: ${BEST_MODEL}"
    
    python3 "${PROJECT_ROOT}/scripts/test_blip.py" \
        --model_path "${BEST_MODEL}" \
        --data_dir "${DATA_DIR}" \
        --batch_size 16 \
        --num_beams 5 \
        --output_file "${RESULT_FILE}"
else
    echo "警告: 未找到最佳模型，跳过测试"
fi

# ==============================================================
# 完成
# ==============================================================
echo ""
echo "=============================================="
echo "✅ 全部完成!"
echo "=============================================="
echo "模型保存位置: ${OUTPUT_DIR}"
echo "测试结果: ${RESULT_FILE}"
echo ""
echo "使用以下命令进行单张图片推理:"
echo "  python scripts/inference_blip.py --model_path ${BEST_MODEL} --image <图片路径>"
