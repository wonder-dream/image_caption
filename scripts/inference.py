import torch
import json
import os
import argparse
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_transformer_model import build_model


def load_model(checkpoint_path, vocab_path, device):
    """
    加载模型和词典
    """
    print(f"正在加载模型: {checkpoint_path}")

    # 1. 加载 Checkpoint
    # 注意：weights_only=False 是为了兼容包含 numpy 数据的 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # 2. 加载词典
    print(f"正在加载词典: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # 3. 构建模型结构
    vocab_size = len(vocab)
    model = build_model(vocab_size, config)

    # 4. 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, vocab, config


def preprocess_image(image_path):
    """
    读取并预处理图片
    """
    # ViT 标准预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image


def generate_caption(model, image, vocab, device, max_len=50):
    """
    生成描述 (Greedy Search)
    """
    # 反向词典: index -> word
    idx2word = {v: k for k, v in vocab.items()}

    # 准备输入
    image = image.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        # 1. 编码图片
        features = model.encoder(image)  # (1, 196, 512)

        # 2. 解码生成
        inputs = torch.tensor([vocab["<start>"]]).unsqueeze(0).to(device)  # (1, 1)
        result_caption = []

        for _ in range(max_len):
            outputs = model.decoder(inputs, features)  # (1, seq_len, vocab_size)

            # 获取最后一个时间步的预测
            last_token_logits = outputs[:, -1, :]
            _, predicted_id = last_token_logits.max(1)

            predicted_id = predicted_id.item()

            # 如果遇到结束符，停止
            if predicted_id == vocab["<end>"]:
                break

            # 记录单词
            word = idx2word.get(predicted_id, "<unk>")
            if word != "<start>" and word != "<pad>":
                result_caption.append(word)

            # 将预测词作为下一次的输入
            inputs = torch.cat(
                [inputs, torch.tensor([[predicted_id]]).to(device)], dim=1
            )

    return " ".join(result_caption)


def main():
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/vit_transformer/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--vocab", type=str, default="data/vocab.json", help="Path to vocab json"
    )
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图片文件不存在 {args.image}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        # 加载模型
        model, vocab, _ = load_model(args.model, args.vocab, device)

        # 处理图片
        image = preprocess_image(args.image)

        # 生成描述
        print("-" * 50)
        print(f"正在分析图片: {args.image} ...")
        caption = generate_caption(model, image, vocab, device)

        print("\n生成的描述:")
        print(f"Step 1: {caption}")
        print("-" * 50)

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
