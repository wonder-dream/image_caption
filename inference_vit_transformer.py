"""
ViT + Transformer 模型推理和测试脚本
"""

import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from vit_transformer_model import build_model


def load_model(checkpoint_path, device='cuda'):
    """
    加载训练好的模型
    
    参数:
        checkpoint_path: 检查点文件路径
        device: 'cuda' 或 'cpu'
    返回:
        model: 加载好的模型
        vocab: 词典
        config: 配置
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 加载词典
    with open(config['vocab_path'], 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 构建模型
    model = build_model(len(vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功！")
    print(f"训练轮次: {checkpoint['epoch']}")
    if 'bleu4' in checkpoint:
        print(f"BLEU-4: {checkpoint['bleu4']:.4f}")
    
    return model, vocab, config


def preprocess_image(image_path, image_size=224):
    """
    预处理图像
    
    参数:
        image_path: 图像路径
        image_size: 目标图像大小
    返回:
        image_tensor: 预处理后的图像张量 (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor


def generate_caption(model, image_path, vocab, device='cuda', method='greedy', max_len=50):
    """
    为单张图像生成描述
    
    参数:
        model: 模型
        image_path: 图像路径
        vocab: 词典
        device: 设备
        method: 'greedy' 或 'beam_search'
        max_len: 最大生成长度
    返回:
        caption: 生成的描述文本
        caption_ids: 生成的ID序列
    """
    # 预处理图像
    image_tensor = preprocess_image(image_path).to(device)
    
    # 生成caption
    with torch.no_grad():
        caption_ids = model.generate(
            image_tensor,
            start_token=vocab['<start>'],
            end_token=vocab['<end>'],
            max_len=max_len,
            method=method
        )
    
    # 转换为文本
    caption_ids = caption_ids[0].cpu().tolist()
    idx2word = {idx: word for word, idx in vocab.items()}
    
    caption_words = []
    for idx in caption_ids:
        if idx == vocab['<end>']:
            break
        if idx not in [vocab['<start>'], vocab['<pad>']]:
            caption_words.append(idx2word[idx])
    
    caption = ' '.join(caption_words)
    
    return caption, caption_ids


def visualize_prediction(image_path, caption, save_path=None):
    """
    可视化图像和生成的描述
    
    参数:
        image_path: 图像路径
        caption: 生成的描述
        save_path: 保存路径（可选）
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Caption:\n{caption}", fontsize=12, wrap=True, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def batch_generate_captions(model, image_paths, vocab, device='cuda', method='greedy'):
    """
    批量生成图像描述
    
    参数:
        model: 模型
        image_paths: 图像路径列表
        vocab: 词典
        device: 设备
        method: 生成方法
    返回:
        captions: 生成的描述列表
    """
    captions = []
    
    print(f"批量生成 {len(image_paths)} 张图像的描述...")
    
    for image_path in image_paths:
        caption, _ = generate_caption(model, image_path, vocab, device, method)
        captions.append(caption)
        print(f"  {os.path.basename(image_path)}: {caption}")
    
    return captions


def evaluate_samples(model, vocab, data_dir='data', num_samples=5, device='cuda'):
    """
    在验证集上评估一些样本
    
    参数:
        model: 模型
        vocab: 词典
        data_dir: 数据目录
        num_samples: 要评估的样本数
        device: 设备
    """
    # 加载验证集数据
    val_data_path = os.path.join(data_dir, 'val_data.json')
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # 随机选择样本
    import random
    indices = random.sample(range(len(val_data['IMAGES'])), num_samples)
    
    idx2word = {idx: word for word, idx in vocab.items()}
    
    print(f"\n在验证集上评估 {num_samples} 个样本")
    print("=" * 80)
    
    for i, idx in enumerate(indices, 1):
        image_path = val_data['IMAGES'][idx]
        ref_caption_ids = val_data['CAPTIONS'][idx]
        
        # 参考caption
        ref_words = [idx2word[idx] for idx in ref_caption_ids 
                    if idx not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
        ref_caption = ' '.join(ref_words)
        
        # 生成caption
        gen_caption, _ = generate_caption(model, image_path, vocab, device, method='greedy')
        
        print(f"\n样本 {i}:")
        print(f"  图像: {os.path.basename(image_path)}")
        print(f"  参考: {ref_caption}")
        print(f"  生成: {gen_caption}")
        print("-" * 80)


def compare_generation_methods(model, image_path, vocab, device='cuda'):
    """
    比较不同生成方法的结果
    
    参数:
        model: 模型
        image_path: 图像路径
        vocab: 词典
        device: 设备
    """
    print(f"\n比较不同生成方法")
    print(f"图像: {image_path}")
    print("=" * 80)
    
    # Greedy搜索
    print("\n1. Greedy搜索:")
    caption_greedy, _ = generate_caption(model, image_path, vocab, device, method='greedy')
    print(f"   {caption_greedy}")
    
    # Beam搜索
    print("\n2. Beam搜索:")
    caption_beam, _ = generate_caption(model, image_path, vocab, device, method='beam_search')
    print(f"   {caption_beam}")
    
    print("=" * 80)


def main():
    """主函数"""
    
    # 配置
    checkpoint_path = 'checkpoints/vit_transformer/best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("ViT + Transformer 图像描述模型推理")
    print("=" * 80)
    
    # 加载模型
    print(f"\n使用设备: {device}")
    print(f"加载模型: {checkpoint_path}")
    model, vocab, config = load_model(checkpoint_path, device)
    
    # 测试单张图像
    print("\n" + "=" * 80)
    print("测试1: 单张图像生成")
    print("=" * 80)
    
    # 从验证集选一张图像
    with open('data/val_data.json', 'r') as f:
        val_data = json.load(f)
    
    test_image = val_data['IMAGES'][0]
    print(f"测试图像: {test_image}")
    
    caption, caption_ids = generate_caption(
        model, test_image, vocab, device, method='greedy'
    )
    print(f"生成的描述: {caption}")
    print(f"ID序列: {caption_ids}")
    
    # 可视化
    visualize_prediction(test_image, caption, save_path='output_sample.png')
    
    # 比较生成方法
    print("\n" + "=" * 80)
    print("测试2: 比较生成方法")
    print("=" * 80)
    compare_generation_methods(model, test_image, vocab, device)
    
    # 评估多个样本
    print("\n" + "=" * 80)
    print("测试3: 验证集样本评估")
    print("=" * 80)
    evaluate_samples(model, vocab, 'data', num_samples=5, device=device)
    
    print("\n" + "=" * 80)
    print("推理测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
