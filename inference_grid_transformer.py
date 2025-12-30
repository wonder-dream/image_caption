"""
Grid-Transformer 模型推理和测试脚本
"""

import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from eval_metrics import COCOScoreEvaluator
from grid_transformer_model import build_model


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {}) # 兼容旧的checkpoint
    
    with open(config.get('vocab_path', 'data/vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    model = build_model(len(vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功！")
    print(f"训练轮次: {checkpoint.get('epoch', 'N/A')}")
    if 'cider' in checkpoint:
        print(f"CIDEr: {checkpoint['cider']:.4f}")
    
    return model, vocab, config


def preprocess_image(image_path, image_size=224):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def generate_caption(model, image_path, vocab, device='cuda', method='greedy', max_len=50, beam_size=5):
    """为单张图像生成描述"""
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        caption_ids = model.generate(
            image_tensor,
            start_token=vocab['<start>'],
            end_token=vocab['<end>'],
            max_len=max_len,
            method=method,
            beam_size=beam_size
        )
    
    caption_ids = caption_ids[0].cpu().tolist()
    idx2word = {idx: word for word, idx in vocab.items()}
    
    caption_words = []
    for idx in caption_ids:
        if idx == vocab['<end>']:
            break
        if idx not in [vocab['<start>'], vocab['<pad>']]:
            caption_words.append(idx2word[idx])
    
    return ' '.join(caption_words), caption_ids


def visualize_prediction(image_path, caption, save_path=None):
    """可视化图像和生成的描述"""
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Caption:\n{caption}", fontsize=12, wrap=True, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def evaluate_full_test_set(model, vocab, data_dir='data', device='cuda'):
    """在整个测试集上进行完整评测"""
    print("\n" + "=" * 80)
    print("开始全量测试集评测 (Grid-Transformer)")
    print("=" * 80)
    
    test_data_path = os.path.join(data_dir, 'test_data.json')
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    ground_truth = {}
    predictions = {}
    idx2word = {idx: word for word, idx in vocab.items()}
    
    print("准备 Ground Truth 数据...")
    for i, (img_path, cap_ids) in enumerate(zip(test_data['IMAGES'], test_data['CAPTIONS'])):
        words = [idx2word[idx] for idx in cap_ids if idx not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
        ground_truth[i] = [' '.join(words)]

    print(f"正在生成预测结果 (共 {len(test_data['IMAGES'])} 张图片)...")
    for i, img_path in enumerate(tqdm(test_data['IMAGES'])):
        try:
            caption, _ = generate_caption(model, img_path, vocab, device, method='greedy')
            predictions[i] = [caption]
        except Exception as e:
            print(f"生成图片 {img_path} 时出错: {e}")
            predictions[i] = [""]

    evaluator = COCOScoreEvaluator()
    scores = evaluator.evaluate(ground_truth, predictions)
    
    print("\n" + "=" * 80)
    print("最终评测结果 (Grid-Transformer)")
    print("=" * 80)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    
    return scores


def main():
    """主函数"""
    checkpoint_path = 'checkpoints/grid_transformer/best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("Grid-Transformer 图像描述模型推理")
    print("=" * 80)
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        print("请先运行 train_grid_transformer.py 进行训练")
        return

    print(f"\n使用设备: {device}")
    print(f"加载模型: {checkpoint_path}")
    model, vocab, config = load_model(checkpoint_path, device)
    
    # 测试单张图像
    with open('data/val_data.json', 'r') as f:
        val_data = json.load(f)
    test_image = val_data['IMAGES'][10] # 换一张图
    
    print("\n" + "=" * 80)
    print("测试1: 单张图像生成 (Greedy vs Beam Search)")
    print(f"测试图像: {test_image}")
    
    caption_greedy, _ = generate_caption(model, test_image, vocab, device, method='greedy')
    print(f"  Greedy: {caption_greedy}")
    
    caption_beam, _ = generate_caption(model, test_image, vocab, device, method='beam_search', beam_size=5)
    print(f"  Beam (k=5): {caption_beam}")
    
    visualize_prediction(test_image, caption_beam, save_path='output_grid_sample.png')
    
    # 全量评测
    print("\n是否进行全量评测? 这可能需要几分钟。")
    choice = input("输入 'y' 开始评测，其他键跳过: ")
    if choice.lower() == 'y':
        evaluate_full_test_set(model, vocab, data_dir='data', device=device)

    print("\n推理测试完成！")


if __name__ == '__main__':
    main()
