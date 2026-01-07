"""
模型评估脚本

在测试集上评估训练好的模型，输出 CIDEr, METEOR, ROUGE-L 等指标
支持评估 ViT-Transformer 和 Grid-Transformer 模型
"""

import os
import sys
import argparse
import json
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.eval_metrics import COCOScoreEvaluator
from utils.deepfashion_dataset import DeepFashionDataset
from utils.optimizations import CaptionAugmentation
from torch.utils.data import DataLoader


def load_model(checkpoint_path, model_type, device):
    """加载模型"""
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # 加载词典
    vocab_path = config.get('vocab_path', os.path.join(PROJECT_ROOT, 'data', 'vocab.json'))
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    print(f"词典大小: {vocab_size}")
    
    # 根据模型类型加载
    if model_type == 'vit':
        from models.vit_transformer_model import build_model
    else:  # grid
        from models.grid_transformer_model import build_model
    
    model = build_model(vocab_size, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功！(来自 epoch {checkpoint.get('epoch', 'unknown')})")
    if 'cider' in checkpoint:
        print(f"训练时最佳 CIDEr: {checkpoint['cider']:.4f}")
    
    return model, vocab, config


def create_test_loader(config, batch_size=32, num_workers=4):
    """创建测试集数据加载器"""
    data_dir = config.get('data_dir', os.path.join(PROJECT_ROOT, 'data'))
    vocab_path = config.get('vocab_path', os.path.join(PROJECT_ROOT, 'data', 'vocab.json'))
    
    transform = CaptionAugmentation.get_val_transforms(image_size=224)
    
    test_dataset = DeepFashionDataset(
        dataset_path=os.path.join(data_dir, 'test_data.json'),
        vocab_path=vocab_path,
        split='test',
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    return test_loader


def evaluate(model, test_loader, vocab, device, num_samples=None):
    """在测试集上评估模型"""
    model.eval()
    
    gts = {}  # ground truth
    res = {}  # results
    idx2word = {idx: word for word, idx in vocab.items()}
    evaluator = COCOScoreEvaluator()
    
    print("\n生成描述中...")
    
    with torch.no_grad():
        for i, (imgs, caps, cap_lens) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if num_samples and i * test_loader.batch_size >= num_samples:
                break
                
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            
            # 生成描述
            generated = model.generate(
                imgs,
                start_token=vocab['<start>'],
                end_token=vocab['<end>'],
                max_len=50,
                method='greedy'
            )
            
            start_idx = i * test_loader.batch_size
            
            for j in range(batch_size):
                img_id = start_idx + j
                
                # 参考描述
                ref_ids = caps[j].cpu().numpy()
                ref_words = [
                    idx2word[idx]
                    for idx in ref_ids
                    if idx not in [vocab["<start>"], vocab["<end>"], vocab["<pad>"]]
                ]
                gts[img_id] = [" ".join(ref_words)]
                
                # 生成的描述
                pred_ids = generated[j].cpu().numpy()
                pred_words = []
                for idx in pred_ids:
                    if idx == vocab["<end>"]:
                        break
                    if idx not in [vocab["<start>"], vocab["<pad>"]]:
                        pred_words.append(idx2word[idx])
                res[img_id] = [" ".join(pred_words)]
                
                # 打印前5个样本
                if img_id < 5:
                    print(f"\n[样本 {img_id}]")
                    print(f"  参考: {gts[img_id][0]}")
                    print(f"  生成: {res[img_id][0]}")
    
    print("\n计算评测分数...")
    scores = evaluator.evaluate(gts, res)
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='评估图像描述模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--model_type', type=str, choices=['vit', 'grid'], required=True,
                        help='模型类型: vit 或 grid')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='评估样本数量 (默认全部)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, vocab, config = load_model(args.checkpoint, args.model_type, device)
    
    # 创建数据加载器
    test_loader = create_test_loader(config, args.batch_size, args.num_workers)
    
    # 评估
    scores = evaluate(model, test_loader, vocab, device, args.num_samples)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    print(f"CIDEr:   {scores.get('CIDEr', 0.0):.4f}")
    print(f"METEOR:  {scores.get('METEOR', 0.0):.4f}")
    print(f"ROUGE-L: {scores.get('ROUGE_L', 0.0):.4f}")
    print("=" * 60)
    
    return scores


if __name__ == '__main__':
    main()
