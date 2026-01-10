#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP 模型测试脚本

在测试集上评估微调后的 BLIP 模型性能
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DeepFashionTestDataset(Dataset):
    """测试数据集"""
    
    def __init__(self, data_path, processor):
        self.processor = processor
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.images = self.data['IMAGES']
        self.captions = self.data['CAPTIONS']
        
        # 加载词典
        vocab_path = os.path.join(os.path.dirname(data_path), 'vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.idx2word = {v: k for k, v in self.vocab.items()}
        
        print(f"测试集: {len(self.images)} 样本")
    
    def __len__(self):
        return len(self.images)
    
    def _decode_caption(self, caption_ids):
        words = []
        for idx in caption_ids:
            word = self.idx2word.get(idx, '')
            if word in ['<start>', '<end>', '<pad>']:
                continue
            words.append(word)
        return ' '.join(words)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        if not os.path.isabs(img_path):
            if img_path.startswith('data/') or img_path.startswith('data\\'):
                img_path = img_path[5:]
            img_path = os.path.join(PROJECT_ROOT, 'data', img_path)
        
        image = Image.open(img_path).convert('RGB')
        
        # 参考描述
        caption_ids = self.captions[idx]
        reference = self._decode_caption(caption_ids)
        
        # 处理图像
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']
        
        return {
            'pixel_values': pixel_values.squeeze(0),
            'reference': reference,
            'image_path': img_path
        }


def test_model(args):
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    
    if args.model_type == 'blip2':
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained(args.model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(args.model_path)
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(args.model_path)
        model = BlipForConditionalGeneration.from_pretrained(args.model_path)
    
    model = model.to(device)
    model.eval()
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 创建测试数据集
    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, 'data')
    test_dataset = DeepFashionTestDataset(
        data_path=os.path.join(data_dir, 'test_data.json'),
        processor=processor
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 生成描述
    print("\n生成描述中...")
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试"):
            pixel_values = batch['pixel_values'].to(device)
            references = batch['reference']
            
            # 生成
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=args.max_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
            
            # 解码
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_predictions.extend(generated_texts)
            all_references.extend(references)
    
    # 打印一些样例
    print("\n" + "="*60)
    print("生成样例:")
    print("="*60)
    for i in range(min(10, len(all_predictions))):
        print(f"\n[样本 {i+1}]")
        print(f"  预测: {all_predictions[i]}")
        print(f"  参考: {all_references[i]}")
    
    # 计算评测指标
    print("\n" + "="*60)
    print("计算评测指标...")
    print("="*60)
    
    from utils.eval_metrics import COCOScoreEvaluator
    
    gts = {i: [ref] for i, ref in enumerate(all_references)}
    res = {i: [pred] for i, pred in enumerate(all_predictions)}
    
    evaluator = COCOScoreEvaluator()
    scores = evaluator.evaluate(gts, res)
    
    print("\n" + "="*60)
    print("测试结果:")
    print("="*60)
    print(f"CIDEr:   {scores.get('CIDEr', 0):.4f}")
    print(f"METEOR:  {scores.get('METEOR', 0):.4f}")
    print(f"ROUGE-L: {scores.get('ROUGE_L', 0):.4f}")
    print("="*60)
    
    # 保存结果
    if args.output_file:
        results = {
            'scores': scores,
            'predictions': [
                {'id': i, 'prediction': pred, 'reference': ref}
                for i, (pred, ref) in enumerate(zip(all_predictions, all_references))
            ]
        }
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_file}")
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='BLIP 模型测试')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='微调后的模型路径')
    parser.add_argument('--model_type', type=str, default='blip',
                        choices=['blip', 'blip2'],
                        help='模型类型')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--max_length', type=int, default=50,
                        help='最大生成长度')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Beam search 宽度')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出结果文件')
    
    args = parser.parse_args()
    
    test_model(args)


if __name__ == '__main__':
    main()
