#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP 模型微调脚本

在 DeepFashion 数据集上微调 BLIP 预训练模型
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DeepFashionBLIPDataset(Dataset):
    """DeepFashion 数据集 - BLIP 格式"""
    
    def __init__(self, data_path, processor, split='train', max_length=50):
        self.processor = processor
        self.split = split
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.images = self.data['IMAGES']
        self.captions = self.data['CAPTIONS']
        
        # 加载词典用于解码
        vocab_path = os.path.join(os.path.dirname(data_path), 'vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.idx2word = {v: k for k, v in self.vocab.items()}
        
        print(f"加载 {split} 数据集: {len(self.images)} 样本")
    
    def __len__(self):
        return len(self.images)
    
    def _decode_caption(self, caption_ids):
        """将 token ids 解码为文本"""
        words = []
        for idx in caption_ids:
            word = self.idx2word.get(idx, '')
            if word in ['<start>', '<end>', '<pad>']:
                continue
            words.append(word)
        return ' '.join(words)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.images[idx]
        
        # 处理路径
        if not os.path.isabs(img_path):
            if img_path.startswith('data/') or img_path.startswith('data\\'):
                img_path = img_path[5:]
            img_path = os.path.join(PROJECT_ROOT, 'data', img_path)
        
        image = Image.open(img_path).convert('RGB')
        
        # 解码 caption
        caption_ids = self.captions[idx]
        caption_text = self._decode_caption(caption_ids)
        
        # 使用 BLIP processor 处理
        encoding = self.processor(
            images=image,
            text=caption_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除 batch 维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 设置 labels (用于计算损失)
        encoding['labels'] = encoding['input_ids'].clone()
        
        return encoding


def collate_fn(batch):
    """自定义 collate 函数"""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, gradient_accumulation_steps=1):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [训练]")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        # 移动数据到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, processor, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # 用于计算指标的数据
    all_predictions = []
    all_references = []
    
    pbar = tqdm(dataloader, desc="评估中")
    
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 计算损失
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        num_batches += 1
        
        # 生成描述 (只取前几个样本用于展示)
        if len(all_predictions) < 100:
            pixel_values = batch['pixel_values']
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=50,
                num_beams=3
            )
            
            # 解码
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(generated_texts)
            
            # 参考文本
            labels = batch['labels']
            labels[labels == -100] = processor.tokenizer.pad_token_id
            reference_texts = processor.batch_decode(labels, skip_special_tokens=True)
            all_references.extend(reference_texts)
        
        pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
    
    avg_loss = total_loss / num_batches
    
    # 打印一些样例
    print("\n生成样例:")
    for i in range(min(5, len(all_predictions))):
        print(f"  预测: {all_predictions[i]}")
        print(f"  参考: {all_references[i]}")
        print()
    
    return avg_loss, all_predictions, all_references


def compute_metrics(predictions, references):
    """计算评测指标"""
    try:
        from utils.eval_metrics import COCOScoreEvaluator
        
        # 构建评测格式
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        
        evaluator = COCOScoreEvaluator()
        scores = evaluator.evaluate(gts, res)
        
        return scores
    except Exception as e:
        print(f"评测指标计算失败: {e}")
        return {'CIDEr': 0, 'METEOR': 0, 'ROUGE_L': 0}


def main():
    parser = argparse.ArgumentParser(description='BLIP 模型微调')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, 
                        default='Salesforce/blip-image-captioning-base',
                        help='预训练模型路径或 HuggingFace 模型名')
    parser.add_argument('--model_type', type=str, default='blip',
                        choices=['blip', 'blip2'],
                        help='模型类型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录 (默认: PROJECT_ROOT/data)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='学习率预热比例')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--fp16', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='每隔几个 epoch 评估一次')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.data_dir is None:
        args.data_dir = os.path.join(PROJECT_ROOT, 'data')
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'blip_finetuned')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载模型和处理器
    print(f"\n加载模型: {args.model_path}")
    
    if args.model_type == 'blip2':
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained(args.model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if args.fp16 else torch.float32
        )
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(args.model_path)
        model = BlipForConditionalGeneration.from_pretrained(args.model_path)
    
    model = model.to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    
    # 创建数据集
    print("\n加载数据集...")
    
    train_dataset = DeepFashionBLIPDataset(
        data_path=os.path.join(args.data_dir, 'train_data.json'),
        processor=processor,
        split='train'
    )
    
    val_dataset = DeepFashionBLIPDataset(
        data_path=os.path.join(args.data_dir, 'val_data.json'),
        processor=processor,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("使用混合精度训练 (FP16)")
    
    # 训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_cider = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            args.gradient_accumulation_steps
        )
        
        print(f"训练损失: {train_loss:.4f}")
        
        # 评估
        if epoch % args.eval_every == 0:
            val_loss, predictions, references = evaluate(
                model, val_loader, processor, device
            )
            
            print(f"验证损失: {val_loss:.4f}")
            
            # 计算评测指标
            if len(predictions) > 0:
                scores = compute_metrics(predictions, references)
                cider = scores.get('CIDEr', 0)
                meteor = scores.get('METEOR', 0)
                rouge_l = scores.get('ROUGE_L', 0)
                
                print(f"CIDEr: {cider:.4f} | METEOR: {meteor:.4f} | ROUGE-L: {rouge_l:.4f}")
                
                # 保存最佳模型
                if cider > best_cider:
                    best_cider = cider
                    save_path = os.path.join(args.output_dir, 'best_model')
                    model.save_pretrained(save_path)
                    processor.save_pretrained(save_path)
                    print(f"✓ 保存最佳模型! CIDEr: {best_cider:.4f}")
        
        # 定期保存检查点
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}')
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            print(f"✓ 保存检查点: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳 CIDEr: {best_cider:.4f}")
    print(f"模型保存位置: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
