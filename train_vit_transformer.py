"""
ViT + Transformer 图像描述模型训练脚本
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from vit_transformer_model import build_model
from deepfashion_dataset import create_data_loaders


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_loader, criterion, optimizer, epoch, device, vocab, writer, global_step):
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    
    # 进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, captions, cap_lens) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)
        cap_lens = cap_lens.to(device)
        
        # 前向传播
        outputs = model(images, captions, cap_lens)
        
        # 计算损失
        # outputs: (batch_size, seq_len-1, vocab_size)
        # targets: captions[:, 1:] (不包括<start>)
        targets = captions[:, 1:]  # (batch_size, seq_len-1)
        
        # 重塑为 (batch_size * seq_len, vocab_size) 和 (batch_size * seq_len,)
        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)
        
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # 记录到tensorboard
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), global_step[0])
            global_step[0] += 1
    
    return losses.avg


def validate(model, val_loader, criterion, device, vocab):
    """在验证集上评估"""
    model.eval()
    
    losses = AverageMeter()
    
    # 用于计算BLEU
    all_predictions = []
    all_references = []
    
    idx2word = {idx: word for word, idx in vocab.items()}
    
    with torch.no_grad():
        for images, captions, cap_lens in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            captions = captions.to(device)
            cap_lens = cap_lens.to(device)
            
            # 计算损失
            outputs = model(images, captions, cap_lens)
            targets = captions[:, 1:]
            
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            losses.update(loss.item(), images.size(0))
            
            # 生成caption用于计算BLEU
            generated = model.generate(
                images,
                start_token=vocab['<start>'],
                end_token=vocab['<end>'],
                max_len=52,
                method='greedy'
            )
            
            # 转换为词序列
            for i in range(images.size(0)):
                # 预测的caption
                pred_ids = generated[i].cpu().tolist()
                pred_words = [idx2word[idx] for idx in pred_ids 
                             if idx not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
                all_predictions.append(pred_words)
                
                # 参考caption
                ref_ids = captions[i, :cap_lens[i]].cpu().tolist()
                ref_words = [idx2word[idx] for idx in ref_ids 
                            if idx not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
                all_references.append([ref_words])  # BLEU需要list of lists
    
    # 计算BLEU-4
    bleu4 = corpus_bleu(all_references, all_predictions, weights=(0.25, 0.25, 0.25, 0.25))
    
    return losses.avg, bleu4


def train(config):
    """主训练函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词典
    with open(config['vocab_path'], 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    print(f"词典大小: {vocab_size}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['data_dir'],
        vocab_path=config['vocab_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=224
    )
    
    # 构建模型
    print("\n构建模型...")
    model = build_model(vocab_size, config)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 是否微调ViT编码器
    if config.get('finetune_encoder_after_epoch', -1) >= 0:
        print(f"将在第 {config['finetune_encoder_after_epoch']} 个epoch后开始微调编码器")
    
    # 损失函数（忽略padding的损失）
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控BLEU（越大越好）
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # TensorBoard
    writer = None
    if config.get('use_tensorboard', False):
        writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/vit_transformer'))
        print(f"TensorBoard日志目录: {writer.log_dir}")
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 70)
    
    best_bleu = 0.0
    global_step = [0]  # 用list包装以便在函数中修改
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)
        
        # 是否开始微调编码器
        if epoch == config.get('finetune_encoder_after_epoch', -1):
            print("开始微调ViT编码器...")
            model.encoder.set_trainable(True)
            # 为编码器使用更小的学习率
            optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'lr': config['learning_rate'] * 0.1},
                {'params': model.decoder.parameters(), 'lr': config['learning_rate']}
            ], betas=(0.9, 0.98), eps=1e-9)
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            epoch, device, vocab, writer, global_step
        )
        
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        if epoch % config.get('eval_every', 1) == 0:
            print("\n验证中...")
            val_loss, bleu4 = validate(model, val_loader, criterion, device, vocab)
            
            print(f"验证损失: {val_loss:.4f}")
            print(f"BLEU-4: {bleu4:.4f}")
            
            # 记录到tensorboard
            if writer is not None:
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/bleu4', bleu4, epoch)
                writer.add_scalar('train/epoch_loss', train_loss, epoch)
            
            # 学习率调度
            scheduler.step(bleu4)
            
            # 保存最佳模型
            if bleu4 > best_bleu:
                best_bleu = bleu4
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'bleu4': bleu4,
                    'config': config
                }
                save_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
                os.makedirs(config['checkpoint_dir'], exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"✓ 保存最佳模型 (BLEU-4: {bleu4:.4f}) 到 {save_path}")
        
        # 定期保存检查点
        if epoch % config.get('save_every', 5) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bleu4': bleu4 if epoch % config.get('eval_every', 1) == 0 else 0,
                'config': config
            }
            save_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"✓ 保存检查点到 {save_path}")
        
        print("-" * 70)
    
    print("\n训练完成！")
    print(f"最佳BLEU-4: {best_bleu:.4f}")
    
    # 关闭writer
    if writer is not None:
        writer.close()
    
    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(os.path.join(config['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_bleu4 = validate(model, test_loader, criterion, device, vocab)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集BLEU-4: {test_bleu4:.4f}")
    
    return model


if __name__ == '__main__':
    # 配置
    config = {
        # 数据
        'data_dir': 'data',
        'vocab_path': 'data/vocab.json',
        'batch_size': 32,
        'num_workers': 0,  # Windows下建议设为0
        
        # 模型
        'd_model': 512,
        'nhead': 8,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_len': 52,
        'pretrained_vit': True,
        
        # 训练
        'num_epochs': 30,
        'learning_rate': 0.0001,
        'finetune_encoder_after_epoch': 10,  # 第10个epoch后开始微调编码器，-1表示不微调
        
        # 评估和保存
        'eval_every': 1,  # 每个epoch验证一次
        'save_every': 5,  # 每5个epoch保存一次
        'checkpoint_dir': 'checkpoints/vit_transformer',
        
        # 日志
        'use_tensorboard': True,
        'log_dir': 'runs/vit_transformer'
    }
    
    print("=" * 70)
    print("ViT + Transformer 图像描述模型训练")
    print("=" * 70)
    print("\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 开始训练
    model = train(config)
