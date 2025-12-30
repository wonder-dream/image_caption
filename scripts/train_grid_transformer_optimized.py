"""
Grid-Transformer 图像描述模型 - 优化版训练脚本

集成多种优化策略:
1. Label Smoothing - 标签平滑
2. Warmup + Cosine 学习率调度
3. 数据增强
4. EMA (指数移动平均)
5. 梯度裁剪
6. 早停机制
7. R-Drop 正则化 (可选)
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eval_metrics import COCOScoreEvaluator
from models.grid_transformer_model import build_model
from utils.deepfashion_dataset import DeepFashionDataset
from utils.optimizations import (
    LabelSmoothingLoss,
    WarmupCosineScheduler,
    CaptionAugmentation,
    ExponentialMovingAverage,
    EarlyStopping,
    GradientClipping,
    R_Drop,
)
from torch.utils.data import DataLoader


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


class MetricsHistory:
    """记录训练过程中的所有指标"""

    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.cider_scores = []
        self.meteor_scores = []
        self.rouge_l_scores = []
        self.learning_rates = []

    def add(self, epoch, train_loss, val_loss, scores, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.cider_scores.append(scores.get("CIDEr", 0.0))
        self.meteor_scores.append(scores.get("METEOR", 0.0))
        self.rouge_l_scores.append(scores.get("ROUGE_L", 0.0))
        self.learning_rates.append(lr)

    def plot_and_save(self, save_path):
        """绘制并保存折线图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Grid-Transformer Optimized Training Metrics", fontsize=16)

        axes[0, 0].plot(self.epochs, self.train_losses, "b-", marker="o", label="Train Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.epochs, self.val_losses, "r-", marker="o", label="Val Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(self.epochs, self.cider_scores, "g-", marker="o", label="CIDEr")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].set_title("CIDEr Score")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        axes[1, 0].plot(self.epochs, self.meteor_scores, "m-", marker="o", label="METEOR")
        axes[1, 0].plot(self.epochs, self.rouge_l_scores, "c-", marker="s", label="ROUGE-L")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_title("METEOR & ROUGE-L")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.epochs, self.learning_rates, "k-", marker="o", label="LR")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        axes[1, 2].plot(self.epochs, self.cider_scores, "g-", marker="o", label="CIDEr")
        axes[1, 2].plot(self.epochs, self.meteor_scores, "m-", marker="s", label="METEOR")
        axes[1, 2].plot(self.epochs, self.rouge_l_scores, "c-", marker="^", label="ROUGE-L")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Score")
        axes[1, 2].set_title("All Metrics")
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✓ 训练曲线已保存到: {save_path}")


def create_optimized_data_loaders(data_dir, vocab_path, batch_size, num_workers, image_size=224, use_augmentation=True):
    """创建带数据增强的数据加载器"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    if use_augmentation:
        train_transform = CaptionAugmentation.get_train_transforms(image_size)
    else:
        train_transform = CaptionAugmentation.get_val_transforms(image_size)
    
    val_transform = CaptionAugmentation.get_val_transforms(image_size)
    
    train_dataset = DeepFashionDataset(
        data_dir=data_dir,
        split='train',
        vocab=vocab,
        transform=train_transform
    )
    
    val_dataset = DeepFashionDataset(
        data_dir=data_dir,
        split='val',
        vocab=vocab,
        transform=val_transform
    )
    
    test_dataset = DeepFashionDataset(
        data_dir=data_dir,
        split='test',
        vocab=vocab,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, device, vocab, 
                writer, global_step, config, ema=None, r_drop=None):
    """训练一个 epoch（优化版）"""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    for batch_idx, (images, captions, cap_lens) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)
        cap_lens = cap_lens.to(device)

        outputs = model(images, captions, cap_lens)
        targets = captions[:, 1:]
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(outputs_flat, targets_flat)
        
        # R-Drop 正则化
        if r_drop is not None and config.get('use_r_drop', False):
            outputs2 = model(images, captions, cap_lens)
            outputs2_flat = outputs2.reshape(-1, outputs2.size(-1))
            loss2 = criterion(outputs2_flat, targets_flat)
            
            pad_mask = (targets_flat == vocab['<pad>'])
            kl_loss = r_drop.compute_kl_loss(outputs_flat, outputs2_flat, pad_mask)
            
            loss = (loss + loss2) / 2 + config.get('r_drop_alpha', 1.0) * kl_loss
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if config.get('gradient_clip', 0) > 0:
                GradientClipping.clip_grad_norm(model, config['gradient_clip'])
            
            optimizer.step()
            optimizer.zero_grad()
            
            if ema is not None:
                ema.update()
            
            scheduler.step()

        losses.update(loss.item() * accumulation_steps, images.size(0))
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "lr": f"{current_lr:.2e}"
        })

        if writer is not None:
            writer.add_scalar("train/loss", loss.item() * accumulation_steps, global_step[0])
            writer.add_scalar("train/lr", current_lr, global_step[0])
            global_step[0] += 1

    return losses.avg


def validate(model, val_loader, criterion, device, vocab, ema=None):
    """验证模型性能"""
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    losses = AverageMeter()

    gts = {}
    res = {}
    idx2word = {idx: word for word, idx in vocab.items()}
    evaluator = COCOScoreEvaluator()

    with torch.no_grad():
        for i, (imgs, caps, cap_lens) in enumerate(tqdm(val_loader, desc="Validation")):
            imgs = imgs.to(device)
            caps = caps.to(device)
            cap_lens = cap_lens.to(device)

            outputs = model(imgs, caps, cap_lens)
            targets = caps[:, 1:]
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            losses.update(loss.item(), imgs.size(0))

            start_idx = i * val_loader.batch_size
            batch_size = imgs.size(0)

            generated = model.generate(
                imgs,
                start_token=vocab['<start>'],
                end_token=vocab['<end>'],
                max_len=50,
                method='greedy'
            )

            for j in range(batch_size):
                img_id = start_idx + j

                ref_ids = caps[j].cpu().numpy()
                ref_words = [
                    idx2word[idx]
                    for idx in ref_ids
                    if idx not in [vocab["<start>"], vocab["<end>"], vocab["<pad>"]]
                ]
                gts[img_id] = [" ".join(ref_words)]

                pred_ids = generated[j].cpu().numpy()
                pred_words = []
                for idx in pred_ids:
                    if idx == vocab["<end>"]:
                        break
                    if idx not in [vocab["<start>"], vocab["<pad>"]]:
                        pred_words.append(idx2word[idx])
                res[img_id] = [" ".join(pred_words)]

    if ema is not None:
        ema.restore()

    print("计算评测分数...")
    scores = evaluator.evaluate(gts, res)

    print(f"\n{'='*60}")
    print(f"Validation Loss: {losses.avg:.4f}")
    print(f"CIDEr:   {scores.get('CIDEr', 0.0):.4f}")
    print(f"METEOR:  {scores.get('METEOR', 0.0):.4f}")
    print(f"ROUGE-L: {scores.get('ROUGE_L', 0.0):.4f}")
    print(f"{'='*60}\n")

    return losses.avg, scores


def train_optimized(config):
    """优化版主训练函数"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    print(f"词典大小: {vocab_size}")

    print("\n创建数据加载器（带数据增强）...")
    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        data_dir=config["data_dir"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=224,
        use_augmentation=config.get('use_data_augmentation', True)
    )

    print("\n构建模型...")
    model = build_model(vocab_size, config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # 损失函数
    if config.get('label_smoothing', 0) > 0:
        print(f"使用 Label Smoothing: {config['label_smoothing']}")
        criterion = LabelSmoothingLoss(
            vocab_size=vocab_size,
            padding_idx=vocab["<pad>"],
            smoothing=config['label_smoothing']
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.get('weight_decay', 0.01)
    )

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = config.get('warmup_steps', int(total_steps * 0.1))
    
    print(f"总训练步数: {total_steps}, Warmup 步数: {warmup_steps}")
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config.get('min_lr', 1e-7)
    )

    ema = None
    if config.get('use_ema', False):
        print(f"使用 EMA，decay = {config.get('ema_decay', 0.999)}")
        ema = ExponentialMovingAverage(model, decay=config.get('ema_decay', 0.999))

    r_drop = None
    if config.get('use_r_drop', False):
        print(f"使用 R-Drop 正则化，alpha = {config.get('r_drop_alpha', 1.0)}")
        r_drop = R_Drop(alpha=config.get('r_drop_alpha', 1.0))

    early_stopping = EarlyStopping(
        patience=config.get('patience', 5),
        min_delta=0.001,
        mode='max'
    )

    writer = None
    if config.get("use_tensorboard", False):
        writer = SummaryWriter(log_dir=config.get("log_dir", "runs/grid_transformer_optimized"))
        print(f"TensorBoard日志目录: {writer.log_dir}")

    metrics_history = MetricsHistory()

    print("\n" + "=" * 70)
    print("开始 Grid-Transformer 优化版训练")
    print("优化策略:")
    print(f"  - Label Smoothing: {config.get('label_smoothing', 0)}")
    print(f"  - Warmup Steps: {warmup_steps}")
    print(f"  - 数据增强: {config.get('use_data_augmentation', True)}")
    print(f"  - EMA: {config.get('use_ema', False)}")
    print(f"  - R-Drop: {config.get('use_r_drop', False)}")
    print(f"  - 梯度裁剪: {config.get('gradient_clip', 0)}")
    print(f"  - 早停 Patience: {config.get('patience', 5)}")
    print("=" * 70)

    best_cider = 0.0
    global_step = [0]

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)

        if epoch == config.get("finetune_encoder_after_epoch", -1):
            print("开始微调 CNN 编码器...")
            model.encoder.set_cnn_trainable(True)
            
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.cnn.parameters(), "lr": config["learning_rate"] * 0.1},
                {"params": model.encoder.projection.parameters(), "lr": config["learning_rate"]},
                {"params": model.encoder.transformer_encoder.parameters(), "lr": config["learning_rate"]},
                {"params": model.decoder.parameters(), "lr": config["learning_rate"]},
            ], betas=(0.9, 0.98), eps=1e-9, weight_decay=config.get('weight_decay', 0.01))
            
            remaining_steps = len(train_loader) * (config["num_epochs"] - epoch + 1)
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_steps=min(500, remaining_steps // 10),
                total_steps=remaining_steps,
                min_lr=config.get('min_lr', 1e-7)
            )

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, device, vocab,
            writer, global_step, config, ema, r_drop
        )

        print(f"训练损失: {train_loss:.4f}")

        if epoch % config.get("eval_every", 1) == 0:
            print("\n验证中...")
            val_loss, scores = validate(model, val_loader, criterion, device, vocab, ema)

            cider = scores.get("CIDEr", 0.0)
            current_lr = scheduler.get_last_lr()[0]

            metrics_history.add(epoch, train_loss, val_loss, scores, current_lr)

            if writer is not None:
                writer.add_scalar("Val/Loss", val_loss, epoch)
                writer.add_scalar("Val/CIDEr", scores.get("CIDEr", 0.0), epoch)
                writer.add_scalar("Val/METEOR", scores.get("METEOR", 0.0), epoch)
                writer.add_scalar("Val/ROUGE_L", scores.get("ROUGE_L", 0.0), epoch)

            if cider > best_cider:
                best_cider = cider
                os.makedirs(config["checkpoint_dir"], exist_ok=True)

                save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
                
                if ema is not None:
                    ema.apply_shadow()
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                    "cider": cider,
                    "scores": scores,
                    "config": config,
                }, save_path)
                
                if ema is not None:
                    ema.restore()
                    
                print(f"✓ 保存最佳模型! CIDEr: {best_cider:.4f}")

            if early_stopping(cider):
                print(f"\n早停触发！在 epoch {epoch} 停止训练")
                break

        print("-" * 70)

    print("\n训练完成！")
    print(f"最佳 CIDEr: {best_cider:.4f}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    metrics_history.plot_and_save(
        os.path.join(config["checkpoint_dir"], "optimized_training_metrics.png")
    )

    if writer is not None:
        writer.close()

    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(
        os.path.join(config["checkpoint_dir"], "best_model.pth"),
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_scores = validate(model, test_loader, criterion, device, vocab)

    print("\n" + "=" * 70)
    print("最终测试集评测结果 (Grid-Transformer 优化版):")
    print("=" * 70)
    print(f"测试集损失:  {test_loss:.4f}")
    print(f"CIDEr:       {test_scores.get('CIDEr', 0.0):.4f}")
    print(f"METEOR:      {test_scores.get('METEOR', 0.0):.4f}")
    print(f"ROUGE-L:     {test_scores.get('ROUGE_L', 0.0):.4f}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    config = {
        # 数据
        "data_dir": "data",
        "vocab_path": "data/vocab.json",
        "batch_size": 32,
        "num_workers": 8,
        
        # 模型
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.15,
        "max_len": 52,
        "pretrained_cnn": True,
        
        # 训练
        "num_epochs": 30,
        "learning_rate": 3e-4,
        "min_lr": 1e-7,
        "weight_decay": 0.01,
        "finetune_encoder_after_epoch": 10,
        
        # 优化策略
        "label_smoothing": 0.1,
        "warmup_steps": 2000,
        "gradient_clip": 1.0,
        "use_data_augmentation": True,
        "use_ema": True,
        "ema_decay": 0.999,
        "use_r_drop": False,
        "r_drop_alpha": 1.0,
        "gradient_accumulation_steps": 1,
        "patience": 7,
        
        # 评估和保存
        "eval_every": 1,
        "checkpoint_dir": "checkpoints/grid_transformer_optimized",
        
        # 日志
        "use_tensorboard": True,
        "log_dir": "runs/grid_transformer_optimized",
    }

    print("=" * 70)
    print("Grid-Transformer 图像描述模型 - 优化版训练")
    print("=" * 70)
    print("\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    model = train_optimized(config)
