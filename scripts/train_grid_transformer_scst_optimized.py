"""
Grid + Transformer 图像描述模型 - SCST强化学习 + 优化技术 训练脚本

结合以下优化:
1. SCST (Self-Critical Sequence Training) - 直接优化CIDEr
2. EMA (Exponential Moving Average) - 模型权重滑动平均
3. Warmup + Cosine Annealing LR - 学习率调度
4. 数据增强 - 训练时图像增强
5. 梯度累积 - 模拟更大batch size
6. 早停 - 防止过拟合
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eval_metrics import COCOScoreEvaluator
from utils.scst_loss import SCSTLoss, get_reference_captions
from utils.optimizations import (
    WarmupCosineScheduler,
    ExponentialMovingAverage,
    EarlyStopping,
    CaptionAugmentation,
    GradientClipping,
)
from models.grid_transformer_model import build_model
from utils.deepfashion_dataset import DeepFashionCaptionDataset


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


class RLMetricsHistory:
    """记录强化学习训练过程中的指标"""

    def __init__(self):
        self.epochs = []
        self.sample_rewards = []
        self.greedy_rewards = []
        self.advantages = []
        self.cider_scores = []
        self.meteor_scores = []
        self.rouge_l_scores = []
        self.learning_rates = []

    def add(self, epoch, sample_reward, greedy_reward, advantage, scores, lr):
        self.epochs.append(epoch)
        self.sample_rewards.append(sample_reward)
        self.greedy_rewards.append(greedy_reward)
        self.advantages.append(advantage)
        self.cider_scores.append(scores.get("CIDEr", 0.0))
        self.meteor_scores.append(scores.get("METEOR", 0.0))
        self.rouge_l_scores.append(scores.get("ROUGE_L", 0.0))
        self.learning_rates.append(lr)

    def plot_and_save(self, save_path):
        """绘制并保存训练曲线"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("Grid Transformer SCST + Optimizations Training Metrics", fontsize=16)

        # 1. Sample vs Greedy Rewards
        axes[0, 0].plot(self.epochs, self.sample_rewards, "b-", marker="o", label="Sample Reward")
        axes[0, 0].plot(self.epochs, self.greedy_rewards, "r-", marker="s", label="Greedy Reward")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Reward (CIDEr)")
        axes[0, 0].set_title("Reward Comparison")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Advantage
        axes[0, 1].plot(self.epochs, self.advantages, "g-", marker="o", label="Advantage")
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Advantage")
        axes[0, 1].set_title("Average Advantage")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. CIDEr
        axes[0, 2].plot(self.epochs, self.cider_scores, "g-", marker="o", label="CIDEr")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].set_title("CIDEr Score (Validation)")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # 4. Learning Rate
        axes[0, 3].plot(self.epochs, self.learning_rates, "y-", marker="o", label="LR")
        axes[0, 3].set_xlabel("Epoch")
        axes[0, 3].set_ylabel("Learning Rate")
        axes[0, 3].set_title("Learning Rate Schedule")
        axes[0, 3].legend()
        axes[0, 3].grid(True)

        # 5. METEOR
        axes[1, 0].plot(self.epochs, self.meteor_scores, "m-", marker="o", label="METEOR")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_title("METEOR Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 6. ROUGE-L
        axes[1, 1].plot(self.epochs, self.rouge_l_scores, "c-", marker="o", label="ROUGE-L")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("ROUGE-L Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 7. All metrics combined
        axes[1, 2].plot(self.epochs, self.cider_scores, "g-", marker="o", label="CIDEr")
        axes[1, 2].plot(self.epochs, self.meteor_scores, "m-", marker="s", label="METEOR")
        axes[1, 2].plot(self.epochs, self.rouge_l_scores, "c-", marker="^", label="ROUGE-L")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Score")
        axes[1, 2].set_title("All Metrics")
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        # 8. Summary text
        axes[1, 3].axis('off')
        if len(self.cider_scores) > 0:
            summary_text = f"""
Training Summary
================
Total Epochs: {len(self.epochs)}
Best CIDEr: {max(self.cider_scores):.4f}
Best METEOR: {max(self.meteor_scores):.4f}
Best ROUGE-L: {max(self.rouge_l_scores):.4f}
Final LR: {self.learning_rates[-1]:.2e}
            """
            axes[1, 3].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                           verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✓ 训练曲线已保存到: {save_path}")


def create_data_loaders_with_augmentation(config):
    """创建带数据增强的数据加载器"""
    
    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # 训练集使用数据增强
    train_transform = CaptionAugmentation.get_train_transforms(image_size=224)
    val_transform = CaptionAugmentation.get_val_transforms(image_size=224)
    
    # 创建数据集
    train_dataset = DeepFashionCaptionDataset(
        data_file=os.path.join(config["data_dir"], "train_data.json"),
        image_dir=os.path.join(config["data_dir"], "images"),
        vocab=vocab,
        transform=train_transform,
        max_len=config.get("max_len", 52),
    )
    
    val_dataset = DeepFashionCaptionDataset(
        data_file=os.path.join(config["data_dir"], "val_data.json"),
        image_dir=os.path.join(config["data_dir"], "images"),
        vocab=vocab,
        transform=val_transform,
        max_len=config.get("max_len", 52),
    )
    
    test_dataset = DeepFashionCaptionDataset(
        data_file=os.path.join(config["data_dir"], "test_data.json"),
        image_dir=os.path.join(config["data_dir"], "images"),
        vocab=vocab,
        transform=val_transform,
        max_len=config.get("max_len", 52),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, vocab


def train_epoch_scst_optimized(
    model, train_loader, scst_loss, optimizer, scheduler, 
    epoch, device, vocab, writer, global_step, config, ema=None
):
    """使用 SCST + 优化技术 训练一个 epoch"""
    model.train()

    reward_meter = AverageMeter()
    advantage_meter = AverageMeter()
    sample_reward_meter = AverageMeter()
    
    accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    idx2word = {idx: word for word, idx in vocab.items()}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} (SCST+Opt)")

    optimizer.zero_grad()
    
    for batch_idx, (images, captions, cap_lens) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)
        
        # 获取参考描述
        references = get_reference_captions(captions, vocab)

        # 计算 SCST 损失
        loss, reward_info = scst_loss(
            model, images, references, vocab, device, max_len=50
        )
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            GradientClipping.clip_grad_norm(model, max_norm=config.get("gradient_clip", 5.0))
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 更新 EMA
            if ema is not None:
                ema.update()

        reward_meter.update(reward_info['greedy_reward'], images.size(0))
        advantage_meter.update(reward_info['advantage'], images.size(0))
        sample_reward_meter.update(reward_info['sample_reward'], images.size(0))

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            "reward": f"{reward_info['greedy_reward']:.3f}",
            "adv": f"{reward_info['advantage']:.3f}",
            "lr": f"{current_lr:.2e}"
        })

        if writer is not None:
            writer.add_scalar("train/sample_reward", reward_info['sample_reward'], global_step[0])
            writer.add_scalar("train/greedy_reward", reward_info['greedy_reward'], global_step[0])
            writer.add_scalar("train/advantage", reward_info['advantage'], global_step[0])
            writer.add_scalar("train/learning_rate", current_lr, global_step[0])
            global_step[0] += 1

    return sample_reward_meter.avg, reward_meter.avg, advantage_meter.avg


def validate(model, val_loader, device, vocab, ema=None):
    """验证模型性能"""
    
    # 如果使用 EMA，应用 EMA 参数
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()

    gts = {}
    res = {}

    idx2word = {idx: word for word, idx in vocab.items()}
    evaluator = COCOScoreEvaluator()

    print("正在验证并生成描述...")

    with torch.no_grad():
        for i, (imgs, caps, cap_lens) in enumerate(tqdm(val_loader, desc="Validation")):
            imgs = imgs.to(device)
            caps = caps.to(device)

            start_idx = i * val_loader.batch_size
            batch_size = imgs.size(0)

            # 使用 Greedy 生成
            generated = model.generate(
                imgs,
                start_token=vocab['<start>'],
                end_token=vocab['<end>'],
                max_len=50,
                method='greedy'
            )

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

                # 生成描述
                pred_ids = generated[j].cpu().numpy()
                pred_words = []
                for idx in pred_ids:
                    if idx == vocab["<end>"]:
                        break
                    if idx not in [vocab["<start>"], vocab["<pad>"]]:
                        pred_words.append(idx2word[idx])
                res[img_id] = [" ".join(pred_words)]

    # 恢复原始参数
    if ema is not None:
        ema.restore()

    print("计算评测分数...")
    scores = evaluator.evaluate(gts, res)

    print(f"\n{'='*60}")
    print(f"CIDEr:   {scores.get('CIDEr', 0.0):.4f}")
    print(f"METEOR:  {scores.get('METEOR', 0.0):.4f}")
    print(f"ROUGE-L: {scores.get('ROUGE_L', 0.0):.4f}")
    print(f"{'='*60}\n")

    return scores


def train_scst_optimized(config):
    """SCST + 优化 主训练函数"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建带数据增强的数据加载器
    print("\n创建数据加载器（带数据增强）...")
    train_loader, val_loader, test_loader, vocab = create_data_loaders_with_augmentation(config)
    
    vocab_size = len(vocab)
    print(f"词典大小: {vocab_size}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")

    print("\n构建模型...")
    model = build_model(vocab_size, config)
    model = model.to(device)

    # 加载预训练的 XE 模型权重（必须）
    if config.get("pretrain_checkpoint"):
        print(f"\n加载预训练模型: {config['pretrain_checkpoint']}")
        checkpoint = torch.load(config["pretrain_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("预训练模型加载成功！")
    else:
        print("\n警告: 未指定预训练模型，SCST 通常需要从 XE 预训练模型开始！")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    # SCST 损失
    scst_loss = SCSTLoss(reward_type=config.get("reward_type", "cider"))

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        betas=(0.9, 0.98), 
        eps=1e-9,
        weight_decay=config.get("weight_decay", 0.01)
    )

    # 计算总步数
    total_steps = len(train_loader) * config["num_epochs"] // config.get("gradient_accumulation_steps", 1)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    
    # Warmup + Cosine 学习率调度
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config.get("min_lr", 1e-7)
    )
    
    print(f"\n学习率调度:")
    print(f"  总步数: {total_steps}")
    print(f"  Warmup步数: {warmup_steps}")
    print(f"  初始LR: {config['learning_rate']}")
    print(f"  最小LR: {config.get('min_lr', 1e-7)}")

    # EMA
    ema = None
    if config.get("use_ema", True):
        ema = ExponentialMovingAverage(model, decay=config.get("ema_decay", 0.999))
        print(f"\n启用 EMA，衰减系数: {config.get('ema_decay', 0.999)}")

    # 早停
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping_patience", 5),
        min_delta=0.001,
        mode='max'
    )

    # TensorBoard
    writer = None
    if config.get("use_tensorboard", False):
        writer = SummaryWriter(log_dir=config.get("log_dir", "runs/grid_transformer_scst_opt"))
        print(f"TensorBoard日志目录: {writer.log_dir}")

    metrics_history = RLMetricsHistory()

    print("\n开始 SCST + 优化 训练...")
    print("=" * 70)
    print("优化技术:")
    print(f"  - SCST 强化学习 (reward_type: {config.get('reward_type', 'cider')})")
    print(f"  - 数据增强 (RandomResizedCrop, ColorJitter, etc.)")
    print(f"  - Warmup + Cosine LR")
    print(f"  - EMA: {config.get('use_ema', True)}")
    print(f"  - 梯度累积: {config.get('gradient_accumulation_steps', 1)} steps")
    print(f"  - 早停: patience={config.get('early_stopping_patience', 5)}")
    print("=" * 70)

    best_cider = 0.0
    global_step = [0]

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)

        # 训练
        sample_reward, greedy_reward, advantage = train_epoch_scst_optimized(
            model, train_loader, scst_loss, optimizer, scheduler,
            epoch, device, vocab, writer, global_step, config, ema
        )

        current_lr = scheduler.get_last_lr()[0]
        print(f"Sample Reward: {sample_reward:.4f}, Greedy Reward: {greedy_reward:.4f}")
        print(f"Advantage: {advantage:.4f}, LR: {current_lr:.2e}")

        # 验证
        if epoch % config.get("eval_every", 1) == 0:
            print("\n验证中...")
            scores = validate(model, val_loader, device, vocab, ema)

            cider = scores.get("CIDEr", 0.0)

            # 记录指标
            metrics_history.add(epoch, sample_reward, greedy_reward, advantage, scores, current_lr)

            # 记录到 TensorBoard
            if writer is not None:
                writer.add_scalar("Val/CIDEr", scores.get("CIDEr", 0.0), epoch)
                writer.add_scalar("Val/METEOR", scores.get("METEOR", 0.0), epoch)
                writer.add_scalar("Val/ROUGE_L", scores.get("ROUGE_L", 0.0), epoch)
                writer.add_scalar("Val/learning_rate", current_lr, epoch)

            # 保存最佳模型
            if cider > best_cider:
                best_cider = cider

                os.makedirs(config["checkpoint_dir"], exist_ok=True)

                save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
                
                # 保存 EMA 参数
                if ema is not None:
                    ema.apply_shadow()
                    
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "cider": cider,
                        "scores": scores,
                        "config": config,
                    },
                    save_path,
                )
                
                if ema is not None:
                    ema.restore()
                    
                print(f"✓ 保存最佳模型! CIDEr: {best_cider:.4f}")

            # 早停检查
            if early_stopping(cider):
                print(f"\n早停! 验证CIDEr已{early_stopping.patience}个epoch未改善")
                break

        print("-" * 70)

    print("\nSCST + 优化 训练完成！")
    print(f"最佳 CIDEr: {best_cider:.4f}")

    # 保存训练曲线
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    metrics_history.plot_and_save(
        os.path.join(config["checkpoint_dir"], "scst_optimized_training_metrics.png")
    )

    if writer is not None:
        writer.close()

    # 在测试集上评估
    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(
        os.path.join(config["checkpoint_dir"], "best_model.pth"), 
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_scores = validate(model, test_loader, device, vocab)

    print("\n" + "=" * 70)
    print("最终测试集评测结果 (SCST + 优化):")
    print("=" * 70)
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
        "batch_size": 16,  # 实际batch = batch_size * gradient_accumulation_steps
        "num_workers": 4,
        "max_len": 52,
        
        # 模型 (需与预训练模型一致)
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 3,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "backbone": "resnet101",
        "pretrained_backbone": True,
        
        # 预训练模型路径 (必须提供 XE 预训练模型)
        "pretrain_checkpoint": "checkpoints/grid_transformer/best_model.pth",
        
        # SCST 训练
        "num_epochs": 30,
        "learning_rate": 5e-6,  # SCST用较小学习率
        "min_lr": 1e-7,
        "weight_decay": 0.01,
        "reward_type": "cider",  # 'cider', 'bleu', 或 'combined'
        
        # 优化技术
        "warmup_ratio": 0.1,  # 10% warmup
        "gradient_accumulation_steps": 2,  # 梯度累积
        "gradient_clip": 1.0,
        "use_ema": True,
        "ema_decay": 0.9999,
        "early_stopping_patience": 8,
        
        # 评估和保存
        "eval_every": 1,
        "checkpoint_dir": "checkpoints/grid_transformer_scst_opt",
        
        # 日志
        "use_tensorboard": True,
        "log_dir": "runs/grid_transformer_scst_opt",
    }

    print("=" * 70)
    print("Grid + Transformer 图像描述模型 - SCST + 优化技术 训练")
    print("=" * 70)
    print("\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    model = train_scst_optimized(config)
