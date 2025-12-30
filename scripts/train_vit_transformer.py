"""
ViT + Transformer 图像描述模型训练脚本
"""

import os
import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eval_metrics import COCOScoreEvaluator
from models.vit_transformer_model import build_model
from utils.deepfashion_dataset import create_data_loaders


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

    def add(self, epoch, train_loss, val_loss, scores):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.cider_scores.append(scores.get("CIDEr", 0.0))
        self.meteor_scores.append(scores.get("METEOR", 0.0))
        self.rouge_l_scores.append(scores.get("ROUGE_L", 0.0))

    def plot_and_save(self, save_path):
        """绘制并保存折线图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Training Metrics Over Epochs", fontsize=16)

        # 1. 训练损失
        axes[0, 0].plot(
            self.epochs, self.train_losses, "b-", marker="o", label="Train Loss"
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 验证损失
        axes[0, 1].plot(
            self.epochs, self.val_losses, "r-", marker="o", label="Val Loss"
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. CIDEr
        axes[0, 2].plot(self.epochs, self.cider_scores, "g-", marker="o", label="CIDEr")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].set_title("CIDEr Score")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # 4. METEOR
        axes[1, 0].plot(
            self.epochs, self.meteor_scores, "m-", marker="o", label="METEOR"
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_title("METEOR Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 5. ROUGE-L
        axes[1, 1].plot(
            self.epochs, self.rouge_l_scores, "c-", marker="o", label="ROUGE-L"
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("ROUGE-L Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 6. Empty (was SPICE)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✓ 训练曲线已保存到: {save_path}")

    def plot_all_metrics_combined(self, save_path):
        """将所有评测指标绘制在同一张图上"""
        plt.figure(figsize=(12, 6))

        plt.plot(
            self.epochs, self.cider_scores, "g-", marker="o", label="CIDEr", linewidth=2
        )
        plt.plot(
            self.epochs,
            self.meteor_scores,
            "m-",
            marker="s",
            label="METEOR",
            linewidth=2,
        )
        plt.plot(
            self.epochs,
            self.rouge_l_scores,
            "c-",
            marker="^",
            label="ROUGE-L",
            linewidth=2,
        )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Evaluation Metrics Over Epochs", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✓ 综合评测曲线已保存到: {save_path}")


def train_epoch(
    model, train_loader, criterion, optimizer, epoch, device, vocab, writer, global_step
):
    """训练一个epoch"""
    model.train()

    losses = AverageMeter()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (images, captions, cap_lens) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)
        cap_lens = cap_lens.to(device)

        outputs = model(images, captions, cap_lens)
        targets = captions[:, 1:]

        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})

        if writer is not None:
            writer.add_scalar("train/loss", loss.item(), global_step[0])
            global_step[0] += 1

    return losses.avg


def validate(model, val_loader, criterion, device, vocab):
    """
    验证模型性能，计算 Loss 和 METEOR, ROUGE-L, CIDEr
    """
    model.eval()
    losses = AverageMeter()

    gts = {}
    res = {}

    idx2word = {idx: word for word, idx in vocab.items()}

    print("正在验证并生成描述...")

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

            features = model.encoder(imgs)

            batch_size = imgs.size(0)
            inputs = (
                torch.tensor([vocab["<start>"]] * batch_size).unsqueeze(1).to(device)
            )
            finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
            sampled_ids = []

            for _ in range(50):
                outputs = model.decoder(inputs, features)
                outputs = outputs[:, -1, :]
                _, predicted = outputs.max(1)

                sampled_ids.append(predicted)
                inputs = torch.cat([inputs, predicted.unsqueeze(1)], 1)

                finished = finished | (predicted == vocab["<end>"])
                if finished.all():
                    break

            sampled_ids = torch.stack(sampled_ids, 1)

            for j in range(batch_size):
                img_id = start_idx + j

                ref_ids = caps[j].cpu().numpy()
                ref_words = [
                    idx2word[idx]
                    for idx in ref_ids
                    if idx not in [vocab["<start>"], vocab["<end>"], vocab["<pad>"]]
                ]
                gts[img_id] = [" ".join(ref_words)]

                pred_ids = sampled_ids[j].cpu().numpy()
                pred_words = []
                for idx in pred_ids:
                    if idx == vocab["<end>"]:
                        break
                    if idx not in [vocab["<start>"], vocab["<pad>"]]:
                        pred_words.append(idx2word[idx])
                res[img_id] = [" ".join(pred_words)]

    print("计算评测分数...")
    scores = evaluator.evaluate(gts, res)

    cider_score = scores.get("CIDEr", 0.0)
    meteor_score = scores.get("METEOR", 0.0)
    rouge_l_score = scores.get("ROUGE_L", 0.0)

    print(f"\n{'='*60}")
    print(f"Validation Loss: {losses.avg:.4f}")
    print(f"CIDEr:   {cider_score:.4f}")
    print(f"METEOR:  {meteor_score:.4f}")
    print(f"ROUGE-L: {rouge_l_score:.4f}")
    print(f"{'='*60}\n")

    return losses.avg, scores


def train(config):
    """主训练函数"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    print(f"词典大小: {vocab_size}")

    print("\n创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config["data_dir"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=224,
    )

    print("\n构建模型...")
    model = build_model(vocab_size, config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    if config.get("finetune_encoder_after_epoch", -1) >= 0:
        print(
            f"将在第 {config['finetune_encoder_after_epoch']} 个epoch后开始微调编码器"
        )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    writer = None
    if config.get("use_tensorboard", False):
        writer = SummaryWriter(log_dir=config.get("log_dir", "runs/vit_transformer"))
        print(f"TensorBoard日志目录: {writer.log_dir}")

    # 初始化指标记录器
    metrics_history = MetricsHistory()

    print("\n开始训练...")
    print("=" * 70)

    best_cider = 0.0
    global_step = [0]

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)

        if epoch == config.get("finetune_encoder_after_epoch", -1):
            print("开始微调ViT编码器...")
            model.encoder.set_trainable(True)
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": model.encoder.parameters(),
                        "lr": config["learning_rate"] * 0.1,
                    },
                    {
                        "params": model.decoder.parameters(),
                        "lr": config["learning_rate"],
                    },
                ],
                betas=(0.9, 0.98),
                eps=1e-9,
            )

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            device,
            vocab,
            writer,
            global_step,
        )

        print(f"训练损失: {train_loss:.4f}")

        if epoch % config.get("eval_every", 1) == 0:
            print("\n验证中...")
            # 训练时使用快速模式 (跳过 SPICE)
            val_loss, scores = validate(
                model, val_loader, criterion, device, vocab
            )

            cider = scores.get("CIDEr", 0.0)

            # 记录指标
            metrics_history.add(epoch, train_loss, val_loss, scores)

            # 记录到 TensorBoard
            if writer is not None:
                writer.add_scalar("Val/Loss", val_loss, epoch)
                writer.add_scalar("Val/CIDEr", scores.get("CIDEr", 0.0), epoch)
                writer.add_scalar("Val/METEOR", scores.get("METEOR", 0.0), epoch)
                writer.add_scalar("Val/ROUGE_L", scores.get("ROUGE_L", 0.0), epoch)

            # 学习率调度
            scheduler.step(cider)

            if cider > best_cider:
                best_cider = cider

                os.makedirs(config["checkpoint_dir"], exist_ok=True)

                save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                        "scores": scores,
                        "config": config,
                    },
                    save_path,
                )
                print(f"保存最佳模型! CIDEr: {best_cider:.4f}")

        if epoch % config.get("save_every", 5) == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }
            save_path = os.path.join(
                config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pth"
            )
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"✓ 保存检查点到 {save_path}")

        print("-" * 70)

    print("\n训练完成！")
    print(f"最佳CIDEr: {best_cider:.4f}")

    # 保存训练曲线图
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    metrics_history.plot_and_save(
        os.path.join(config["checkpoint_dir"], "training_metrics.png")
    )
    metrics_history.plot_all_metrics_combined(
        os.path.join(config["checkpoint_dir"], "evaluation_metrics_combined.png")
    )

    if writer is not None:
        writer.close()

    # 在测试集上评估最佳模型 (完整模式，包含 SPICE)
    print("\n在测试集上评估最佳模型 (完整评测，包含 SPICE)...")

    checkpoint = torch.load(
        os.path.join(config["checkpoint_dir"], "best_model.pth"), weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_scores = validate(
        model, test_loader, criterion, device, vocab
    )

    print("\n" + "=" * 70)
    print("最终测试集评测结果:")
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
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_len": 52,
        "pretrained_vit": True,
        # 训练
        "num_epochs": 30,
        "learning_rate": 0.0001,
        "finetune_encoder_after_epoch": 10,
        # 评估和保存
        "eval_every": 1,
        "save_every": 5,
        "checkpoint_dir": "checkpoints/vit_transformer",
        # 日志
        "use_tensorboard": True,
        "log_dir": "runs/vit_transformer",
    }

    print("=" * 70)
    print("ViT + Transformer 图像描述模型训练")
    print("=" * 70)
    print("\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    model = train(config)
