"""
Grid-Transformer 图像描述模型训练脚本
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval_metrics import COCOScoreEvaluator
from grid_transformer_model import build_model
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


def train_epoch(model, train_loader, criterion, optimizer, epoch, device, writer, global_step):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for images, captions, cap_lens in pbar:
        images = images.to(device)
        captions = captions.to(device)
        cap_lens = cap_lens.to(device)

        outputs = model(images, captions, cap_lens)
        targets = captions[:, 1:]

        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(outputs_flat, targets_flat)

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
    """验证模型性能"""
    model.eval()
    losses = AverageMeter()
    gts = {}
    res = {}
    idx2word = {idx: word for word, idx in vocab.items()}
    evaluator = COCOScoreEvaluator()

    print("正在验证并生成描述...")
    with torch.no_grad():
        for i, (imgs, caps, cap_lens) in enumerate(tqdm(val_loader, desc="Validation")):
            imgs = imgs.to(device)
            caps = caps.to(device)
            cap_lens = cap_lens.to(device)

            # 1. 计算 Loss
            outputs = model(imgs, caps, cap_lens)
            targets = caps[:, 1:]
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            losses.update(loss.item(), imgs.size(0))

            # 2. 生成描述
            start_idx = i * val_loader.batch_size
            generated_captions = model.generate(
                imgs, vocab['<start>'], vocab['<end>'], max_len=50, method='greedy'
            )

            # 收集结果
            for j in range(imgs.size(0)):
                img_id = start_idx + j
                
                # Ground Truth
                ref_ids = caps[j].cpu().numpy()
                ref_words = [idx2word[idx] for idx in ref_ids if idx not in [vocab["<start>"], vocab["<end>"], vocab["<pad>"]]]
                gts[img_id] = [" ".join(ref_words)]

                # Prediction
                pred_ids = generated_captions[j].cpu().numpy()
                pred_words = []
                for idx in pred_ids:
                    if idx == vocab["<end>"]:
                        break
                    if idx not in [vocab["<start>"], vocab["<pad>"]]:
                        pred_words.append(idx2word[idx])
                res[img_id] = [" ".join(pred_words)]

    print("计算评测分数...")
    scores = evaluator.evaluate(gts, res, fast_mode=True)
    cider_score = scores.get("CIDEr", 0.0)
    bleu4_score = scores.get("Bleu_4", 0.0)

    print(f"Validation Loss: {losses.avg:.4f}, CIDEr: {cider_score:.4f}, BLEU-4: {bleu4_score:.4f}")
    return losses.avg, cider_score


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
        print(f"将在第 {config['finetune_encoder_after_epoch']} 个epoch后开始微调CNN编码器")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    writer = SummaryWriter(log_dir=config.get("log_dir", "runs/grid_transformer"))
    print(f"TensorBoard日志目录: {writer.log_dir}")

    print("\n开始训练...")
    best_cider = 0.0
    global_step = [0]

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        if epoch == config.get("finetune_encoder_after_epoch", -1):
            print("开始微调CNN编码器...")
            model.encoder.set_cnn_trainable(True)
            optimizer = torch.optim.Adam([
                {'params': model.encoder.cnn.parameters(), 'lr': config["learning_rate"] * 0.1},
                {'params': model.encoder.projection.parameters()},
                {'params': model.encoder.transformer_encoder.parameters()},
                {'params': model.decoder.parameters()}
            ], lr=config["learning_rate"])
            print(f"可训练参数量更新为: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, device, writer, global_step)
        print(f"训练损失: {train_loss:.4f}")

        if epoch % config.get("eval_every", 1) == 0:
            val_loss, cider = validate(model, val_loader, criterion, device, vocab)
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/CIDEr", cider, epoch)

            if cider > best_cider:
                best_cider = cider
                os.makedirs(config["checkpoint_dir"], exist_ok=True)
                save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cider": cider,
                    "config": config,
                }, save_path)
                print(f"保存最佳模型! CIDEr: {best_cider:.4f}")

        if epoch % config.get("save_every", 5) == 0:
            save_path = os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pth")
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, save_path)
            print(f"✓ 保存检查点到 {save_path}")

    print(f"\n训练完成！最佳CIDEr: {best_cider:.4f}")
    writer.close()

    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_cider = validate(model, test_loader, criterion, device, vocab)
    print(f"测试集损失: {test_loss:.4f}, 测试集CIDEr: {test_cider:.4f}")


if __name__ == "__main__":
    config = {
        # 数据
        "data_dir": "data",
        "vocab_path": "data/vocab.json",
        "batch_size": 32,
        "num_workers": 0, # Windows下设为0
        # 模型 (Grid-Transformer)
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_len": 52,
        "pretrained_cnn": True,
        # 训练
        "num_epochs": 30,
        "learning_rate": 1e-4,
        "finetune_encoder_after_epoch": 5, # 第5个epoch后微调CNN
        # 评估和保存
        "eval_every": 1,
        "save_every": 5,
        "checkpoint_dir": "checkpoints/grid_transformer",
        # 日志
        "log_dir": "runs/grid_transformer",
    }

    print("=" * 70)
    print("Grid-Transformer 图像描述模型训练")
    print("=" * 70)
    print("\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    train(config)
