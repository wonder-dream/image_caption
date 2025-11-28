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
from nltk.translate.bleu_score import corpus_bleu
from eval_metrics import COCOScoreEvaluator

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


def train_epoch(
    model, train_loader, criterion, optimizer, epoch, device, vocab, writer, global_step
):
    """训练一个epoch"""
    model.train()

    losses = AverageMeter()

    # 进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

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
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})

        # 记录到tensorboard
        if writer is not None:
            writer.add_scalar("train/loss", loss.item(), global_step[0])
            global_step[0] += 1

    return losses.avg


def validate(model, val_loader, criterion, device, vocab):
    """
    验证模型性能，计算 Loss 和 CIDEr
    """
    model.eval()
    losses = AverageMeter()

    # 准备 COCO 评测所需的数据容器
    gts = {}  # Ground Truth
    res = {}  # Results (Predictions)

    idx2word = {idx: word for word, idx in vocab.items()}

    print("正在验证并生成描述...")

    # 初始化评测器
    evaluator = COCOScoreEvaluator()

    with torch.no_grad():
        for i, (imgs, caps, cap_lens) in enumerate(tqdm(val_loader, desc="Validation")):
            imgs = imgs.to(device)
            caps = caps.to(device)
            cap_lens = cap_lens.to(device)  # <--- 添加这一行，将长度也移到 GPU

            # 1. 计算 Loss (Teacher Forcing)
            outputs = model(imgs, caps, cap_lens)
            targets = caps[:, 1:]

            # === 修改开始: 移除 pack_padded_sequence，改用简单的 reshape ===
            # 因为 criterion 已经设置了 ignore_index，所以直接 reshape 即可
            # 这样可以避免 batch size 不匹配的问题

            # outputs: (batch, seq_len, vocab) -> (batch * seq_len, vocab)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            # targets: (batch, seq_len) -> (batch * seq_len)
            targets_flat = targets.reshape(-1)

            loss = criterion(outputs_flat, targets_flat)
            # === 修改结束 ===

            # 修复：更新损失统计
            losses.update(loss.item(), imgs.size(0))

            # 2. 生成描述用于计算 CIDEr (Greedy Search)
            # 注意：为了速度，我们只对每个 batch 生成一次，或者你可以选择只验证部分数据
            # 这里演示对整个验证集生成

            # 获取当前 batch 的全局索引 (为了构建 gts/res 字典)
            start_idx = i * val_loader.batch_size

            # 生成 Caption
            # 注意：model.generate 通常处理单张图片，如果你的模型支持 batch generate 最好
            # 如果不支持 batch generate，这里为了速度，我们可能需要简化
            # 这里假设我们使用简单的贪婪搜索循环

            # === 简化的 Batch 生成逻辑 (为了验证速度) ===
            features = model.encoder(imgs)  # (batch, 196, 512)

            # 简单的循环生成 (类似于 inference 中的逻辑，但针对 batch)
            # 如果你的 model.generate 不支持 batch，这里写一个简单的 batch 生成循环
            batch_size = imgs.size(0)
            inputs = (
                torch.tensor([vocab["<start>"]] * batch_size).unsqueeze(1).to(device)
            )  # (batch, 1)
            finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
            sampled_ids = []

            for _ in range(50):  # max_len
                outputs = model.decoder(inputs, features)  # (batch, seq_len, vocab)
                outputs = outputs[:, -1, :]  # 取最后一个时间步 (batch, vocab)
                _, predicted = outputs.max(1)  # (batch)

                sampled_ids.append(predicted)
                inputs = torch.cat([inputs, predicted.unsqueeze(1)], 1)

                # 标记结束
                finished = finished | (predicted == vocab["<end>"])
                if finished.all():
                    break

            sampled_ids = torch.stack(sampled_ids, 1)  # (batch, seq_len)

            # === 收集结果 ===
            for j in range(batch_size):
                img_id = start_idx + j

                # 处理 Ground Truth
                ref_ids = caps[j].cpu().numpy()
                ref_words = [
                    idx2word[idx]
                    for idx in ref_ids
                    if idx not in [vocab["<start>"], vocab["<end>"], vocab["<pad>"]]
                ]
                gts[img_id] = [" ".join(ref_words)]

                # 处理 Prediction
                pred_ids = sampled_ids[j].cpu().numpy()
                pred_words = []
                for idx in pred_ids:
                    if idx == vocab["<end>"]:
                        break
                    if idx not in [vocab["<start>"], vocab["<pad>"]]:
                        pred_words.append(idx2word[idx])
                res[img_id] = [" ".join(pred_words)]

    # 计算分数
    print("计算 CIDEr 分数...")
    # 注意：这里我们只关心 CIDEr，忽略 METEOR 以节省时间
    # 如果 eval_metrics.py 默认跑全套，你可以修改它增加一个参数只跑 CIDEr
    # 或者直接用，只是会慢一点
    scores = evaluator.evaluate(gts, res, fast_mode=True)

    cider_score = scores.get("CIDEr", 0.0)
    bleu4_score = scores.get("Bleu_4", 0.0)

    print(
        f"Validation Loss: {losses.avg:.4f}, CIDEr: {cider_score:.4f}, BLEU-4: {bleu4_score:.4f}"
    )

    return losses.avg, cider_score


def train(config):
    """主训练函数"""

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载词典
    with open(config["vocab_path"], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    print(f"词典大小: {vocab_size}")

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config["data_dir"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=224,
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
    if config.get("finetune_encoder_after_epoch", -1) >= 0:
        print(
            f"将在第 {config['finetune_encoder_after_epoch']} 个epoch后开始微调编码器"
        )

    # 损失函数（忽略padding的损失）
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # 监控BLEU（越大越好）
        factor=0.5,
        patience=3,
    )

    # TensorBoard
    writer = None
    if config.get("use_tensorboard", False):
        writer = SummaryWriter(log_dir=config.get("log_dir", "runs/vit_transformer"))
        print(f"TensorBoard日志目录: {writer.log_dir}")

    # 训练循环
    print("\n开始训练...")
    print("=" * 70)

    best_cider = 0.0  # 替换原来的 best_bleu
    global_step = [0]  # 用list包装以便在函数中修改

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)

        # 是否开始微调编码器
        if epoch == config.get("finetune_encoder_after_epoch", -1):
            print("开始微调ViT编码器...")
            model.encoder.set_trainable(True)
            # 为编码器使用更小的学习率
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

        # 训练
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

        # 验证
        if epoch % config.get("eval_every", 1) == 0:
            print("\n验证中...")
            val_loss, cider = validate(model, val_loader, criterion, device, vocab)

            # 记录到 TensorBoard
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/CIDEr", cider, epoch)

            # 保存最佳模型 (基于 CIDEr)
            if cider > best_cider:
                best_cider = cider

                # 修复：确保目录存在，防止报错
                os.makedirs(config["checkpoint_dir"], exist_ok=True)

                save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                        "cider": cider,  # 保存 CIDEr 分数
                        "config": config,
                    },
                    save_path,
                )
                print(f"保存最佳模型! CIDEr: {best_cider:.4f}")

        # 定期保存检查点
        if epoch % config.get("save_every", 5) == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cider": (cider if epoch % config.get("eval_every", 1) == 0 else 0),
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

    # 关闭writer
    if writer is not None:
        writer.close()

    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_cider = validate(model, test_loader, criterion, device, vocab)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集CIDEr: {test_cider:.4f}")

    return model


if __name__ == "__main__":
    # 配置
    config = {
        # 数据
        "data_dir": "data",
        "vocab_path": "data/vocab.json",
        "batch_size": 32,
        "num_workers": 8,  # Windows下建议设为0
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
        "finetune_encoder_after_epoch": 10,  # 第10个epoch后开始微调编码器，-1表示不微调
        # 评估和保存
        "eval_every": 1,  # 每个epoch验证一次
        "save_every": 5,  # 每5个epoch保存一次
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

    # 开始训练
    model = train(config)
