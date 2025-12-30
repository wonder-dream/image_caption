"""
模型优化工具模块

包含各种提升模型性能的技术:
1. Label Smoothing - 标签平滑
2. Warmup Learning Rate - 学习率预热
3. Data Augmentation - 数据增强
4. Attention Improvements - 注意力机制改进
5. Regularization - 正则化技术
6. Gradient Accumulation - 梯度累积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
import math
import random
from PIL import Image


# ======================== 1. Label Smoothing ========================

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    
    防止模型过于自信，提高泛化能力
    """
    
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits, targets):
        """
        参数:
            logits: (batch_size * seq_len, vocab_size)
            targets: (batch_size * seq_len,)
        """
        # 创建平滑的目标分布
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (self.vocab_size - 2))  # 排除 padding 和真实标签
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        smooth_targets[:, self.padding_idx] = 0
        
        # 创建 mask，忽略 padding 位置
        mask = targets != self.padding_idx
        
        # 计算 KL 散度损失
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # 应用 mask
        loss = loss.masked_select(mask).mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡问题
    
    对困难样本给予更大的权重
    """
    
    def __init__(self, vocab_size, padding_idx, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets):
        """
        参数:
            logits: (batch_size * seq_len, vocab_size)
            targets: (batch_size * seq_len,)
        """
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # 获取目标类别的概率
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # 交叉熵
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.padding_idx)
        
        # Focal loss
        loss = focal_weight * ce_loss
        
        # 创建 mask
        mask = targets != self.padding_idx
        loss = loss.masked_select(mask).mean()
        
        return loss


# ======================== 2. Learning Rate Scheduling ========================

class WarmupCosineScheduler(_LRScheduler):
    """
    带 Warmup 的余弦退火学习率调度器
    
    - Warmup: 学习率从小到大线性增长
    - Cosine: 之后按余弦函数衰减
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup 阶段：线性增长
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine 退火阶段
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


class TransformerScheduler(_LRScheduler):
    """
    Transformer 原论文中的学习率调度
    
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super(TransformerScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = self.factor * (self.d_model ** (-0.5)) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


# ======================== 3. Data Augmentation ========================

class CaptionAugmentation:
    """
    图像描述任务专用的数据增强
    
    包含适合 caption 任务的图像变换
    """
    
    @staticmethod
    def get_train_transforms(image_size=224):
        """训练时的数据增强"""
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    
    @staticmethod
    def get_val_transforms(image_size=224):
        """验证/测试时的变换（无增强）"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class MixUp:
    """
    MixUp 数据增强
    
    将两个样本混合，提高模型鲁棒性
    """
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, images, captions):
        """
        参数:
            images: (batch_size, 3, H, W)
            captions: (batch_size, seq_len)
        返回:
            mixed_images, lambda
        """
        if self.alpha > 0:
            lam = random.betavariate(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, lam, index


# ======================== 4. Attention Improvements ========================

class MultiHeadAttentionWithDropout(nn.Module):
    """
    带有 Attention Dropout 的多头注意力
    
    在注意力权重上应用 dropout
    """
    
    def __init__(self, d_model, nhead, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size = query.size(0)
        
        # 投影
        q = self.q_proj(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用 mask
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax 和 attention dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码
    
    相比绝对位置编码，更好地建模序列中元素间的相对关系
    """
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 可学习的相对位置编码
        self.relative_positions = nn.Embedding(2 * max_len - 1, d_model)
        
    def forward(self, seq_len):
        """
        返回相对位置编码矩阵
        """
        positions = torch.arange(seq_len, device=self.relative_positions.weight.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0) + self.max_len - 1
        return self.relative_positions(relative_positions)


# ======================== 5. Regularization ========================

class DropPath(nn.Module):
    """
    随机深度 (Stochastic Depth)
    
    随机跳过某些层，相当于训练不同深度的网络集成
    """
    
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x.div(keep_prob) * random_tensor


class LayerDropout(nn.Module):
    """
    层 Dropout
    
    随机丢弃整个 Transformer 层
    """
    
    def __init__(self, layers, drop_prob=0.1):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.drop_prob = drop_prob
        
    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            if self.training and random.random() < self.drop_prob:
                continue  # 跳过这一层
            x = layer(x, *args, **kwargs)
        return x


class R_Drop(nn.Module):
    """
    R-Drop 正则化
    
    通过最小化同一输入两次前向传播结果的 KL 散度来提高一致性
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def compute_kl_loss(self, p, q, pad_mask=None):
        """计算两个分布间的 KL 散度"""
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        if pad_mask is not None:
            p_loss = p_loss.masked_fill(pad_mask.unsqueeze(-1), 0)
            q_loss = q_loss.masked_fill(pad_mask.unsqueeze(-1), 0)
            
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        
        return (p_loss + q_loss) / 2


# ======================== 6. Gradient Utilities ========================

class GradientAccumulator:
    """
    梯度累积工具
    
    用于在内存受限时模拟大 batch size
    """
    
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_step(self):
        """是否应该执行优化器 step"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self):
        """重置计数"""
        self.current_step = 0


class GradientClipping:
    """
    梯度裁剪工具
    """
    
    @staticmethod
    def clip_grad_norm(model, max_norm=1.0):
        """按范数裁剪梯度"""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def clip_grad_value(model, clip_value=0.5):
        """按值裁剪梯度"""
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)


# ======================== 7. Model EMA ========================

class ExponentialMovingAverage:
    """
    指数移动平均 (EMA)
    
    维护模型参数的滑动平均，通常能提升泛化性能
    """
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self.register()
        
    def register(self):
        """注册所有参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """更新 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
    def apply_shadow(self):
        """应用 EMA 参数（用于评估）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        """恢复原始参数（用于继续训练）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ======================== 8. Training Utilities ========================

class EarlyStopping:
    """
    早停机制
    
    当验证指标不再改善时停止训练
    """
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


# ======================== 9. Ensemble Utilities ========================

class ModelEnsemble(nn.Module):
    """
    模型集成
    
    组合多个模型的预测结果
    """
    
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        
    def forward(self, *args, **kwargs):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(*args, **kwargs)
            outputs.append(output * weight)
        return sum(outputs)
    
    def generate(self, images, start_token, end_token, max_len=50, method='greedy'):
        """集成生成（使用第一个模型生成，或可以实现更复杂的策略）"""
        # 简单策略：使用第一个模型生成
        return self.models[0].generate(images, start_token, end_token, max_len, method)


# ======================== 10. 配置模板 ========================

def get_optimized_config(base_config, optimization_level='medium'):
    """
    获取优化后的配置
    
    参数:
        base_config: 基础配置
        optimization_level: 'light', 'medium', 'heavy'
    """
    config = base_config.copy()
    
    if optimization_level == 'light':
        # 轻度优化
        config.update({
            'label_smoothing': 0.1,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'use_ema': False,
            'dropout': 0.1,
        })
    elif optimization_level == 'medium':
        # 中度优化
        config.update({
            'label_smoothing': 0.1,
            'warmup_steps': 2000,
            'gradient_clip': 0.5,
            'use_ema': True,
            'ema_decay': 0.999,
            'dropout': 0.15,
            'attention_dropout': 0.1,
            'drop_path': 0.1,
            'use_data_augmentation': True,
        })
    elif optimization_level == 'heavy':
        # 重度优化
        config.update({
            'label_smoothing': 0.15,
            'warmup_steps': 4000,
            'gradient_clip': 0.25,
            'use_ema': True,
            'ema_decay': 0.9999,
            'dropout': 0.2,
            'attention_dropout': 0.15,
            'drop_path': 0.2,
            'use_data_augmentation': True,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'use_r_drop': True,
            'r_drop_alpha': 1.0,
            'gradient_accumulation_steps': 4,
        })
    
    return config


if __name__ == '__main__':
    # 测试
    print("测试优化模块")
    print("=" * 60)
    
    # 测试 Label Smoothing
    vocab_size = 100
    batch_size = 4
    seq_len = 10
    
    logits = torch.randn(batch_size * seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
    targets[::3] = 0  # 一些 padding
    
    ls_loss = LabelSmoothingLoss(vocab_size, padding_idx=0, smoothing=0.1)
    loss = ls_loss(logits, targets)
    print(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # 测试学习率调度器
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=100, total_steps=1000)
    
    lrs = []
    for i in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    print(f"LR at step 0: {lrs[0]:.6f}")
    print(f"LR at step 100: {lrs[100]:.6f}")
    print(f"LR at step 500: {lrs[500]:.6f}")
    print(f"LR at step 999: {lrs[999]:.6f}")
    
    # 测试 EMA
    ema = ExponentialMovingAverage(model, decay=0.999)
    print("\nEMA 初始化成功")
    
    print("\n测试完成！")
