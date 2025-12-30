"""
Self-Critical Sequence Training (SCST) 强化学习损失函数

参考论文: Self-critical Sequence Training for Image Captioning (Rennie et al., 2017)

核心思想:
- 使用 REINFORCE 算法训练
- Baseline 使用 greedy 解码的结果
- Reward 使用评测指标 (CIDEr, BLEU 等)
- 损失 = -E[reward - baseline] * log_prob
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu


class CiderRewardCalculator:
    """
    计算 CIDEr 奖励的类
    支持快速批量计算
    """
    
    def __init__(self):
        self.cider_scorer = Cider()
    
    def compute_reward(self, predictions, references):
        """
        计算 CIDEr 奖励
        
        参数:
            predictions: list of str, 生成的句子列表
            references: list of list of str, 参考句子列表
            
        返回:
            rewards: list of float, 每个预测的奖励
        """
        # 构建 gts 和 res 字典
        gts = {}
        res = {}
        
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            gts[i] = refs if isinstance(refs, list) else [refs]
            res[i] = [pred]
        
        # 计算 CIDEr
        try:
            avg_score, per_image_scores = self.cider_scorer.compute_score(gts, res)
            return list(per_image_scores)
        except Exception as e:
            print(f"CIDEr 计算出错: {e}")
            return [0.0] * len(predictions)


class BleuRewardCalculator:
    """
    计算 BLEU-4 奖励的类
    """
    
    def __init__(self):
        self.bleu_scorer = Bleu(4)  # BLEU-4
    
    def compute_reward(self, predictions, references):
        """
        计算 BLEU-4 奖励
        """
        gts = {}
        res = {}
        
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            gts[i] = refs if isinstance(refs, list) else [refs]
            res[i] = [pred]
        
        try:
            # BLEU 返回 (bleu1, bleu2, bleu3, bleu4) 的分数
            scores, per_image_scores = self.bleu_scorer.compute_score(gts, res)
            # 使用 BLEU-4
            return list(per_image_scores[3])  # 索引 3 是 BLEU-4
        except Exception as e:
            print(f"BLEU 计算出错: {e}")
            return [0.0] * len(predictions)


class CombinedRewardCalculator:
    """
    组合多个指标的奖励
    """
    
    def __init__(self, cider_weight=1.0, bleu_weight=0.0):
        self.cider_weight = cider_weight
        self.bleu_weight = bleu_weight
        
        if cider_weight > 0:
            self.cider_calc = CiderRewardCalculator()
        if bleu_weight > 0:
            self.bleu_calc = BleuRewardCalculator()
    
    def compute_reward(self, predictions, references):
        """
        计算组合奖励
        """
        rewards = [0.0] * len(predictions)
        
        if self.cider_weight > 0:
            cider_rewards = self.cider_calc.compute_reward(predictions, references)
            rewards = [r + self.cider_weight * c for r, c in zip(rewards, cider_rewards)]
        
        if self.bleu_weight > 0:
            bleu_rewards = self.bleu_calc.compute_reward(predictions, references)
            rewards = [r + self.bleu_weight * b for r, b in zip(rewards, bleu_rewards)]
        
        return rewards


class SCSTLoss(nn.Module):
    """
    Self-Critical Sequence Training 损失函数
    
    使用 REINFORCE 算法，以 greedy 解码结果作为 baseline
    """
    
    def __init__(self, reward_type='cider', cider_weight=1.0, bleu_weight=0.0):
        super(SCSTLoss, self).__init__()
        
        if reward_type == 'cider':
            self.reward_calculator = CiderRewardCalculator()
        elif reward_type == 'bleu':
            self.reward_calculator = BleuRewardCalculator()
        elif reward_type == 'combined':
            self.reward_calculator = CombinedRewardCalculator(cider_weight, bleu_weight)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
    
    def forward(self, model, images, references, vocab, device, 
                sample_method='sample', max_len=50):
        """
        计算 SCST 损失
        
        参数:
            model: 图像描述模型
            images: 图像张量 (batch_size, 3, H, W)
            references: 参考描述列表 (batch_size, [ref1, ref2, ...])
            vocab: 词典
            device: 设备
            sample_method: 采样方法 ('sample' 或 'beam_sample')
            max_len: 最大生成长度
            
        返回:
            loss: SCST 损失
            reward_info: 奖励信息字典
        """
        batch_size = images.size(0)
        start_token = vocab['<start>']
        end_token = vocab['<end>']
        pad_token = vocab['<pad>']
        
        idx2word = {idx: word for word, idx in vocab.items()}
        
        model.train()
        
        # 1. 采样生成 (sample)
        sample_ids, sample_log_probs = self._sample_with_log_probs(
            model, images, start_token, end_token, max_len, device
        )
        
        # 2. Greedy 生成 (baseline)
        with torch.no_grad():
            greedy_ids = model.generate(
                images, start_token, end_token, max_len, method='greedy'
            )
        
        # 3. 解码为文本
        sample_captions = self._decode_captions(sample_ids, idx2word, end_token, pad_token, start_token)
        greedy_captions = self._decode_captions(greedy_ids, idx2word, end_token, pad_token, start_token)
        
        # 4. 计算奖励
        sample_rewards = self.reward_calculator.compute_reward(sample_captions, references)
        greedy_rewards = self.reward_calculator.compute_reward(greedy_captions, references)
        
        # 5. 计算优势 (advantage = reward - baseline)
        sample_rewards = torch.tensor(sample_rewards, device=device, dtype=torch.float)
        greedy_rewards = torch.tensor(greedy_rewards, device=device, dtype=torch.float)
        advantages = sample_rewards - greedy_rewards
        
        # 6. 计算 REINFORCE 损失
        # loss = -advantage * sum(log_probs)
        # 对每个序列求和 log_probs
        seq_log_probs = sample_log_probs.sum(dim=1)  # (batch_size,)
        loss = -(advantages * seq_log_probs).mean()
        
        # 收集奖励信息
        reward_info = {
            'sample_reward': sample_rewards.mean().item(),
            'greedy_reward': greedy_rewards.mean().item(),
            'advantage': advantages.mean().item(),
        }
        
        return loss, reward_info
    
    def _sample_with_log_probs(self, model, images, start_token, end_token, max_len, device):
        """
        采样生成序列并记录 log probabilities
        
        返回:
            sample_ids: (batch_size, seq_len) 采样的 token IDs
            log_probs: (batch_size, seq_len) 每个 token 的 log probability
        """
        batch_size = images.size(0)
        
        # 编码图像
        memory = model.encoder(images)
        
        # 初始化
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        log_probs_list = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            # 解码
            tgt_mask = model.decoder.generate_square_subsequent_mask(generated.size(1), device)
            output = model.decoder(generated, memory, tgt_mask=tgt_mask)
            
            # 获取最后一个时间步的 logits
            logits = output[:, -1, :]  # (batch_size, vocab_size)
            
            # 计算概率分布
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            
            # 采样
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch_size,)
            
            # 获取采样 token 的 log prob
            token_log_prob = log_prob.gather(1, next_token.unsqueeze(1)).squeeze(1)  # (batch_size,)
            
            # 对已完成的序列，log_prob 设为 0
            token_log_prob = token_log_prob.masked_fill(finished, 0.0)
            log_probs_list.append(token_log_prob)
            
            # 更新生成序列
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            # 检查是否完成
            finished = finished | (next_token == end_token)
            
            if finished.all():
                break
        
        # Stack log probs
        log_probs = torch.stack(log_probs_list, dim=1)  # (batch_size, seq_len)
        
        return generated, log_probs
    
    def _decode_captions(self, caption_ids, idx2word, end_token, pad_token, start_token):
        """
        将 token IDs 解码为文本
        """
        captions = []
        for ids in caption_ids:
            words = []
            for idx in ids.cpu().tolist():
                if idx == end_token:
                    break
                if idx not in [start_token, pad_token]:
                    words.append(idx2word.get(idx, '<unk>'))
            captions.append(' '.join(words))
        return captions


class MixedLoss(nn.Module):
    """
    混合损失：交叉熵 + SCST
    
    在训练初期使用交叉熵预热，后期逐渐过渡到 SCST
    """
    
    def __init__(self, vocab_size, pad_idx, reward_type='cider', 
                 xe_weight=1.0, rl_weight=1.0):
        super(MixedLoss, self).__init__()
        
        self.xe_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.scst_loss = SCSTLoss(reward_type=reward_type)
        self.xe_weight = xe_weight
        self.rl_weight = rl_weight
    
    def forward(self, model, images, captions, caption_lengths, references, 
                vocab, device, use_rl=True, max_len=50):
        """
        计算混合损失
        
        参数:
            model: 模型
            images: 图像
            captions: 目标 caption (用于 XE loss)
            caption_lengths: caption 长度
            references: 参考描述 (用于 RL loss)
            vocab: 词典
            device: 设备
            use_rl: 是否使用 RL 损失
            max_len: 最大生成长度
        """
        # 交叉熵损失
        outputs = model(images, captions, caption_lengths)
        targets = captions[:, 1:]  # 去掉 <start>
        
        # Reshape for cross entropy
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        xe_loss = self.xe_criterion(outputs_flat, targets_flat)
        
        total_loss = self.xe_weight * xe_loss
        reward_info = {'xe_loss': xe_loss.item()}
        
        # SCST 损失 (可选)
        if use_rl and self.rl_weight > 0:
            rl_loss, rl_info = self.scst_loss(
                model, images, references, vocab, device, max_len=max_len
            )
            total_loss = total_loss + self.rl_weight * rl_loss
            reward_info.update(rl_info)
            reward_info['rl_loss'] = rl_loss.item()
        
        return total_loss, reward_info


def get_reference_captions(caption_ids, vocab):
    """
    从 caption IDs 获取参考描述文本
    
    参数:
        caption_ids: (batch_size, seq_len) caption token IDs
        vocab: 词典
        
    返回:
        references: list of list of str
    """
    idx2word = {idx: word for word, idx in vocab.items()}
    start_token = vocab['<start>']
    end_token = vocab['<end>']
    pad_token = vocab['<pad>']
    
    references = []
    for ids in caption_ids:
        words = []
        for idx in ids.tolist():
            if idx == end_token:
                break
            if idx not in [start_token, pad_token]:
                words.append(idx2word.get(idx, '<unk>'))
        references.append([' '.join(words)])  # 包装成列表
    
    return references


if __name__ == '__main__':
    # 简单测试
    print("测试 SCST 损失模块")
    print("=" * 60)
    
    # 测试奖励计算
    calc = CiderRewardCalculator()
    
    predictions = ["a red dress", "blue jeans with white shirt"]
    references = [["a beautiful red dress", "red dress"], ["blue jeans"]]
    
    rewards = calc.compute_reward(predictions, references)
    print(f"CIDEr 奖励: {rewards}")
    
    print("\n测试完成！")
