"""
Vision Transformer + Transformer Decoder 图像描述模型
使用ViT作为图像编码器，Transformer Decoder作为文本解码器
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer 图像编码器
    使用预训练的ViT模型提取图像特征
    """
    
    def __init__(self, model_name='vit_b_16', pretrained=True, d_model=512):
        super(VisionTransformerEncoder, self).__init__()
        
        # 加载预训练的ViT模型
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.vit = vit_b_16(weights=weights)
        else:
            self.vit = vit_b_16()
        
        # ViT输出维度是768 (对于vit_b_16)
        vit_dim = 768
        
        # 移除ViT的分类头，只保留特征提取部分
        self.vit.heads = nn.Identity()
        
        # 投影层：将ViT特征映射到Transformer的d_model维度
        self.projection = nn.Linear(vit_dim, d_model)
        
        # 是否微调ViT
        self.set_trainable(False)  # 默认冻结ViT参数
    
    def set_trainable(self, trainable=True):
        """设置ViT是否可训练"""
        for param in self.vit.parameters():
            param.requires_grad = trainable
    
    def forward(self, images):
        """
        参数:
            images: (batch_size, 3, 224, 224)
        返回:
            features: (batch_size, num_patches, d_model)
        """
        # 获取ViT的特征
        # ViT内部会将图像分成patches并添加位置编码
        x = images
        
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]
        
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 通过ViT的encoder
        x = self.vit.encoder(x)
        
        # 移除class token，只保留patch tokens
        # x shape: (batch_size, num_patches+1, 768)
        patch_features = x[:, 1:, :]  # (batch_size, num_patches, 768)
        
        # 投影到d_model维度
        features = self.projection(patch_features)  # (batch_size, num_patches, d_model)
        
        return features


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器
    用于生成图像描述文本
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=100):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz, device):
        """生成因果mask（上三角mask）"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        参数:
            tgt: 目标序列 (batch_size, tgt_len)
            memory: 编码器输出 (batch_size, src_len, d_model)
            tgt_mask: 目标序列的mask (tgt_len, tgt_len)
            tgt_key_padding_mask: 目标序列的padding mask (batch_size, tgt_len)
        返回:
            output: (batch_size, tgt_len, vocab_size)
        """
        # 词嵌入
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # 位置编码
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Transformer解码
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 输出层
        output = self.fc_out(output)
        
        return output


class ViTTransformerCaptioning(nn.Module):
    """
    完整的图像描述模型：ViT编码器 + Transformer解码器
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_len=100, pretrained_vit=True):
        super(ViTTransformerCaptioning, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 图像编码器 (ViT)
        self.encoder = VisionTransformerEncoder(
            model_name='vit_b_16',
            pretrained=pretrained_vit,
            d_model=d_model
        )
        
        # 文本解码器 (Transformer)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )
    
    def forward(self, images, captions, caption_lengths=None):
        """
        训练时的前向传播
        
        参数:
            images: (batch_size, 3, 224, 224)
            captions: (batch_size, max_len) - 包含<start>和<end>的完整序列
            caption_lengths: (batch_size,) - 每个caption的实际长度
        返回:
            output: (batch_size, max_len-1, vocab_size) - 预测的词分布
        """
        batch_size = images.size(0)
        device = images.device
        
        # 1. 图像编码
        memory = self.encoder(images)  # (batch_size, num_patches, d_model)
        
        # 2. 准备解码器输入（不包括最后一个词）
        tgt = captions[:, :-1]  # (batch_size, max_len-1)
        tgt_len = tgt.size(1)
        
        # 3. 创建mask
        # 因果mask（防止看到未来的词）
        tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_len, device)
        
        # Padding mask（标记哪些位置是padding）
        if caption_lengths is not None:
            tgt_key_padding_mask = self._generate_padding_mask(tgt, caption_lengths)
        else:
            tgt_key_padding_mask = None
        
        # 4. 解码
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return output
    
    def _generate_padding_mask(self, tgt, lengths):
        """
        生成padding mask
        
        参数:
            tgt: (batch_size, seq_len)
            lengths: (batch_size,) - 实际长度
        返回:
            mask: (batch_size, seq_len) - True表示padding位置
        """
        batch_size, seq_len = tgt.size()
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tgt.device)
        
        for i, length in enumerate(lengths):
            if length < seq_len:
                mask[i, length:] = True
        
        return mask
    
    def generate(self, images, start_token, end_token, max_len=50, method='greedy'):
        """
        生成图像描述（推理时使用）
        
        参数:
            images: (batch_size, 3, 224, 224)
            start_token: 开始标记的索引
            end_token: 结束标记的索引
            max_len: 最大生成长度
            method: 'greedy' 或 'beam_search'
        返回:
            captions: (batch_size, seq_len) - 生成的caption序列
        """
        if method == 'greedy':
            return self._greedy_search(images, start_token, end_token, max_len)
        elif method == 'beam_search':
            return self._beam_search(images, start_token, end_token, max_len, beam_size=5)
        else:
            raise ValueError(f"Unknown generation method: {method}")
    
    def _greedy_search(self, images, start_token, end_token, max_len):
        """贪婪搜索生成"""
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        with torch.no_grad():
            # 编码图像
            memory = self.encoder(images)
            
            # 初始化生成序列（从start_token开始）
            generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
            
            for _ in range(max_len - 1):
                # 解码
                tgt_mask = self.decoder.generate_square_subsequent_mask(generated.size(1), device)
                output = self.decoder(generated, memory, tgt_mask=tgt_mask)
                
                # 获取最后一个时间步的预测
                next_word_logits = output[:, -1, :]  # (batch_size, vocab_size)
                next_word = next_word_logits.argmax(dim=-1)  # (batch_size,)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_word.unsqueeze(1)], dim=1)
                
                # 如果所有序列都生成了end_token，则停止
                if (next_word == end_token).all():
                    break
        
        return generated
    
    def _beam_search(self, images, start_token, end_token, max_len, beam_size=5):
        """束搜索生成"""
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        if batch_size != 1:
            # 为简化实现，束搜索一次只处理一张图片
            results = []
            for i in range(batch_size):
                result = self._beam_search(
                    images[i:i+1], start_token, end_token, max_len, beam_size
                )
                results.append(result)
            return torch.cat(results, dim=0)
        
        with torch.no_grad():
            # 编码图像
            memory = self.encoder(images)  # (1, num_patches, d_model)
            
            # 初始化beam
            # sequences: (beam_size, seq_len)
            sequences = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
            # scores: (beam_size,)
            scores = torch.zeros(beam_size, device=device)
            
            # 扩展memory以匹配beam_size
            memory = memory.expand(beam_size, -1, -1)
            
            # 完成的序列
            completed_sequences = []
            completed_scores = []
            
            for step in range(max_len - 1):
                # 解码当前序列
                tgt_mask = self.decoder.generate_square_subsequent_mask(sequences.size(1), device)
                output = self.decoder(sequences, memory, tgt_mask=tgt_mask)
                
                # 获取最后一个时间步的预测
                next_word_logits = output[:, -1, :]  # (beam_size, vocab_size)
                next_word_log_probs = torch.log_softmax(next_word_logits, dim=-1)
                
                # 计算所有可能的下一步得分
                if step == 0:
                    # 第一步，所有beam都是相同的
                    scores = next_word_log_probs[0]  # (vocab_size,)
                    top_scores, top_words = scores.topk(beam_size)
                    sequences = torch.cat([
                        sequences[0:1].expand(beam_size, -1),
                        top_words.unsqueeze(1)
                    ], dim=1)
                    scores = top_scores
                else:
                    # 计算所有候选的得分
                    candidate_scores = scores.unsqueeze(1) + next_word_log_probs  # (beam_size, vocab_size)
                    candidate_scores = candidate_scores.view(-1)  # (beam_size * vocab_size,)
                    
                    # 选择top-k
                    top_scores, top_indices = candidate_scores.topk(beam_size)
                    
                    # 确定哪个beam和哪个word
                    beam_indices = top_indices // self.vocab_size
                    word_indices = top_indices % self.vocab_size
                    
                    # 更新sequences
                    sequences = torch.cat([
                        sequences[beam_indices],
                        word_indices.unsqueeze(1)
                    ], dim=1)
                    scores = top_scores
                
                # 检查是否有序列完成
                end_mask = (sequences[:, -1] == end_token)
                if end_mask.any():
                    for idx in end_mask.nonzero(as_tuple=True)[0]:
                        completed_sequences.append(sequences[idx])
                        completed_scores.append(scores[idx])
                    
                    # 移除已完成的序列
                    if end_mask.all():
                        break
                    
                    keep_mask = ~end_mask
                    sequences = sequences[keep_mask]
                    scores = scores[keep_mask]
                    memory = memory[keep_mask]
                    beam_size = sequences.size(0)
            
            # 选择得分最高的序列
            if len(completed_sequences) > 0:
                best_idx = torch.tensor(completed_scores).argmax()
                return completed_sequences[best_idx].unsqueeze(0)
            else:
                return sequences[scores.argmax()].unsqueeze(0)


def build_model(vocab_size, config):
    """
    构建模型的辅助函数
    
    参数:
        vocab_size: 词典大小
        config: 配置字典，包含模型超参数
    返回:
        model: ViTTransformerCaptioning模型
    """
    model = ViTTransformerCaptioning(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        max_len=config.get('max_len', 100),
        pretrained_vit=config.get('pretrained_vit', True)
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试 ViT + Transformer 图像描述模型")
    print("=" * 60)
    
    # 配置
    vocab_size = 109
    batch_size = 4
    
    config = {
        'd_model': 512,
        'nhead': 8,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_len': 52,
        'pretrained_vit': False  # 测试时不加载预训练权重
    }
    
    # 构建模型
    model = build_model(vocab_size, config)
    print(f"模型构建成功！")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 测试前向传播
    print("\n测试前向传播...")
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, 20))
    caption_lengths = torch.tensor([20, 18, 15, 17])
    
    output = model(images, captions, caption_lengths)
    print(f"输入图像形状: {images.shape}")
    print(f"输入caption形状: {captions.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试生成
    print("\n测试生成...")
    model.eval()
    generated = model.generate(images, start_token=107, end_token=108, max_len=20, method='greedy')
    print(f"生成的caption形状: {generated.shape}")
    print(f"生成的caption: {generated[0].tolist()}")
    
    print("\n测试完成！")
