"""
模型二：网格特征 (CNN) + Transformer 编码器 + Transformer 解码器
"""

import torch
import torch.nn as nn
import math
from torchvision.models import resnet101, ResNet101_Weights


class PositionalEncoding(nn.Module):
    """位置编码，与vit_transformer_model.py中的相同"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GridFeatureEncoder(nn.Module):
    """
    图像编码器：使用CNN提取网格特征，然后用Transformer编码器增强
    """
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, pretrained_cnn=True):
        super(GridFeatureEncoder, self).__init__()
        
        # 1. CNN特征提取器 (使用ResNet-101)
        if pretrained_cnn:
            weights = ResNet101_Weights.DEFAULT
            self.cnn = resnet101(weights=weights)
        else:
            self.cnn = resnet101()
            
        # 移除ResNet的平均池化层和全连接层，保留卷积特征图
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        
        # ResNet-101的输出通道数为2048
        cnn_out_dim = 2048
        
        # 2. 投影层，将CNN特征维度降到d_model
        self.projection = nn.Conv2d(cnn_out_dim, d_model, kernel_size=1)
        
        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 4. 位置编码
        # 假设输入图像224x224，ResNet输出特征图7x7，共49个位置
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=100) 
        
        self.d_model = d_model
        
        # 默认冻结CNN参数
        self.set_cnn_trainable(False)

    def set_cnn_trainable(self, trainable=True):
        """设置CNN部分是否可训练"""
        for param in self.cnn.parameters():
            param.requires_grad = trainable

    def forward(self, images):
        """
        参数:
            images: (batch_size, 3, H, W), e.g., (batch_size, 3, 224, 224)
        返回:
            features: (batch_size, num_patches, d_model), e.g., (batch_size, 49, 512)
        """
        # 1. 经过CNN提取特征
        x = self.cnn(images)  # (batch_size, 2048, H/32, W/32), e.g., (batch_size, 2048, 7, 7)
        
        # 2. 投影到d_model维度
        x = self.projection(x) # (batch_size, d_model, 7, 7)
        
        # 3. 展平并调整维度顺序以适应Transformer
        batch_size, _, h, w = x.shape
        x = x.flatten(2)  # (batch_size, d_model, 49)
        x = x.permute(0, 2, 1) # (batch_size, 49, d_model)
        
        # 4. 添加位置编码
        x = self.pos_encoder(x)
        
        # 5. 经过Transformer编码器
        features = self.transformer_encoder(x) # (batch_size, 49, d_model)
        
        return features


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器，与vit_transformer_model.py中的相同
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=100):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        output = self.fc_out(output)
        return output


class GridTransformerCaptioning(nn.Module):
    """
    完整的图像描述模型：CNN网格特征 + Transformer编码器 + Transformer解码器
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 max_len=100, pretrained_cnn=True):
        super(GridTransformerCaptioning, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.encoder = GridFeatureEncoder(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, pretrained_cnn=pretrained_cnn
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead,
            num_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, max_len=max_len
        )
    
    def forward(self, images, captions, caption_lengths=None):
        batch_size = images.size(0)
        device = images.device
        
        memory = self.encoder(images)
        
        tgt = captions[:, :-1]
        tgt_len = tgt.size(1)
        
        tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_len, device)
        
        if caption_lengths is not None:
            # 注意：这里的padding mask需要根据实际长度调整
            # lengths 应该包含<start>和<end>
            # 我们输入解码器的是去掉<end>的序列，所以长度要减1
            adjusted_lengths = [l - 1 for l in caption_lengths]
            tgt_key_padding_mask = self._generate_padding_mask(tgt, adjusted_lengths)
        else:
            tgt_key_padding_mask = None
            
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return output
    
    def _generate_padding_mask(self, tgt, lengths):
        batch_size, seq_len = tgt.size()
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tgt.device)
        
        for i, length in enumerate(lengths):
            if length < seq_len:
                mask[i, length:] = True
        
        return mask

    def generate(self, images, start_token, end_token, max_len=50, method='greedy', beam_size=5):
        if method == 'greedy':
            return self._greedy_search(images, start_token, end_token, max_len)
        elif method == 'beam_search':
            return self._beam_search(images, start_token, end_token, max_len, beam_size)
        else:
            raise ValueError(f"Unknown generation method: {method}")

    def _greedy_search(self, images, start_token, end_token, max_len):
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        with torch.no_grad():
            memory = self.encoder(images)
            generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
            
            for _ in range(max_len - 1):
                tgt_mask = self.decoder.generate_square_subsequent_mask(generated.size(1), device)
                output = self.decoder(generated, memory, tgt_mask=tgt_mask)
                
                next_word_logits = output[:, -1, :]
                next_word = next_word_logits.argmax(dim=-1)
                
                generated = torch.cat([generated, next_word.unsqueeze(1)], dim=1)
                
                if (next_word == end_token).all():
                    break
        
        return generated

    def _beam_search(self, images, start_token, end_token, max_len, beam_size=5):
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        if batch_size != 1:
            results = []
            for i in range(batch_size):
                result = self._beam_search(images[i:i+1], start_token, end_token, max_len, beam_size)
                results.append(result)
            return torch.cat(results, dim=0)
        
        with torch.no_grad():
            memory = self.encoder(images)
            
            sequences = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
            scores = torch.zeros(beam_size, device=device)
            
            memory = memory.expand(beam_size, -1, -1)
            
            completed_sequences = []
            completed_scores = []
            
            for step in range(max_len - 1):
                if sequences.size(0) == 0: break
                
                tgt_mask = self.decoder.generate_square_subsequent_mask(sequences.size(1), device)
                output = self.decoder(sequences, memory, tgt_mask=tgt_mask)
                
                next_word_logits = output[:, -1, :]
                next_word_log_probs = torch.log_softmax(next_word_logits, dim=-1)
                
                if step == 0:
                    scores = next_word_log_probs[0]
                    top_scores, top_words = scores.topk(beam_size)
                    sequences = torch.cat([sequences[0:1].expand(beam_size, -1), top_words.unsqueeze(1)], dim=1)
                    scores = top_scores
                else:
                    candidate_scores = scores.unsqueeze(1) + next_word_log_probs
                    candidate_scores = candidate_scores.view(-1)
                    
                    top_scores, top_indices = candidate_scores.topk(beam_size)
                    
                    beam_indices = top_indices // self.vocab_size
                    word_indices = top_indices % self.vocab_size
                    
                    sequences = torch.cat([sequences[beam_indices], word_indices.unsqueeze(1)], dim=1)
                    scores = top_scores
                
                end_mask = (sequences[:, -1] == end_token)
                if end_mask.any():
                    for idx in end_mask.nonzero(as_tuple=True)[0]:
                        completed_sequences.append(sequences[idx])
                        completed_scores.append(scores[idx])
                    
                    if len(completed_sequences) >= beam_size: break
                    
                    keep_mask = ~end_mask
                    sequences = sequences[keep_mask]
                    scores = scores[keep_mask]
                    memory = memory[keep_mask]
                    
                    if sequences.size(0) == 0: break
                    beam_size = sequences.size(0)

            if len(completed_sequences) > 0:
                best_idx = torch.tensor(completed_scores).argmax()
                return completed_sequences[best_idx].unsqueeze(0)
            else:
                return sequences[scores.argmax()].unsqueeze(0)


def build_model(vocab_size, config):
    """构建模型的辅助函数"""
    model = GridTransformerCaptioning(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        max_len=config.get('max_len', 100),
        pretrained_cnn=config.get('pretrained_cnn', True)
    )
    return model


if __name__ == '__main__':
    print("测试 Grid-Transformer 图像描述模型")
    print("=" * 60)
    
    vocab_size = 109
    batch_size = 4
    
    config = {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 3, # 使用较少的层进行快速测试
        'num_decoder_layers': 3,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_len': 52,
        'pretrained_cnn': False
    }
    
    model = build_model(vocab_size, config)
    print(f"模型构建成功！")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.encoder.set_cnn_trainable(True)
    print(f"可训练参数量 (CNN解冻后): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    print("\n测试前向传播...")
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, 20))
    caption_lengths = torch.tensor([20, 18, 15, 17])
    
    output = model(images, captions, caption_lengths)
    print(f"输入图像形状: {images.shape}")
    print(f"输入caption形状: {captions.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n测试生成 (Greedy)...")
    generated = model.generate(images, start_token=107, end_token=108, max_len=20, method='greedy')
    print(f"生成的caption形状: {generated.shape}")
    print(f"生成的caption: {generated[0].tolist()}")

    print("\n测试生成 (Beam Search)...")
    generated_beam = model.generate(images[0:1], start_token=107, end_token=108, max_len=20, method='beam_search', beam_size=3)
    print(f"生成的caption形状 (beam): {generated_beam.shape}")
    print(f"生成的caption (beam): {generated_beam[0].tolist()}")
    
    print("\n测试完成！")
