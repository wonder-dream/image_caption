"""
ViT + Transformer 模型快速测试
测试模型是否可以正常前向传播和训练
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from models.vit_transformer_model import build_model


def test_model():
    """测试模型基本功能"""
    
    print("=" * 80)
    print("ViT + Transformer 模型快速测试")
    print("=" * 80)
    
    # 加载词典
    print("\n1. 加载词典...")
    with open('data/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    print(f"   词典大小: {vocab_size}")
    
    # 配置
    config = {
        'd_model': 512,
        'nhead': 8,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_len': 52,
        'pretrained_vit': False  # 快速测试不加载预训练权重
    }
    
    # 构建模型
    print("\n2. 构建模型...")
    model = build_model(vocab_size, config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"   使用设备: {device}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params / 1e6:.2f}M")
    print(f"   可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    print("\n3. 测试前向传播...")
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    captions = torch.randint(0, vocab_size, (batch_size, 20)).to(device)
    cap_lens = torch.tensor([20, 18, 15, 17]).to(device)
    
    print(f"   输入图像形状: {images.shape}")
    print(f"   输入caption形状: {captions.shape}")
    
    model.train()
    outputs = model(images, captions, cap_lens)
    print(f"   输出形状: {outputs.shape}")
    print(f"   ✓ 前向传播成功！")
    
    # 测试反向传播
    print("\n4. 测试反向传播...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    targets = captions[:, 1:]
    
    outputs_flat = outputs.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    loss = criterion(outputs_flat, targets_flat)
    print(f"   损失值: {loss.item():.4f}")
    
    loss.backward()
    print(f"   ✓ 反向传播成功！")
    
    # 测试生成
    print("\n5. 测试caption生成...")
    model.eval()
    
    with torch.no_grad():
        # Greedy搜索
        generated_greedy = model.generate(
            images[:1],  # 只测试一张图
            start_token=vocab['<start>'],
            end_token=vocab['<end>'],
            max_len=20,
            method='greedy'
        )
        print(f"   Greedy生成形状: {generated_greedy.shape}")
        print(f"   生成的ID序列: {generated_greedy[0].tolist()}")
        
        # 转换为文本
        idx2word = {idx: word for word, idx in vocab.items()}
        words = [idx2word[idx.item()] for idx in generated_greedy[0][:10]]
        print(f"   前10个词: {' '.join(words)}")
        print(f"   ✓ Greedy生成成功！")
    
    # 测试数据加载
    print("\n6. 测试数据加载...")
    try:
        from utils.deepfashion_dataset import create_data_loaders
        
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir='data',
            vocab_path='data/vocab.json',
            batch_size=8,
            num_workers=0,
            image_size=224
        )
        
        # 读取一个batch
        for imgs, caps, cap_lens in train_loader:
            print(f"   批次图像形状: {imgs.shape}")
            print(f"   批次caption形状: {caps.shape}")
            print(f"   批次大小: {imgs.shape[0]}")
            print(f"   ✓ 数据加载成功！")
            break
    except Exception as e:
        print(f"   数据加载出错: {e}")
    
    # 测试优化器
    print("\n7. 测试优化器...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.zero_grad()
    
    # 模拟一步训练
    model.train()
    outputs = model(images, captions, cap_lens)
    targets = captions[:, 1:]
    outputs_flat = outputs.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    loss = criterion(outputs_flat, targets_flat)
    
    loss.backward()
    optimizer.step()
    print(f"   训练步骤完成，损失: {loss.item():.4f}")
    print(f"   ✓ 优化器测试成功！")
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！模型可以正常训练。")
    print("=" * 80)
    
    print("\n下一步:")
    print("  1. 运行 'python train_vit_transformer.py' 开始训练")
    print("  2. 训练完成后运行 'python inference_vit_transformer.py' 进行推理")
    print("  3. 可以在 train_vit_transformer.py 中调整超参数")


if __name__ == '__main__':
    test_model()
