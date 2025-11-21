"""
DeepFashion-MultiModal数据集整理脚本（不依赖PyTorch）
仅用于生成词典和数据集JSON文件
"""

import os
import json
import random
from collections import Counter


def create_deepfashion_dataset(data_dir='data', 
                                min_word_count=5, 
                                max_len=50,
                                train_ratio=0.8,
                                val_ratio=0.1,
                                test_ratio=0.1):
    """
    整理DeepFashion-MultiModal数据集
    """
    
    captions_path = os.path.join(data_dir, 'captions.json')
    image_folder = os.path.join(data_dir, 'images')
    output_folder = data_dir
    
    print("正在加载captions.json...")
    with open(captions_path, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    print(f"总共有 {len(captions_data)} 条图像-文本对")
    
    # 收集所有图片路径和对应的caption
    all_data = []
    vocab = Counter()
    missing_images = 0
    
    for img_filename, caption_text in captions_data.items():
        img_path = os.path.join(image_folder, img_filename)
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            missing_images += 1
            continue
        
        # 简单的分词处理：转小写，按空格和标点分割
        tokens = caption_text.lower().replace('.', ' .').replace(',', ' ,').split()
        
        # 过滤超长文本
        if len(tokens) <= max_len:
            all_data.append({
                'image_path': img_path,
                'tokens': tokens,
                'raw_caption': caption_text
            })
            vocab.update(tokens)
    
    if missing_images > 0:
        print(f"警告: {missing_images} 张图片不存在")
    
    print(f"有效数据: {len(all_data)} 条")
    print(f"原始词汇量: {len(vocab)}")
    
    # 创建词典
    words = [w for w in vocab.keys() if vocab[w] >= min_word_count]
    vocab_dict = {k: v + 1 for v, k in enumerate(words)}
    vocab_dict['<pad>'] = 0
    vocab_dict['<unk>'] = len(vocab_dict)
    vocab_dict['<start>'] = len(vocab_dict)
    vocab_dict['<end>'] = len(vocab_dict)
    
    print(f"过滤后词汇量: {len(vocab_dict)}")
    
    # 保存词典
    vocab_path = os.path.join(output_folder, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"词典已保存到: {vocab_path}")
    
    # 随机打乱数据
    random.seed(42)  # 设置随机种子以保证可复现
    random.shuffle(all_data)
    
    # 划分数据集
    total_num = len(all_data)
    train_num = int(total_num * train_ratio)
    val_num = int(total_num * val_ratio)
    
    train_data = all_data[:train_num]
    val_data = all_data[train_num:train_num + val_num]
    test_data = all_data[train_num + val_num:]
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_data)} 条 ({len(train_data)/total_num*100:.1f}%)")
    print(f"验证集: {len(val_data)} 条 ({len(val_data)/total_num*100:.1f}%)")
    print(f"测试集: {len(test_data)} 条 ({len(test_data)/total_num*100:.1f}%)")
    
    # 编码并保存各个数据集
    for split_name, split_data in [('train', train_data), 
                                     ('val', val_data), 
                                     ('test', test_data)]:
        
        image_paths = []
        encoded_captions = []
        
        for item in split_data:
            # 编码caption
            enc_caption = [vocab_dict['<start>']] + \
                         [vocab_dict.get(word, vocab_dict['<unk>']) for word in item['tokens']] + \
                         [vocab_dict['<end>']]
            
            image_paths.append(item['image_path'])
            encoded_captions.append(enc_caption)
        
        # 保存数据
        data_dict = {
            'IMAGES': image_paths,
            'CAPTIONS': encoded_captions
        }
        
        output_path = os.path.join(output_folder, f'{split_name}_data.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False)
        print(f"{split_name}数据已保存到: {output_path}")
    
    print("\n数据集整理完成！")
    
    # 显示一些统计信息
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    
    for split_name, split_data in [('train', train_data), 
                                     ('val', val_data), 
                                     ('test', test_data)]:
        caption_lengths = [len(item['tokens']) + 2 for item in split_data]  # +2 for <start> and <end>
        print(f"\n{split_name.upper()}集:")
        print(f"  样本数: {len(split_data)}")
        print(f"  Caption平均长度: {sum(caption_lengths)/len(caption_lengths):.2f}")
        print(f"  Caption最大长度: {max(caption_lengths)}")
        print(f"  Caption最小长度: {min(caption_lengths)}")
    
    return vocab_dict


if __name__ == '__main__':
    print("=" * 60)
    print("DeepFashion-MultiModal数据集整理工具")
    print("=" * 60)
    print()
    
    vocab = create_deepfashion_dataset(
        data_dir='data',
        min_word_count=5,
        max_len=50,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - data/vocab.json (词典)")
    print("  - data/train_data.json (训练集)")
    print("  - data/val_data.json (验证集)")
    print("  - data/test_data.json (测试集)")
    print("\n下一步:")
    print("  1. 检查生成的文件")
    print("  2. 运行 test_deepfashion.py 进行测试（需要安装PyTorch）")
    print("  3. 开始训练模型")
