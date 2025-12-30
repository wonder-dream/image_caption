"""
DeepFashion-MultiModal数据集处理模块
包含数据集整理、数据集类定义和批量读取功能
"""

import os
import json
import random
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def create_deepfashion_dataset(data_dir='data', 
                                min_word_count=5, 
                                max_len=50,
                                train_ratio=0.8,
                                val_ratio=0.1,
                                test_ratio=0.1):
    """
    整理DeepFashion-MultiModal数据集
    
    参数：
        data_dir：数据集目录
        min_word_count：仅考虑出现次数≥该值的词
        max_len：文本描述包含的最大单词数，超过则截断
        train_ratio：训练集比例
        val_ratio：验证集比例
        test_ratio：测试集比例
    
    输出：
        vocab.json：词典文件
        train_data.json、val_data.json、test_data.json：数据集文件
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
    
    for img_filename, caption_text in captions_data.items():
        img_path = os.path.join(image_folder, img_filename)
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在 {img_path}")
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
    random.shuffle(all_data)
    
    # 划分数据集
    total_num = len(all_data)
    train_num = int(total_num * train_ratio)
    val_num = int(total_num * val_ratio)
    
    train_data = all_data[:train_num]
    val_data = all_data[train_num:train_num + val_num]
    test_data = all_data[train_num + val_num:]
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
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
    return vocab_dict


class DeepFashionDataset(Dataset):
    """
    DeepFashion-MultiModal PyTorch数据集类
    用于PyTorch DataLoader按批次产生数据
    """
    
    def __init__(self, dataset_path, vocab_path, split, max_len=50, transform=None):
        """
        参数：
            dataset_path：json格式数据文件路径
            vocab_path：json格式词典文件路径
            split：'train'、'val'或'test'
            max_len：文本描述包含的最大单词数
            transform：图像预处理方法
        """
        self.split = split
        assert self.split in {'train', 'val', 'test'}
        self.max_len = max_len
        
        # 载入数据集
        print(f"正在加载 {split} 数据集...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 载入词典
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # 图像预处理
        self.transform = transform
        
        # 数据集大小
        self.dataset_size = len(self.data['CAPTIONS'])
        print(f"{split} 数据集加载完成，共 {self.dataset_size} 条数据")
    
    def __getitem__(self, i):
        """
        获取第i个数据样本
        
        返回：
            img：预处理后的图像张量
            caption：编码后的caption张量
            caplen：caption的实际长度
        """
        # 读取并转换图像
        img = Image.open(self.data['IMAGES'][i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        # 获取caption和长度
        caption_list = self.data['CAPTIONS'][i]
        caplen = len(caption_list)
        
        # padding到最大长度
        caption = torch.LongTensor(
            caption_list + [self.vocab['<pad>']] * (self.max_len + 2 - caplen)
        )
        
        return img, caption, caplen
    
    def __len__(self):
        return self.dataset_size


def get_transform(split, image_size=224):
    """
    获取图像预处理transform
    
    参数：
        split：'train'、'val'或'test'
        image_size：目标图像大小
    
    返回：
        transform对象
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_data_loaders(data_dir='data', 
                        vocab_path='data/vocab.json',
                        batch_size=32, 
                        num_workers=4,
                        image_size=224):
    """
    创建训练、验证和测试数据加载器
    
    参数：
        data_dir：数据集目录
        vocab_path：词典文件路径
        batch_size：批次大小
        num_workers：数据加载线程数
        image_size：图像大小
    
    返回：
        train_loader, val_loader, test_loader
    """
    
    # 创建数据集
    train_dataset = DeepFashionDataset(
        os.path.join(data_dir, 'train_data.json'),
        vocab_path,
        'train',
        transform=get_transform('train', image_size)
    )
    
    val_dataset = DeepFashionDataset(
        os.path.join(data_dir, 'val_data.json'),
        vocab_path,
        'val',
        transform=get_transform('val', image_size)
    )
    
    test_dataset = DeepFashionDataset(
        os.path.join(data_dir, 'test_data.json'),
        vocab_path,
        'test',
        transform=get_transform('test', image_size)
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\n数据加载器创建完成！")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 示例：整理数据集
    print("=" * 50)
    print("开始整理DeepFashion-MultiModal数据集")
    print("=" * 50)
    
    vocab = create_deepfashion_dataset(
        data_dir='data',
        min_word_count=5,
        max_len=50,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print("\n" + "=" * 50)
    print("创建数据加载器")
    print("=" * 50)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='data',
        vocab_path='data/vocab.json',
        batch_size=32,
        num_workers=0,  # Windows下建议设置为0
        image_size=224
    )
    
    # 测试读取一个batch
    print("\n" + "=" * 50)
    print("测试读取一个batch")
    print("=" * 50)
    
    for imgs, caps, cap_lens in train_loader:
        print(f"图像批次形状: {imgs.shape}")
        print(f"Caption批次形状: {caps.shape}")
        print(f"Caption长度: {cap_lens[:5]}")
        break
    
    print("\n全部完成！")
