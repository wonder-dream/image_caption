#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP 预训练模型下载脚本

自动从 HuggingFace 下载 BLIP 模型用于图像描述任务微调
支持 Linux 服务器环境，可离线使用
"""

import os
import argparse
from pathlib import Path


def download_blip_model(model_name, save_dir, use_mirror=False):
    """
    下载 BLIP 预训练模型
    
    参数:
        model_name: 模型名称
        save_dir: 保存目录
        use_mirror: 是否使用国内镜像 (hf-mirror.com)
    """
    # 设置镜像 (可选，国内服务器推荐)
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("使用 HuggingFace 镜像: https://hf-mirror.com")
    
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    save_path = os.path.join(save_dir, model_name.replace('/', '_'))
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"下载模型: {model_name}")
    print(f"保存路径: {save_path}")
    print(f"{'='*60}\n")
    
    try:
        if 'blip2' in model_name.lower():
            # BLIP-2 模型
            print("正在下载 BLIP-2 Processor...")
            processor = Blip2Processor.from_pretrained(model_name)
            processor.save_pretrained(save_path)
            
            print("正在下载 BLIP-2 Model...")
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto"
            )
            model.save_pretrained(save_path)
        else:
            # BLIP 模型
            print("正在下载 BLIP Processor...")
            processor = BlipProcessor.from_pretrained(model_name)
            processor.save_pretrained(save_path)
            
            print("正在下载 BLIP Model...")
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            model.save_pretrained(save_path)
        
        print(f"\n✅ 模型下载完成!")
        print(f"保存位置: {save_path}")
        
        # 统计文件大小
        total_size = sum(
            f.stat().st_size for f in Path(save_path).rglob('*') if f.is_file()
        )
        print(f"总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")
        
        return save_path
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 使用镜像: --mirror")
        print("3. 设置代理: export https_proxy=http://your_proxy:port")
        raise


def verify_model(model_path, model_type='blip'):
    """验证下载的模型是否完整"""
    print(f"\n验证模型: {model_path}")
    
    required_files = [
        'config.json',
        'preprocessor_config.json',
    ]
    
    # 检查必需文件
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            missing.append(f)
    
    # 检查模型权重文件
    has_weights = any([
        os.path.exists(os.path.join(model_path, 'pytorch_model.bin')),
        os.path.exists(os.path.join(model_path, 'model.safetensors')),
        any(Path(model_path).glob('pytorch_model-*.bin')),
        any(Path(model_path).glob('model-*.safetensors')),
    ])
    
    if missing:
        print(f"❌ 缺少文件: {missing}")
        return False
    
    if not has_weights:
        print("❌ 缺少模型权重文件")
        return False
    
    print("✅ 模型文件完整")
    
    # 尝试加载测试
    print("正在测试加载模型...")
    try:
        if 'blip2' in model_path.lower():
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            processor = Blip2Processor.from_pretrained(model_path)
            model = Blip2ForConditionalGeneration.from_pretrained(model_path)
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            processor = BlipProcessor.from_pretrained(model_path)
            model = BlipForConditionalGeneration.from_pretrained(model_path)
        
        print("✅ 模型加载成功")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='BLIP 预训练模型下载脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 BLIP-base (推荐，约 1GB)
  python download_blip.py --model blip-base
  
  # 下载 BLIP-large (约 2GB)
  python download_blip.py --model blip-large
  
  # 下载 BLIP-2 (约 15GB，需要更多显存)
  python download_blip.py --model blip2
  
  # 使用国内镜像下载
  python download_blip.py --model blip-base --mirror
  
  # 指定保存路径
  python download_blip.py --model blip-base --save_dir /data/models
        """
    )
    
    parser.add_argument(
        '--model', type=str, default='blip-base',
        choices=['blip-base', 'blip-large', 'blip2', 'blip2-flan-t5'],
        help='要下载的模型 (默认: blip-base)'
    )
    parser.add_argument(
        '--save_dir', type=str, default='pretrained_models',
        help='模型保存目录 (默认: pretrained_models)'
    )
    parser.add_argument(
        '--mirror', action='store_true',
        help='使用 HuggingFace 国内镜像 (hf-mirror.com)'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='下载后验证模型完整性'
    )
    
    args = parser.parse_args()
    
    # 模型名称映射
    model_map = {
        'blip-base': 'Salesforce/blip-image-captioning-base',
        'blip-large': 'Salesforce/blip-image-captioning-large',
        'blip2': 'Salesforce/blip2-opt-2.7b',
        'blip2-flan-t5': 'Salesforce/blip2-flan-t5-xl',
    }
    
    model_name = model_map[args.model]
    
    print("="*60)
    print("BLIP 预训练模型下载器")
    print("="*60)
    print(f"模型: {args.model} ({model_name})")
    print(f"保存目录: {args.save_dir}")
    print(f"使用镜像: {args.mirror}")
    print()
    
    # 下载模型
    save_path = download_blip_model(
        model_name=model_name,
        save_dir=args.save_dir,
        use_mirror=args.mirror
    )
    
    # 验证模型
    if args.verify:
        verify_model(save_path, args.model)
    
    print("\n" + "="*60)
    print("下载完成!")
    print("="*60)
    print(f"\n下一步: 使用以下命令进行微调:")
    print(f"  python scripts/finetune_blip.py --model_path {save_path}")


if __name__ == '__main__':
    main()
