#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BLIP 预训练模型下载脚本

支持多种下载方式，适合中国大陆网络环境：
1. ModelScope (阿里云魔搭) - 国内最稳定
2. HuggingFace 镜像 (hf-mirror.com)
3. 原版 HuggingFace
"""

import os
import argparse
from pathlib import Path


def download_from_modelscope(model_name, save_dir):
    """
    从 ModelScope (阿里云魔搭) 下载模型
    国内速度最快最稳定
    """
    from modelscope import snapshot_download
    
    # ModelScope 上的模型名称映射
    modelscope_map = {
        'Salesforce/blip-image-captioning-base': 'AI-ModelScope/blip-image-captioning-base',
        'Salesforce/blip-image-captioning-large': 'AI-ModelScope/blip-image-captioning-large',
        'Salesforce/blip2-opt-2.7b': 'AI-ModelScope/blip2-opt-2.7b',
    }
    
    ms_model_name = modelscope_map.get(model_name)
    if not ms_model_name:
        print(f"ModelScope 暂不支持: {model_name}")
        return None
    
    print(f"\n{'='*60}")
    print(f"从 ModelScope 下载: {ms_model_name}")
    print(f"{'='*60}\n")
    
    save_path = os.path.join(save_dir, model_name.replace('/', '_'))
    
    try:
        model_dir = snapshot_download(
            ms_model_name,
            cache_dir=save_dir,
            local_dir=save_path
        )
        print(f"\n✅ 下载完成: {model_dir}")
        return save_path
    except Exception as e:
        print(f"ModelScope 下载失败: {e}")
        return None


def download_from_huggingface(model_name, save_dir, use_mirror=False):
    """
    从 HuggingFace 下载模型
    """
    # 设置镜像
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("使用 HuggingFace 镜像: https://hf-mirror.com")
    
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    save_path = os.path.join(save_dir, model_name.replace('/', '_'))
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"从 HuggingFace 下载: {model_name}")
    print(f"保存路径: {save_path}")
    print(f"{'='*60}\n")
    
    try:
        if 'blip2' in model_name.lower():
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
            print("正在下载 BLIP Processor...")
            processor = BlipProcessor.from_pretrained(model_name)
            processor.save_pretrained(save_path)
            
            print("正在下载 BLIP Model...")
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            model.save_pretrained(save_path)
        
        print(f"\n✅ 模型下载完成!")
        print(f"保存位置: {save_path}")
        
        total_size = sum(
            f.stat().st_size for f in Path(save_path).rglob('*') if f.is_file()
        )
        print(f"总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")
        
        return save_path
        
    except Exception as e:
        print(f"\n❌ HuggingFace 下载失败: {e}")
        return None


def download_blip_model(model_name, save_dir, source='auto', use_mirror=False):
    """
    下载 BLIP 预训练模型
    
    参数:
        model_name: 模型名称
        save_dir: 保存目录
        source: 下载源 ('modelscope', 'huggingface', 'auto')
        use_mirror: HuggingFace 是否使用镜像
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = None
    
    if source == 'modelscope' or source == 'auto':
        print("\n尝试从 ModelScope (阿里云魔搭) 下载...")
        try:
            save_path = download_from_modelscope(model_name, save_dir)
        except ImportError:
            print("ModelScope 未安装，请运行: pip install modelscope")
            if source == 'modelscope':
                raise
        except Exception as e:
            print(f"ModelScope 下载失败: {e}")
            if source == 'modelscope':
                raise
    
    if save_path is None and (source == 'huggingface' or source == 'auto'):
        print("\n尝试从 HuggingFace 下载...")
        save_path = download_from_huggingface(model_name, save_dir, use_mirror)
    
    if save_path is None:
        print("\n" + "="*60)
        print("❌ 自动下载失败！请尝试手动下载：")
        print("="*60)
        print_manual_download_guide(model_name, save_dir)
        raise RuntimeError("下载失败")
    
    return save_path


def print_manual_download_guide(model_name, save_dir):
    """打印手动下载指南"""
    print(f"""
方法1: 使用 ModelScope (推荐)
    pip install modelscope
    然后重新运行此脚本: python download_blip.py --source modelscope

方法2: 使用 huggingface-cli + 镜像
    export HF_ENDPOINT=https://hf-mirror.com
    pip install huggingface_hub
    huggingface-cli download {model_name} --local-dir {save_dir}/{model_name.replace('/', '_')}

方法3: 手动下载
    1. 访问 https://hf-mirror.com/{model_name}
    2. 下载所有文件到 {save_dir}/{model_name.replace('/', '_')}/
    3. 确保包含: config.json, pytorch_model.bin 或 model.safetensors

方法4: 使用学术网络/VPN
    export https_proxy=http://your_proxy:port
    python download_blip.py --source huggingface
""")


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
        description='BLIP 预训练模型下载脚本 (支持中国大陆网络)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 ModelScope 下载 (国内推荐，最稳定)
  python download_blip.py --model blip-base --source modelscope
  
  # 从 HuggingFace 镜像下载
  python download_blip.py --model blip-base --source huggingface --mirror
  
  # 自动选择下载源 (先尝试 ModelScope)
  python download_blip.py --model blip-base --source auto
  
  # 下载 BLIP-large
  python download_blip.py --model blip-large --source modelscope
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
        '--source', type=str, default='auto',
        choices=['auto', 'modelscope', 'huggingface'],
        help='下载源: auto(自动), modelscope(阿里云魔搭), huggingface'
    )
    parser.add_argument(
        '--mirror', action='store_true',
        help='HuggingFace 使用国内镜像 (hf-mirror.com)'
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
    print("BLIP 预训练模型下载器 (中国大陆优化版)")
    print("="*60)
    print(f"模型: {args.model} ({model_name})")
    print(f"保存目录: {args.save_dir}")
    print(f"下载源: {args.source}")
    if args.source == 'huggingface':
        print(f"使用镜像: {args.mirror}")
    print()
    
    # 下载模型
    save_path = download_blip_model(
        model_name=model_name,
        save_dir=args.save_dir,
        source=args.source,
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
