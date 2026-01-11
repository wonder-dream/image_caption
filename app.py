"""
图像描述生成 Web 应用
简单的 Flask 后端，提供模型推理 API
支持自训练模型和 BLIP 预训练模型
"""

import os
import sys
import json
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import io
import base64

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# 全局变量存储已加载的模型
loaded_models = {}

# 可用的模型列表
AVAILABLE_MODELS = {
    # ===== 自训练模型 =====
    'vit_base': {
        'name': 'ViT Transformer (基础版)',
        'path': 'checkpoints/vit_transformer/best_model.pth',
        'type': 'vit',
        'category': '自训练模型'
    },
    'vit_optimized': {
        'name': 'ViT Transformer (优化版)',
        'path': 'checkpoints/vit_transformer_optimized/best_model.pth',
        'type': 'vit',
        'category': '自训练模型'
    },
    'vit_scst': {
        'name': 'ViT Transformer (SCST强化学习)',
        'path': 'checkpoints/vit_transformer_scst_opt/best_model.pth',
        'type': 'vit',
        'category': '自训练模型'
    },
    'grid_base': {
        'name': 'Grid Transformer (基础版)',
        'path': 'checkpoints/grid_transformer/best_model.pth',
        'type': 'grid',
        'category': '自训练模型'
    },
    'grid_optimized': {
        'name': 'Grid Transformer (优化版)',
        'path': 'checkpoints/grid_transformer_optimized/best_model.pth',
        'type': 'grid',
        'category': '自训练模型'
    },
    'grid_scst': {
        'name': 'Grid Transformer (SCST强化学习)',
        'path': 'checkpoints/grid_transformer_scst_opt/best_model.pth',
        'type': 'grid',
        'category': '自训练模型'
    },
    # ===== BLIP 预训练模型 =====
    'blip_base': {
        'name': 'BLIP (预训练原始版)',
        'path': 'pretrained_models/Salesforce_blip-image-captioning-base',
        'type': 'blip',
        'category': 'BLIP大模型'
    },
    'blip_finetuned': {
        'name': 'BLIP (微调后)',
        'path': 'checkpoints/blip_finetuned/best_model',
        'type': 'blip',
        'category': 'BLIP大模型'
    }
}


def get_device():
    """获取可用设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_blip_model(model_path):
    """加载 BLIP 模型"""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    device = get_device()
    print(f"加载 BLIP 模型: {model_path}")
    
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    return {
        'model': model,
        'processor': processor,
        'device': device,
        'type': 'blip'
    }


def load_transformer_model(model_key, model_info):
    """加载自训练的 Transformer 模型"""
    checkpoint_path = os.path.join(PROJECT_ROOT, model_info['path'])
    device = get_device()
    
    print(f"加载模型: {model_info['name']} ({checkpoint_path})")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # 加载词典
    vocab_path = config.get('vocab_path', os.path.join(PROJECT_ROOT, 'data', 'vocab.json'))
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(PROJECT_ROOT, 'data', 'vocab.json')
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 根据模型类型加载
    if model_info['type'] == 'vit':
        from models.vit_transformer_model import build_model
    else:
        from models.grid_transformer_model import build_model
    
    model = build_model(len(vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return {
        'model': model,
        'vocab': vocab,
        'config': config,
        'device': device,
        'type': 'transformer'
    }


def load_model(model_key):
    """加载指定模型"""
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"未知模型: {model_key}")
    
    model_info = AVAILABLE_MODELS[model_key]
    checkpoint_path = os.path.join(PROJECT_ROOT, model_info['path'])
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # 根据模型类型选择加载方式
    if model_info['type'] == 'blip':
        model_data = load_blip_model(checkpoint_path)
    else:
        model_data = load_transformer_model(model_key, model_info)
    
    # 缓存模型
    loaded_models[model_key] = model_data
    print(f"模型加载成功！")
    
    return model_data


def preprocess_image(image, image_size=224):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def generate_caption_blip(model_data, image, method='greedy', beam_size=5):
    """使用 BLIP 模型生成描述"""
    model = model_data['model']
    processor = model_data['processor']
    device = model_data['device']
    
    # BLIP 使用自己的预处理
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if method == 'beam_search':
            output = model.generate(
                **inputs,
                max_length=50,
                num_beams=beam_size,
                early_stopping=True
            )
        else:
            output = model.generate(
                **inputs,
                max_length=50,
                do_sample=False
            )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_caption_transformer(model_data, image_tensor, method='greedy', beam_size=5):
    """使用自训练 Transformer 模型生成描述"""
    model = model_data['model']
    vocab = model_data['vocab']
    device = model_data['device']
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 生成描述
        if method == 'beam_search':
            try:
                caption_ids = model.generate(
                    image_tensor,
                    start_token=vocab['<start>'],
                    end_token=vocab['<end>'],
                    max_len=50,
                    method='beam_search',
                    beam_size=beam_size
                )
            except:
                # 如果 beam_search 失败，回退到 greedy
                caption_ids = model.generate(
                    image_tensor,
                    start_token=vocab['<start>'],
                    end_token=vocab['<end>'],
                    max_len=50,
                    method='greedy'
                )
        else:
            caption_ids = model.generate(
                image_tensor,
                start_token=vocab['<start>'],
                end_token=vocab['<end>'],
                max_len=50,
                method='greedy'
            )
    
    # 转换为文本
    caption_ids = caption_ids[0].cpu().tolist()
    idx2word = {idx: word for word, idx in vocab.items()}
    
    words = []
    for idx in caption_ids:
        if idx == vocab['<end>']:
            break
        if idx not in [vocab['<start>'], vocab['<pad>']]:
            words.append(idx2word.get(idx, '<unk>'))
    
    return ' '.join(words)


@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('frontend', 'index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    models = []
    for key, info in AVAILABLE_MODELS.items():
        checkpoint_path = os.path.join(PROJECT_ROOT, info['path'])
        models.append({
            'key': key,
            'name': info['name'],
            'type': info['type'],
            'category': info.get('category', '其他'),
            'available': os.path.exists(checkpoint_path)
        })
    return jsonify({'models': models})


@app.route('/api/generate', methods=['POST'])
def generate():
    """生成图像描述"""
    try:
        # 获取参数
        model_key = request.form.get('model', 'grid_scst')
        method = request.form.get('method', 'greedy')
        beam_size = int(request.form.get('beam_size', 5))
        
        # 获取图像
        if 'image' not in request.files:
            return jsonify({'error': '请上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '请选择图片文件'}), 400
        
        # 读取图像
        image = Image.open(file.stream).convert('RGB')
        
        # 加载模型
        model_data = load_model(model_key)
        
        # 根据模型类型生成描述
        if model_data.get('type') == 'blip':
            caption = generate_caption_blip(model_data, image, method, beam_size)
        else:
            # 预处理图像
            image_tensor = preprocess_image(image)
            caption = generate_caption_transformer(model_data, image_tensor, method, beam_size)
        
        return jsonify({
            'success': True,
            'caption': caption,
            'model': AVAILABLE_MODELS[model_key]['name'],
            'method': method
        })
        
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'生成失败: {str(e)}'}), 500


@app.route('/api/device', methods=['GET'])
def get_device_info():
    """获取设备信息"""
    device = get_device()
    return jsonify({
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })


if __name__ == '__main__':
    print("=" * 50)
    print("图像描述生成 Web 应用")
    print("=" * 50)
    print(f"设备: {get_device()}")
    print(f"可用模型: {len(AVAILABLE_MODELS)} 个")
    print("=" * 50)
    print("启动服务器: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
