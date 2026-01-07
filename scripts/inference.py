import os
import sys
import argparse
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(checkpoint_path, model_type, device):
    print(f"Loading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    vocab_path = config.get('vocab_path', os.path.join(PROJECT_ROOT, 'data', 'vocab.json'))
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    if model_type == 'vit':
        from models.vit_transformer_model import build_model
    else:
        from models.grid_transformer_model import build_model
    
    model = build_model(len(vocab), config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model, vocab, config


def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def generate_caption(model, image_tensor, vocab, device, method='greedy', beam_size=5, max_len=50):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        caption_ids = model.generate(
            image_tensor,
            start_token=vocab['<start>'],
            end_token=vocab['<end>'],
            max_len=max_len,
            method=method,
            beam_size=beam_size
        )
    
    caption_ids = caption_ids[0].cpu().tolist()
    idx2word = {idx: word for word, idx in vocab.items()}
    
    words = []
    for idx in caption_ids:
        if idx == vocab['<end>']:
            break
        if idx not in [vocab['<start>'], vocab['<pad>']]:
            words.append(idx2word[idx])
    
    return ' '.join(words)


def visualize(image_path, caption, save_path=None):
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Caption:\n{caption}", fontsize=12, wrap=True, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Image Caption Inference')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--model_type', type=str, choices=['vit', 'grid'], required=True, help='Model type')
    parser.add_argument('--method', type=str, default='greedy', choices=['greedy', 'beam_search'], help='Generation method')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--save', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found {args.image}")
        return
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found {args.checkpoint}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, vocab, config = load_model(args.checkpoint, args.model_type, device)
    image_tensor = preprocess_image(args.image)
    
    print(f"\nGenerating caption using {args.method}...")
    caption = generate_caption(model, image_tensor, vocab, device, method=args.method, beam_size=args.beam_size)
    
    print("\n" + "=" * 60)
    print(f"Image: {args.image}")
    print(f"Caption: {caption}")
    print("=" * 60)
    
    if not args.no_vis:
        visualize(args.image, caption, save_path=args.save)


if __name__ == '__main__':
    main()
