import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_image_only_dataloaders, get_fusion_dataloaders, SELECTED_CLASSES
from model import build_image_only_model, build_fusion_model


@torch.no_grad()
def evaluate_image_only(model, test_loader, device):
    """在测试集上评估image-only模型"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(test_loader, desc="Evaluating Image-Only"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, all_preds, all_labels


@torch.no_grad()
def evaluate_fusion(model, test_loader, device):
    """在测试集上评估fusion模型"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, captions, labels in tqdm(test_loader, desc="Evaluating Fusion"):
        images = images.to(device)
        captions = captions.to(device)
        labels = labels.to(device)
        
        outputs = model(images, captions)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, all_preds, all_labels


def load_image_only_model(checkpoint_path, device):
    """加载image-only最佳模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint['num_classes']
    
    model = build_image_only_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"加载image-only模型，最佳val_acc: {checkpoint['val_acc']:.2f}% (epoch {checkpoint['epoch']})")
    return model, num_classes


def load_fusion_model(checkpoint_path, device):
    """加载fusion最佳模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint['num_classes']
    vocab = checkpoint['vocab']
    use_gru = checkpoint.get('use_gru', True)
    vocab_size = len(vocab)
    pad_idx = vocab.get('<pad>', 0)
    
    model = build_fusion_model(
        num_classes=num_classes,
        vocab_size=vocab_size,
        use_gru=use_gru,
        pad_idx=pad_idx
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"加载fusion模型，最佳val_acc: {checkpoint['val_acc']:.2f}% (epoch {checkpoint['epoch']})")
    return model, num_classes, vocab


def generate_prediction_samples(data_dir, captions_json, img_only_model, fusion_model, 
                                 vocab, device, output_dir, num_samples=5):
    """生成预测样例展示"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试集
    _, _, test_loader, num_classes = get_image_only_dataloaders(data_dir, batch_size=1)
    
    # 加载caption映射
    with open(captions_json, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    caption_map = {}
    for item in captions_data:
        img_name = os.path.basename(item['image_path'])
        caption_map[img_name] = item['caption']
    
    # 获取测试集图像路径
    test_dataset = test_loader.dataset
    if hasattr(test_dataset, 'dataset'):
        base_dataset = test_dataset.dataset
        indices = test_dataset.indices
    else:
        base_dataset = test_dataset
        indices = list(range(len(test_dataset)))
    
    # 随机选几个样本
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(indices)), min(num_samples, len(indices)))
    
    samples = []
    pad_idx = vocab.get('<pad>', 0)
    
    print(f"\n生成 {len(sample_indices)} 个预测样例...")
    
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(4 * len(sample_indices), 5))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        real_idx = indices[idx]
        img_path, true_label = base_dataset.samples[real_idx]
        img_name = os.path.basename(img_path)
        true_class = SELECTED_CLASSES[true_label]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        transform = test_loader.dataset.dataset.transform if hasattr(test_loader.dataset, 'dataset') else test_loader.dataset.transform
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Image-only预测
        with torch.no_grad():
            img_only_output = img_only_model(img_tensor)
            _, img_only_pred = img_only_output.max(1)
            img_only_pred = img_only_pred.item()
            img_only_class = SELECTED_CLASSES[img_only_pred]
        
        # Fusion预测
        caption = caption_map.get(img_name, "a pet")
        # tokenize
        words = caption.lower().split()
        tokens = [vocab.get('<start>', 2)]
        for word in words:
            tokens.append(vocab.get(word, vocab.get('<unk>', 1)))
        tokens.append(vocab.get('<end>', 3))
        max_len = 30
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [pad_idx] * (max_len - len(tokens))
        caption_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            fusion_output = fusion_model(img_tensor, caption_tensor)
            _, fusion_pred = fusion_output.max(1)
            fusion_pred = fusion_pred.item()
            fusion_class = SELECTED_CLASSES[fusion_pred]
        
        samples.append({
            'image_path': img_path,
            'image_name': img_name,
            'true_class': true_class,
            'caption': caption,
            'image_only_pred': img_only_class,
            'image_only_correct': img_only_pred == true_label,
            'fusion_pred': fusion_class,
            'fusion_correct': fusion_pred == true_label,
        })
        
        # 显示图像
        axes[i].imshow(image)
        title = f"True: {true_class}\nImgOnly: {img_only_class}\nFusion: {fusion_class}"
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存样例详情
    with open(os.path.join(output_dir, 'prediction_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    # 打印样例
    print("\n=== 预测样例 ===")
    for i, s in enumerate(samples):
        print(f"\n[{i+1}] {s['image_name']}")
        print(f"    真实类别: {s['true_class']}")
        print(f"    Caption: {s['caption']}")
        print(f"    Image-Only预测: {s['image_only_pred']} {'✓' if s['image_only_correct'] else '✗'}")
        print(f"    Fusion预测: {s['fusion_pred']} {'✓' if s['fusion_correct'] else '✗'}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--data_dir", type=str, default="../data", help="Dataset directory")
    parser.add_argument("--captions", type=str, default="../captions/captions.json", 
                        help="Captions JSON file")
    parser.add_argument("--image_only_ckpt", type=str, 
                        default="../checkpoints/image_only_best.pth",
                        help="Image-only model checkpoint")
    parser.add_argument("--fusion_ckpt", type=str, 
                        default="../checkpoints/fusion_best.pth",
                        help="Fusion model checkpoint")
    parser.add_argument("--output_dir", type=str, default="../results", 
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of prediction samples to show")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # 评估image-only模型
    if os.path.exists(args.image_only_ckpt):
        print("\n" + "=" * 50)
        print("评估 Image-Only 模型")
        print("=" * 50)
        
        img_only_model, num_classes = load_image_only_model(args.image_only_ckpt, device)
        _, _, test_loader, _ = get_image_only_dataloaders(args.data_dir, batch_size=32)
        
        img_only_acc, _, _ = evaluate_image_only(img_only_model, test_loader, device)
        print(f"\nImage-Only 测试集准确率: {img_only_acc:.2f}%")
        results['image_only_test_acc'] = img_only_acc
    else:
        print(f"Image-only checkpoint不存在: {args.image_only_ckpt}")
        img_only_model = None
    
    # 评估fusion模型
    if os.path.exists(args.fusion_ckpt) and os.path.exists(args.captions):
        print("\n" + "=" * 50)
        print("评估 Image-Text Fusion 模型")
        print("=" * 50)
        
        fusion_model, num_classes, vocab = load_fusion_model(args.fusion_ckpt, device)
        _, _, test_loader, _, _ = get_fusion_dataloaders(
            args.data_dir, args.captions, batch_size=32
        )
        
        fusion_acc, _, _ = evaluate_fusion(fusion_model, test_loader, device)
        print(f"\nFusion 测试集准确率: {fusion_acc:.2f}%")
        results['fusion_test_acc'] = fusion_acc
    else:
        print(f"Fusion checkpoint或captions不存在")
        fusion_model = None
        vocab = None
    
    # 对比结果
    if 'image_only_test_acc' in results and 'fusion_test_acc' in results:
        print("\n" + "=" * 50)
        print("结果对比")
        print("=" * 50)
        print(f"Image-Only 准确率: {results['image_only_test_acc']:.2f}%")
        print(f"Fusion 准确率:     {results['fusion_test_acc']:.2f}%")
        diff = results['fusion_test_acc'] - results['image_only_test_acc']
        print(f"差异: {diff:+.2f}%")
        results['accuracy_diff'] = diff
    
    # 生成预测样例
    if img_only_model is not None and fusion_model is not None and vocab is not None:
        print("\n" + "=" * 50)
        print("生成预测样例")
        print("=" * 50)
        generate_prediction_samples(
            args.data_dir, args.captions,
            img_only_model, fusion_model, vocab,
            device, args.output_dir, args.num_samples
        )
    
    # 保存最终结果
    results['num_classes'] = len(SELECTED_CLASSES)
    results['selected_classes'] = SELECTED_CLASSES
    
    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n最终结果已保存到: {os.path.join(args.output_dir, 'final_results.json')}")
    
    # 打印总结
    print("\n" + "=" * 50)
    print("评估总结")
    print("=" * 50)
    print(f"使用类别数: {len(SELECTED_CLASSES)}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        elif isinstance(v, list):
            continue
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
