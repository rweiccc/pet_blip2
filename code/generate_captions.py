import os
import json
import argparse
from tqdm import tqdm

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import PetDataset, SELECTED_CLASSES


def generate_captions(data_dir, output_path, model_name="Salesforce/blip2-opt-2.7b", 
                      max_new_tokens=30, device=None):
   
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型和processor
    print(f"加载BLIP-2模型: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    model.eval()
    
    # 加载数据集
    print("加载数据集...")
    dataset = PetDataset(data_dir, selected_classes=SELECTED_CLASSES, transform=None, download=True)
    
    # 生成captions
    results = []
    print(f"开始生成caption，共 {len(dataset)} 张图像...")
    
    for idx in tqdm(range(len(dataset))):
        img_path, label_idx = dataset.samples[idx]
        class_name = dataset.selected_classes[label_idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 预处理
        inputs = processor(images=image, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = inputs.to(torch.float16)
        
        # 生成caption
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        results.append({
            "image_path": str(img_path),
            "image_name": os.path.basename(img_path),
            "class_name": class_name,
            "class_idx": label_idx,
            "caption": generated_text
        })
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nCaption生成完成，已保存到: {output_path}")
    print(f"总计生成 {len(results)} 条caption")
    
    # 展示5个样例
    print("\n=== Caption样例展示 ===")
    for i, item in enumerate(results[:5]):
        print(f"\n[{i+1}] 类别: {item['class_name']}")
        print(f"    图像: {item['image_name']}")
        print(f"    Caption: {item['caption']}")
    
    return results


def generate_mock_captions(data_dir, output_path):
    
    print("生成模拟caption（用于测试流程）...")
    
    # 加载数据集
    dataset = PetDataset(data_dir, selected_classes=SELECTED_CLASSES, transform=None, download=True)
    
    # 各类别的模板描述
    caption_templates = {
        'Abyssinian': [
            "a brown abyssinian cat with large ears sitting on a white background",
            "an abyssinian cat with short fur looking at the camera",
            "a beautiful abyssinian cat portrait in studio lighting",
        ],
        'Bengal': [
            "a bengal cat with leopard spots standing on a wooden floor",
            "a spotted bengal cat with green eyes looking curious",
            "a bengal cat with distinctive rosette patterns on its fur",
        ],
        'Birman': [
            "a fluffy birman cat with white paws and blue eyes",
            "a birman cat with long silky fur sitting gracefully",
            "a sacred birman cat with dark points and white gloves",
        ],
        'Persian': [
            "a fluffy white persian cat with a flat face",
            "a persian cat with long thick fur sitting on a cushion",
            "a beautiful persian cat with big round eyes",
        ],
        'Siamese': [
            "a siamese cat with blue eyes and dark brown points",
            "a sleek siamese cat sitting on a white background",
            "a siamese cat with distinctive colorpoint coat pattern",
        ],
        'american_bulldog': [
            "a muscular american bulldog with a white coat",
            "an american bulldog with a strong build looking forward",
            "a stocky american bulldog standing on grass",
        ],
        'american_pit_bull_terrier': [
            "an american pit bull terrier with a brown and white coat",
            "a muscular pit bull terrier looking at the camera",
            "a pit bull terrier with short fur and floppy ears",
        ],
        'english_cocker_spaniel': [
            "a golden english cocker spaniel with long floppy ears",
            "an english cocker spaniel sitting on green grass",
            "a friendly cocker spaniel with wavy fur and big eyes",
        ],
        'english_setter': [
            "an english setter dog with white and brown speckled fur",
            "a graceful english setter standing in a field",
            "an english setter with long silky ears looking alert",
        ],
        'staffordshire_bull_terrier': [
            "a staffordshire bull terrier with a muscular build",
            "a stocky staffordshire bull terrier with a short coat",
            "a staffy dog with a broad head and strong jaw",
        ],
    }
    
    results = []
    import random
    random.seed(42)
    
    for idx in range(len(dataset)):
        img_path, label_idx = dataset.samples[idx]
        class_name = dataset.selected_classes[label_idx]
        
        templates = caption_templates.get(class_name, ["a dog or cat sitting in front of white background"])
        caption = random.choice(templates)
        
        results.append({
            "image_path": str(img_path),
            "image_name": os.path.basename(img_path),
            "class_name": class_name,
            "class_idx": label_idx,
            "caption": caption
        })
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n模拟caption生成完成，已保存到: {output_path}")
    print(f"总计生成 {len(results)} 条caption")
    
    # 展示5个样例
    print("\n=== Caption样例展示 ===")
    for i, item in enumerate(results[:5]):
        print(f"\n[{i+1}] 类别: {item['class_name']}")
        print(f"    图像: {item['image_name']}")
        print(f"    Caption: {item['caption']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for pet images using BLIP-2")
    parser.add_argument("--data_dir", type=str, default="../data", help="Dataset root directory")
    parser.add_argument("--output", type=str, default="../captions/captions.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b", help="BLIP-2 model name")
    parser.add_argument("--mock", action="store_true", help="Use mock captions for testing (no BLIP-2 needed)")
    parser.add_argument("--max_tokens", type=int, default=30, help="Max new tokens for generation")
    
    args = parser.parse_args()
    
    if args.mock:
        generate_mock_captions(args.data_dir, args.output)
    else:
        generate_captions(args.data_dir, args.output, args.model, args.max_tokens)
