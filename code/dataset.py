import os
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


# 推荐的10个宠物类别
SELECTED_CLASSES = [
    'Abyssinian', 'Bengal', 'Birman', 'Persian', 'Siamese',
    'american_bulldog', 'american_pit_bull_terrier',
    'english_cocker_spaniel', 'english_setter', 'staffordshire_bull_terrier'
]


def get_image_transforms(img_size=224, train=True):
    """获取图像预处理transform"""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class PetDataset(Dataset):
    """
    宠物品种图像数据集（仅图像，用于image-only模型）
    从Oxford-IIIT Pet数据集中选择指定类别
    """
    def __init__(self, root_dir, selected_classes=None, transform=None, download=True):
        """
        Args:
            root_dir: 数据集根目录
            selected_classes: 选择的类别列表，None则使用全部推荐类别
            transform: 图像变换
            download: 是否自动下载
        """
        self.root_dir = root_dir
        self.transform = transform
        
        if selected_classes is None:
            selected_classes = SELECTED_CLASSES
        self.selected_classes = selected_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        self.num_classes = len(selected_classes)
        
        # 加载完整数据集
        full_dataset = datasets.OxfordIIITPet(
            root=root_dir,
            split='trainval',  # 使用trainval合并集再自己划分
            target_types='category',
            download=download
        )
        
        # 获取类别名称映射
        self.idx_to_class = full_dataset.classes
        
        # 筛选指定类别的样本
        self.samples = []  # (image_path, label_idx)
        for idx in range(len(full_dataset)):
            img_path, target = full_dataset._images[idx], full_dataset._labels[idx]
            class_name = self.idx_to_class[target]
            if class_name in self.class_to_idx:
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"加载完成：共 {len(self.samples)} 张图像，{self.num_classes} 个类别")
        for cls in selected_classes:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[cls])
            print(f"  {cls}: {count} 张")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    将数据集划分为train/val/test
    保证每个类别在各子集中都有样本
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    
    # 按类别分组
    class_samples = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(idx)
    
    # 每个类别按比例划分
    train_indices, val_indices, test_indices = [], [], []
    random.seed(seed)
    
    for label, indices in class_samples.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # 创建子集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"\n数据集划分：")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")
    
    return train_dataset, val_dataset, test_dataset


class PetCaptionDataset(Dataset):
    """
    图文融合数据集：同时加载图像和对应的caption文本
    """
    def __init__(self, base_dataset, captions_json, vocab=None, max_len=30):
        """
        Args:
            base_dataset: 基础图像数据集（PetDataset或其子集）
            captions_json: caption文件路径
            vocab: 词汇表，None则自动构建
            max_len: 文本最大长度
        """
        self.base_dataset = base_dataset
        self.max_len = max_len
        
        # 加载captions
        with open(captions_json, 'r', encoding='utf-8') as f:
            self.captions_data = json.load(f)
        
        # 构建caption路径映射
        self.caption_map = {}
        for item in self.captions_data:
            img_path = item['image_path']
            # 只取文件名作为key
            img_name = os.path.basename(img_path)
            self.caption_map[img_name] = item['caption']
        
        # 构建词汇表
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        self.vocab_size = len(self.vocab)
    
    def _build_vocab(self):
        """从所有caption中构建词汇表"""
        word_counts = Counter()
        for caption in self.caption_map.values():
            words = caption.lower().split()
            word_counts.update(words)
        
        # 特殊token
        vocab = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        idx = 4
        for word, count in word_counts.items():
            if count >= 1:  # 所有词都保留，因为数据集小
                vocab[word] = idx
                idx += 1
        return vocab
    
    def _tokenize(self, text):
        """将文本转换为token序列"""
        words = text.lower().split()
        tokens = [self.vocab.get('<start>', 2)]
        for word in words:
            tokens.append(self.vocab.get(word, self.vocab.get('<unk>', 1)))
        tokens.append(self.vocab.get('<end>', 3))
        
        # 截断或填充
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.vocab.get('<pad>', 0)] * (self.max_len - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # 获取对应图像的路径
        if hasattr(self.base_dataset, 'dataset'):
            # Subset情况
            base_idx = self.base_dataset.indices[idx]
            img_path, _ = self.base_dataset.dataset.samples[base_idx]
        else:
            img_path, _ = self.base_dataset.samples[idx]
        
        img_name = os.path.basename(img_path)
        caption = self.caption_map.get(img_name, "a pet")
        caption_tokens = self._tokenize(caption)
        
        return image, caption_tokens, label


def get_image_only_dataloaders(data_dir, batch_size=16, img_size=224, num_workers=0):
    """获取image-only模型的train/val/test dataloader"""
    train_transform = get_image_transforms(img_size, train=True)
    test_transform = get_image_transforms(img_size, train=False)
    
    # 完整数据集
    full_dataset = PetDataset(data_dir, transform=test_transform)
    
    # 划分
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    
    # 训练集使用训练transform
    train_dataset.dataset.transform = train_transform
    
    # 创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, full_dataset.num_classes


def get_fusion_dataloaders(data_dir, captions_json, batch_size=16, img_size=224, num_workers=0):
    """获取图文融合模型的train/val/test dataloader"""
    train_transform = get_image_transforms(img_size, train=True)
    test_transform = get_image_transforms(img_size, train=False)
    
    # 完整图像数据集
    full_image_dataset = PetDataset(data_dir, transform=test_transform)
    
    # 划分
    train_img_dataset, val_img_dataset, test_img_dataset = split_dataset(full_image_dataset)
    
    # 训练集使用训练transform
    train_img_dataset.dataset.transform = train_transform
    
    # 创建图文数据集
    train_dataset = PetCaptionDataset(train_img_dataset, captions_json, vocab=None)
    vocab = train_dataset.vocab  # 复用训练集词汇表
    
    val_dataset = PetCaptionDataset(val_img_dataset, captions_json, vocab=vocab)
    test_dataset = PetCaptionDataset(test_img_dataset, captions_json, vocab=vocab)
    
    # 创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, full_image_dataset.num_classes, vocab
