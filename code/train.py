import os
import json
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_image_only_dataloaders, get_fusion_dataloaders
from model import build_image_only_model, build_fusion_model


def train_one_epoch(model, dataloader, criterion, optimizer, device, is_fusion=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if is_fusion:
            images, captions, labels = batch
            images = images.to(device)
            captions = captions.to(device)
            labels = labels.to(device)
            outputs = model(images, captions)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, is_fusion=False):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        if is_fusion:
            images, captions, labels = batch
            images = images.to(device)
            captions = captions.to(device)
            labels = labels.to(device)
            outputs = model(images, captions)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_image_only(data_dir, save_dir, epochs=10, batch_size=16, lr=1e-3, 
                     optimizer_type='adam', freeze_backbone=False, device=None):
    """训练image-only模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "..", "checkpoints")
    log_dir = os.path.join(save_dir, "..", "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader, num_classes = get_image_only_dataloaders(
        data_dir, batch_size=batch_size
    )
    
    # 创建模型
    print(f"\n创建image-only模型，类别数: {num_classes}")
    model = build_image_only_model(num_classes, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # 训练记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\n开始训练，共 {epochs} 个epoch...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_fusion=False
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, is_fusion=False
        )
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
            }, os.path.join(checkpoint_dir, 'image_only_best.pth'))
            print(f"保存最佳模型 (val_acc: {val_acc:.2f}%)")
    
    elapsed = time.time() - start_time
    print(f"\n训练完成！耗时: {elapsed/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    # 保存训练日志
    log_data = {
        'model_type': 'image_only',
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'optimizer': optimizer_type,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }
    with open(os.path.join(log_dir, 'image_only_log.json'), 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         os.path.join(log_dir, 'image_only_curves.png'),
                         title='Image-Only Model Training Curves')
    
    return model, log_data


def train_fusion(data_dir, captions_json, save_dir, epochs=10, batch_size=16, lr=1e-3,
                 optimizer_type='adam', use_gru=True, freeze_image_backbone=False, device=None):
    """训练图文融合模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "..", "checkpoints")
    log_dir = os.path.join(save_dir, "..", "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader, num_classes, vocab = get_fusion_dataloaders(
        data_dir, captions_json, batch_size=batch_size
    )
    vocab_size = len(vocab)
    pad_idx = vocab.get('<pad>', 0)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    print(f"\n创建图文融合模型，类别数: {num_classes}")
    model = build_fusion_model(
        num_classes=num_classes,
        vocab_size=vocab_size,
        use_gru=use_gru,
        freeze_image_backbone=freeze_image_backbone,
        pad_idx=pad_idx
    )
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # 训练记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\n开始训练，共 {epochs} 个epoch...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_fusion=True
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, is_fusion=True
        )
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'vocab_size': vocab_size,
                'vocab': vocab,
                'use_gru': use_gru,
            }, os.path.join(checkpoint_dir, 'fusion_best.pth'))
            print(f"保存最佳模型 (val_acc: {val_acc:.2f}%)")
    
    elapsed = time.time() - start_time
    print(f"\n训练完成！耗时: {elapsed/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (epoch {best_epoch})")
    
    # 保存训练日志
    log_data = {
        'model_type': 'image_text_fusion',
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'optimizer': optimizer_type,
        'use_gru': use_gru,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }
    with open(os.path.join(log_dir, 'fusion_log.json'), 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(log_dir, 'fusion_curves.png'),
                         title='Image-Text Fusion Model Training Curves')
    
    return model, log_data, vocab


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path, title='Training Curves'):
    """绘制训练曲线并保存"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy曲线
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pet classification models")
    parser.add_argument("--model_type", type=str, default="both", 
                        choices=["image_only", "fusion", "both"],
                        help="Which model to train")
    parser.add_argument("--data_dir", type=str, default="../data", help="Dataset directory")
    parser.add_argument("--captions", type=str, default="../captions/captions.json", 
                        help="Captions JSON file (for fusion model)")
    parser.add_argument("--save_dir", type=str, default="../results", help="Save directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--use_gru", action="store_true", default=True, 
                        help="Use GRU for text encoder (default: True)")
    parser.add_argument("--freeze_backbone", action="store_true", 
                        help="Freeze image backbone")
    
    args = parser.parse_args()
    
    if args.model_type in ["image_only", "both"]:
        print("=" * 50)
        print("训练 Image-Only 模型")
        print("=" * 50)
        train_image_only(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_type=args.optimizer,
            freeze_backbone=args.freeze_backbone
        )
    
    if args.model_type in ["fusion", "both"]:
        print("\n" + "=" * 50)
        print("训练 Image-Text Fusion 模型")
        print("=" * 50)
        train_fusion(
            data_dir=args.data_dir,
            captions_json=args.captions,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_type=args.optimizer,
            use_gru=args.use_gru,
            freeze_image_backbone=args.freeze_backbone
        )
