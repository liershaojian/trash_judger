"""
垃圾分类模型微调训练脚本
模型: MobileNetV3-Large
数据集: 支持 TrashNet / 华为垃圾分类 / 自定义数据集

使用方法:
    python training/train.py --data_dir ./data --epochs 20 --batch_size 32
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from dataset import WasteDataset, get_data_transforms

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.2f}%")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main(args):
    print("=" * 60)
    print("🗑️  垃圾分类模型微调训练")
    print("=" * 60)
    print(f"模型: MobileNetV3-Large")
    print(f"数据目录: {args.data_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("=" * 60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  运行设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 数据增强
    train_transform, val_transform = get_data_transforms()
    
    # 加载数据集
    print(f"\n📂 加载数据集...")
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"❌ 错误: 训练目录不存在: {train_dir}")
        print(f"\n请按以下结构组织数据:")
        print(f"  {args.data_dir}/")
        print(f"  ├── train/")
        print(f"  │   ├── cardboard/")
        print(f"  │   ├── glass/")
        print(f"  │   ├── metal/")
        print(f"  │   └── ...")
        print(f"  └── val/")
        print(f"      └── (同上)")
        return
    
    train_dataset = WasteDataset(train_dir, transform=train_transform)
    val_dataset = WasteDataset(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    labels = train_dataset.classes
    num_classes = len(labels)
    print(f"✅ 训练集: {len(train_dataset)} 张图片")
    print(f"✅ 验证集: {len(val_dataset)} 张图片")
    print(f"✅ 类别数: {num_classes}")
    print(f"   类别: {labels}")
    
    # 创建模型
    print(f"\n🔧 创建模型...")
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    # 冻结特征提取层（可选，加快训练）
    if args.freeze_backbone:
        print("   冻结 backbone 层")
        for param in model.features.parameters():
            param.requires_grad = False
    
    # 修改分类头
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    print(f"\n🚀 开始训练...")
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\n{'='*40}")
        print(f"Epoch [{epoch+1}/{args.epochs}]  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*40}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📊 Epoch {epoch+1} 结果:")
        print(f"   Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, 'waste_classifier.pt')
            os.makedirs(args.output_dir, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'labels': labels,
                'num_classes': num_classes,
                'model_name': 'MobileNetV3-Large',
                'best_acc': best_acc,
                'epoch': epoch + 1,
                'training_args': vars(args)
            }, save_path)
            print(f"   💾 保存最佳模型 (Acc: {best_acc:.2f}%)")
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 同时更新 labels.json
    labels_path = os.path.join(args.output_dir, 'labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"🎉 训练完成!")
    print(f"   最佳验证准确率: {best_acc:.2f}%")
    print(f"   模型保存位置: {os.path.join(args.output_dir, 'waste_classifier.pt')}")
    print(f"   训练历史: {history_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='垃圾分类模型微调训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data_split',
                        help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数 (Windows 建议设为 0)')
    
    # 模型参数
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='是否冻结 backbone 层')
    
    args = parser.parse_args()
    main(args)
