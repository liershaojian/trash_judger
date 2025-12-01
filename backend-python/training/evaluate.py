"""
模型评估脚本
计算准确率、混淆矩阵、分类报告等指标
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)

from dataset import WasteDataset, get_data_transforms

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("提示: 安装 matplotlib 和 seaborn 可生成可视化图表")


def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    labels = checkpoint.get('labels', [])
    num_classes = checkpoint.get('num_classes', len(labels))
    
    # 创建模型
    model = mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, labels


def evaluate(model, dataloader, device):
    """评估模型，返回预测结果和真实标签"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, labels, save_path):
    """绘制混淆矩阵"""
    if not HAS_PLOT:
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")


def plot_training_history(history_path, save_path):
    """绘制训练曲线"""
    if not HAS_PLOT:
        return
    
    if not os.path.exists(history_path):
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss 曲线')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy 曲线
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('准确率曲线')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 训练曲线已保存: {save_path}")


def main(args):
    print("=" * 60)
    print("🔍 垃圾分类模型评估")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {device}")
    
    # 加载模型
    print(f"\n📦 加载模型: {args.model_path}")
    model, labels = load_model(args.model_path, device)
    print(f"   类别数: {len(labels)}")
    print(f"   类别: {labels}")
    
    # 加载数据
    print(f"\n📂 加载验证数据: {args.data_dir}")
    _, val_transform = get_data_transforms()
    val_dataset = WasteDataset(args.data_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    print(f"   样本数: {len(val_dataset)}")
    
    # 评估
    print(f"\n🚀 开始评估...")
    preds, true_labels, probs = evaluate(model, val_loader, device)
    
    # 计算指标
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average='weighted'
    )
    
    print(f"\n{'='*60}")
    print(f"📊 评估结果")
    print(f"{'='*60}")
    print(f"准确率 (Accuracy):  {accuracy*100:.2f}%")
    print(f"精确率 (Precision): {precision*100:.2f}%")
    print(f"召回率 (Recall):    {recall*100:.2f}%")
    print(f"F1 分数 (F1-Score): {f1*100:.2f}%")
    
    # 分类报告
    print(f"\n📋 分类报告:")
    print("-" * 60)
    report = classification_report(true_labels, preds, target_names=labels)
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, preds)
    print(f"\n📊 混淆矩阵:")
    print(cm)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存评估报告
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    report_dict = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'classification_report': classification_report(
            true_labels, preds, target_names=labels, output_dict=True
        )
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估报告已保存: {report_path}")
    
    # 绘制图表
    if HAS_PLOT:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, labels, cm_path)
        
        history_path = os.path.join(os.path.dirname(args.model_path), 'training_history.json')
        if os.path.exists(history_path):
            curve_path = os.path.join(args.output_dir, 'training_curves.png')
            plot_training_history(history_path, curve_path)
    
    print(f"\n{'='*60}")
    print(f"✅ 评估完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='垃圾分类模型评估')
    
    parser.add_argument('--model_path', type=str, default='./models/waste_classifier.pt',
                        help='模型权重路径')
    parser.add_argument('--data_dir', type=str, default='./data/val',
                        help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    
    args = parser.parse_args()
    main(args)
