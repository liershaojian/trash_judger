"""
模型下载脚本
自动下载预训练的垃圾多分类模型权重
"""

import os
import json
import sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'waste_classifier.pt')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')

# 12 分类标签（细分类 -> 四大类）
WASTE_12_LABELS = [
    'cardboard',    # 纸板 -> 可回收
    'glass',        # 玻璃 -> 可回收
    'metal',        # 金属 -> 可回收
    'paper',        # 纸张 -> 可回收
    'plastic',      # 塑料 -> 可回收
    'trash',        # 其他 -> 干垃圾
    'battery',      # 电池 -> 有害
    'clothes',      # 衣物 -> 可回收
    'food_waste',   # 厨余 -> 湿垃圾
    'shoes',        # 鞋子 -> 可回收
    'wood',         # 木材 -> 可回收
    'ceramic',      # 陶瓷 -> 干垃圾
]

# TrashNet 6分类标签（基础版）
TRASHNET_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def create_model_dir():
    """创建模型目录"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"✅ 创建目录: {MODEL_DIR}")


def create_default_labels():
    """创建默认标签文件（12分类）"""
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(WASTE_12_LABELS, f, indent=2)
    print(f"✅ 创建标签文件: {LABELS_PATH}")
    print(f"   包含 {len(WASTE_12_LABELS)} 个细分类别")


def download_from_huggingface():
    """
    从 HuggingFace 下载预训练模型
    注意：这里提供的是示例代码，实际权重需要从 HuggingFace 获取
    """
    print("\n📦 方案1: 使用 HuggingFace transformers 模型")
    print("=" * 50)
    print("运行以下命令安装依赖：")
    print("  pip install transformers pillow")
    print("\n然后可以直接使用：")
    print("""
from transformers import pipeline
classifier = pipeline("image-classification", 
                      model="yangy50/garbage-classification")
result = classifier("test.jpg")
""")


def download_mobilenet_pretrained():
    """
    下载 MobileNetV3-Large 预训练模型
    修改为 12 分类头
    """
    print("\n📦 下载 MobileNetV3-Large 预训练模型")
    print("=" * 50)
    
    try:
        import torch
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        
        print("正在下载 MobileNetV3-Large 预训练权重...")
        print("模型参数量: 5.4M")
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # 修改分类头为 12 分类
        num_classes = len(WASTE_12_LABELS)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features,
            num_classes
        )
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'labels': WASTE_12_LABELS,
            'num_classes': num_classes,
            'model_name': 'MobileNetV3-Large',
            'note': 'ImageNet 预训练 + 12类垃圾分类头（未微调）'
        }, MODEL_PATH)
        
        # 同时更新标签文件
        with open(LABELS_PATH, 'w', encoding='utf-8') as f:
            json.dump(WASTE_12_LABELS, f, indent=2)
        
        print(f"✅ 模型已保存: {MODEL_PATH}")
        print(f"✅ 标签已更新: {LABELS_PATH}")
        print(f"   模型: MobileNetV3-Large")
        print(f"   分类数量: {num_classes} 类")
        print(f"\n⚠️  注意: 这是 ImageNet 预训练权重")
        print(f"   分类头已修改为 12 类，但尚未在垃圾数据集上微调")
        print(f"   建议后续使用垃圾分类数据集进行微调训练以获得更好效果")
        
        return True
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install torch torchvision")
        return False


def print_training_guide():
    """打印微调训练指南"""
    print("\n" + "=" * 60)
    print("📚 如何获得更好的垃圾分类模型？")
    print("=" * 60)
    print("""
方案 A: 使用现成的在线模型权重
----------------------------------------
1. 访问 https://universe.roboflow.com
2. 搜索 "garbage classification" 或 "waste detection"
3. 下载 PyTorch 格式的权重文件
4. 将 .pt 文件放到 backend-python/models/ 目录

方案 B: 自己微调训练（推荐用于论文）
----------------------------------------
1. 下载数据集:
   - TrashNet: https://github.com/garythung/trashnet
   - 华为垃圾分类: 华为云 AI Gallery 搜索

2. 准备数据目录结构:
   data/
   ├── train/
   │   ├── cardboard/
   │   ├── glass/
   │   ├── metal/
   │   ├── paper/
   │   ├── plastic/
   │   └── trash/
   └── val/
       └── (同上)

3. 运行训练脚本:
   python training/train.py --data_dir ./data --epochs 20

4. 训练完成后权重会自动保存到 models/ 目录

方案 C: 使用大模型 API（当前项目已支持）
----------------------------------------
项目已集成 Qwen/Gemini 在线模型，适合:
- 复杂/未知垃圾识别
- 文本描述查询
- 混杂垃圾分析
""")


def main():
    print("=" * 60)
    print("🗑️  垃圾分类模型下载工具")
    print("=" * 60)
    
    # 1. 创建目录
    create_model_dir()
    
    # 2. 创建标签文件
    create_default_labels()
    
    # 3. 下载模型
    print("\n选择下载方式:")
    print("  1. 下载 MobileNetV3 预训练权重（推荐，可快速测试）")
    print("  2. 查看 HuggingFace 模型使用方法")
    print("  3. 查看微调训练指南")
    print("  4. 全部执行")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请输入选项 (1/2/3/4) [默认: 1]: ").strip() or "1"
    
    if choice == "1":
        download_mobilenet_pretrained()
    elif choice == "2":
        download_from_huggingface()
    elif choice == "3":
        print_training_guide()
    elif choice == "4":
        download_mobilenet_pretrained()
        download_from_huggingface()
        print_training_guide()
    else:
        print("无效选项，执行默认下载...")
        download_mobilenet_pretrained()
    
    print("\n✅ 完成！现在可以启动后端服务测试本地模型推理。")


if __name__ == '__main__':
    main()
