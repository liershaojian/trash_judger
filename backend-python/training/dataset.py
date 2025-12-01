"""
垃圾分类数据集加载器
支持标准 ImageFolder 格式
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class WasteDataset(Dataset):
    """
    垃圾分类数据集
    
    目录结构:
        data_dir/
        ├── cardboard/
        │   ├── img001.jpg
        │   └── ...
        ├── glass/
        ├── metal/
        └── ...
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 使用 ImageFolder 加载
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def get_data_transforms(img_size=224):
    """
    获取训练和验证的数据增强
    
    Args:
        img_size: 图像大小
    
    Returns:
        train_transform, val_transform
    """
    
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # 稍大一点
        transforms.RandomCrop(img_size),                     # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),              # 随机水平翻转
        transforms.RandomRotation(15),                       # 随机旋转
        transforms.ColorJitter(                              # 颜色抖动
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        transforms.RandomAffine(                             # 随机仿射变换
            degrees=0, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1),                     # 随机擦除
    ])
    
    # 验证集只做基本变换
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return train_transform, val_transform


# 标签映射（细分类 -> 四大类）
CATEGORY_MAPPING = {
    # 可回收物
    'cardboard': 'Recyclable',
    'paper': 'Recyclable',
    'glass': 'Recyclable',
    'metal': 'Recyclable',
    'plastic': 'Recyclable',
    'clothes': 'Recyclable',
    'shoes': 'Recyclable',
    'wood': 'Recyclable',
    
    # 有害垃圾
    'battery': 'Hazardous',
    'medicine': 'Hazardous',
    'lightbulb': 'Hazardous',
    
    # 厨余垃圾
    'food_waste': 'Wet',
    'food': 'Wet',
    'fruit': 'Wet',
    'vegetable': 'Wet',
    
    # 其他垃圾
    'trash': 'Dry',
    'ceramic': 'Dry',
    'tissue': 'Dry',
}


if __name__ == '__main__':
    # 测试数据集加载
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './data/train'
    
    if os.path.exists(data_dir):
        train_transform, _ = get_data_transforms()
        dataset = WasteDataset(data_dir, transform=train_transform)
        print(f"数据集路径: {data_dir}")
        print(f"样本数量: {len(dataset)}")
        print(f"类别数量: {len(dataset.classes)}")
        print(f"类别列表: {dataset.classes}")
    else:
        print(f"目录不存在: {data_dir}")
