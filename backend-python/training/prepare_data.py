"""
数据集准备脚本
帮助下载和组织垃圾分类数据集
"""

import os
import sys
import shutil
import random
from pathlib import Path


def download_trashnet():
    """下载 TrashNet 数据集"""
    print("=" * 60)
    print("📥 TrashNet 数据集下载指南")
    print("=" * 60)
    print("""
TrashNet 是一个经典的垃圾分类数据集，包含 6 类垃圾图片。

下载方式:
---------
1. 访问 GitHub: https://github.com/garythung/trashnet

2. 下载 dataset-resized.zip (已调整大小的版本，约 100MB)
   直接链接: https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip

3. 解压到 data 目录:
   data/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   └── trash/

4. 运行本脚本的 split 命令划分训练/验证集:
   python prepare_data.py split --data_dir ./data --split_ratio 0.8

类别说明:
---------
- cardboard: 纸板 (403张) -> 可回收物
- glass: 玻璃 (501张) -> 可回收物
- metal: 金属 (410张) -> 可回收物
- paper: 纸张 (594张) -> 可回收物
- plastic: 塑料 (482张) -> 可回收物
- trash: 其他垃圾 (137张) -> 干垃圾

总计: 2,527 张图片
""")


def download_garbage12():
    """下载 Garbage Classification 12类数据集"""
    print("=" * 60)
    print("📥 Garbage Classification (12类) 数据集下载指南")
    print("=" * 60)
    print("""
这是一个更大的垃圾分类数据集，包含约 15,000 张图片，12 个类别。

下载方式:
---------
1. 访问 Kaggle: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

2. 下载数据集 (需要 Kaggle 账号)

3. 解压到 data 目录，结构如下:
   data/
   ├── battery/
   ├── biological/
   ├── cardboard/
   ├── clothes/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   ├── shoes/
   ├── trash/
   └── ...

4. 运行 split 命令划分数据集

类别映射到四大类:
-----------------
- 可回收物: cardboard, glass, metal, paper, plastic, clothes, shoes
- 有害垃圾: battery
- 厨余垃圾: biological (food waste)
- 其他垃圾: trash
""")


def download_huawei():
    """下载华为垃圾分类数据集"""
    print("=" * 60)
    print("📥 华为垃圾分类数据集下载指南")
    print("=" * 60)
    print("""
华为云 AI Gallery 提供的中国标准四分类垃圾数据集。

下载方式:
---------
1. 访问华为云 AI Gallery:
   https://developer.huaweicloud.com/develop/aigallery/dataset/detail?id=xxx
   (搜索 "垃圾分类")

2. 或者使用 ModelArts 直接下载

3. 解压后组织为以下结构:
   data/
   ├── train/
   │   ├── recyclable/     # 可回收物
   │   ├── hazardous/      # 有害垃圾
   │   ├── wet/            # 厨余垃圾
   │   └── dry/            # 其他垃圾
   └── val/
       └── (同上)

特点:
-----
- 符合中国国标四分类
- 图片来源于真实生活场景
- 约 14,000+ 张图片
""")


def split_dataset(data_dir, output_dir, split_ratio=0.8):
    """
    将数据集划分为训练集和验证集
    
    Args:
        data_dir: 原始数据目录（包含各类别子目录）
        output_dir: 输出目录
        split_ratio: 训练集比例
    """
    print(f"\n📂 划分数据集: {data_dir}")
    print(f"   训练集比例: {split_ratio}")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # 获取所有类别
    categories = [d.name for d in data_path.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    if not categories:
        print(f"❌ 错误: 未找到类别目录")
        return
    
    print(f"   发现 {len(categories)} 个类别: {categories}")
    
    # 创建输出目录
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    total_train = 0
    total_val = 0
    
    for category in categories:
        src_dir = data_path / category
        
        # 获取所有图片
        images = list(src_dir.glob('*.[jJ][pP][gG]')) + \
                 list(src_dir.glob('*.[pP][nN][gG]')) + \
                 list(src_dir.glob('*.[jJ][pP][eE][gG]'))
        
        if not images:
            print(f"   ⚠️ {category}: 无图片")
            continue
        
        # 随机打乱
        random.shuffle(images)
        
        # 划分
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 创建目录
        train_cat_dir = train_dir / category
        val_cat_dir = val_dir / category
        train_cat_dir.mkdir(parents=True, exist_ok=True)
        val_cat_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        for img in train_images:
            shutil.copy2(img, train_cat_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_cat_dir / img.name)
        
        total_train += len(train_images)
        total_val += len(val_images)
        print(f"   ✅ {category}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\n✅ 划分完成!")
    print(f"   训练集: {total_train} 张")
    print(f"   验证集: {total_val} 张")
    print(f"   输出目录: {output_path}")


def check_dataset(data_dir):
    """检查数据集结构"""
    print(f"\n📂 检查数据集: {data_dir}")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    # 检查是否有 train/val 子目录
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    if train_dir.exists() and val_dir.exists():
        print("✅ 已划分为 train/val 结构")
        dirs_to_check = [('train', train_dir), ('val', val_dir)]
    else:
        print("ℹ️ 未划分，检查原始结构")
        dirs_to_check = [('root', data_path)]
    
    for name, dir_path in dirs_to_check:
        print(f"\n[{name}]")
        categories = sorted([d.name for d in dir_path.iterdir() 
                            if d.is_dir() and not d.name.startswith('.')])
        
        total = 0
        for cat in categories:
            cat_dir = dir_path / cat
            images = list(cat_dir.glob('*.[jJ][pP][gG]')) + \
                     list(cat_dir.glob('*.[pP][nN][gG]')) + \
                     list(cat_dir.glob('*.[jJ][pP][eE][gG]'))
            total += len(images)
            print(f"  {cat}: {len(images)} 张")
        
        print(f"  总计: {total} 张")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='垃圾分类数据集准备工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # download 命令
    download_parser = subparsers.add_parser('download', help='显示数据集下载指南')
    download_parser.add_argument('--dataset', type=str, default='trashnet',
                                 choices=['trashnet', 'garbage12', 'huawei'],
                                 help='数据集名称')
    
    # split 命令
    split_parser = subparsers.add_parser('split', help='划分训练/验证集')
    split_parser.add_argument('--data_dir', type=str, required=True,
                              help='原始数据目录')
    split_parser.add_argument('--output_dir', type=str, default=None,
                              help='输出目录（默认与 data_dir 相同）')
    split_parser.add_argument('--split_ratio', type=float, default=0.8,
                              help='训练集比例')
    
    # check 命令
    check_parser = subparsers.add_parser('check', help='检查数据集结构')
    check_parser.add_argument('--data_dir', type=str, required=True,
                              help='数据目录')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        if args.dataset == 'trashnet':
            download_trashnet()
        elif args.dataset == 'garbage12':
            download_garbage12()
        elif args.dataset == 'huawei':
            download_huawei()
    elif args.command == 'split':
        output_dir = args.output_dir or args.data_dir
        split_dataset(args.data_dir, output_dir, args.split_ratio)
    elif args.command == 'check':
        check_dataset(args.data_dir)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python prepare_data.py download --dataset trashnet")
        print("  python prepare_data.py split --data_dir ./raw_data --output_dir ./data")
        print("  python prepare_data.py check --data_dir ./data")


if __name__ == '__main__':
    main()
