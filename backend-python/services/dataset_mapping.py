# 数据集映射配置文件
# 用于将各种数据集的英文/中文目录名映射到四大类，并提供中文展示名称

# Kaggle Garbage Classification (12 classes) 映射
# 目录名 -> (中文名称, 四大类)
KAGGLE_12_MAPPING = {
    # 纸类
    'paper': ('纸张', 'Recyclable'),
    'cardboard': ('纸板/纸箱', 'Recyclable'),
    
    # 塑料
    'plastic': ('塑料制品', 'Recyclable'),
    
    # 金属
    'metal': ('金属制品', 'Recyclable'),
    
    # 玻璃 (归并为玻璃制品)
    'glass': ('玻璃制品', 'Recyclable'),
    'green-glass': ('绿色玻璃', 'Recyclable'),
    'brown-glass': ('棕色玻璃', 'Recyclable'),
    'white-glass': ('透明玻璃', 'Recyclable'),
    
    # 衣物鞋帽
    'clothes': ('旧衣物', 'Recyclable'),
    'shoes': ('旧鞋子', 'Recyclable'),
    
    # 电池
    'battery': ('废电池', 'Hazardous'),
    'batteries': ('废电池', 'Hazardous'),
    
    # 厨余
    'biological': ('厨余垃圾', 'Wet'),
    
    # 其他
    'trash': ('其他垃圾', 'Dry'),
}

# 通用中文映射 (用于模糊匹配或直接中文目录)
CHINESE_GENERIC_MAPPING = {
    # 可回收物
    '易拉罐': 'Recyclable',
    '塑料瓶': 'Recyclable',
    '报纸': 'Recyclable',
    '书本': 'Recyclable',
    '玻璃瓶': 'Recyclable',
    '旧衣服': 'Recyclable',
    
    # 有害垃圾
    '电池': 'Hazardous',
    '药': 'Hazardous',
    '灯泡': 'Hazardous',
    '油漆': 'Hazardous',
    
    # 厨余垃圾
    '剩饭': 'Wet',
    '果皮': 'Wet',
    '菜叶': 'Wet',
    '骨头': 'Wet', # 注：大骨头通常是干垃圾，这里泛指小骨头
    '鸡蛋壳': 'Wet',
    
    # 其他垃圾
    '烟蒂': 'Dry',
    '餐巾纸': 'Dry',
    '卫生纸': 'Dry',
    '陶瓷': 'Dry',
}

def get_mapped_info(label):
    """
    获取标签的映射信息
    Args:
        label: 原始标签 (目录名)
    Returns:
        (chinese_name, category)
    """
    label_lower = label.lower()
    
    # 1. 尝试 Kaggle 12类映射
    if label_lower in KAGGLE_12_MAPPING:
        return KAGGLE_12_MAPPING[label_lower]
        
    # 2. 尝试中文直接映射
    # 如果 label 本身就是中文且在映射表中
    if label in CHINESE_GENERIC_MAPPING:
        return (label, CHINESE_GENERIC_MAPPING[label])
        
    # 3. 尝试模糊匹配中文
    for key, category in CHINESE_GENERIC_MAPPING.items():
        if key in label:
            return (label, category)
            
    # 4. 默认回退
    return (label, 'Unknown')
