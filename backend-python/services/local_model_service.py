"""
本地模型推理服务
使用预训练的 MobileNetV3 进行垃圾多分类
支持 CPU 和 GPU 推理

分类体系：
- 细分类：12+ 种常见垃圾类型
- 汇总到四大类：可回收物、有害垃圾、厨余垃圾、其他垃圾
"""

import os
import json
import base64
import io
from typing import Dict, Any, Optional, List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .dataset_mapping import get_mapped_info

# 模型文件路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'waste_classifier.pt')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')




class WasteClassifier:
    """本地垃圾多分类模型 - MobileNetV3-Large"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.labels = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self._load_model()
    
    def _load_model(self):
        """加载模型和标签"""
        try:
            # 加载标签
            if os.path.exists(LABELS_PATH):
                with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                    self.labels = json.load(f)
            else:
                # 使用默认标签 (如果 labels.json 不存在)
                # 注意：这里我们不再硬编码 DEFAULT_LABELS，而是依赖模型输出的索引
                # 但为了兼容，如果真的没有 labels.json，我们还是需要一个默认列表
                # 暂时保留一个最小集，但强烈建议用户训练后生成 labels.json
                self.labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
            
            num_classes = len(self.labels)
            
            # 加载模型
            if os.path.exists(MODEL_PATH):
                # 加载自定义训练的模型
                checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
                
                # 创建 MobileNetV3-Large 模型
                from torchvision.models import mobilenet_v3_large
                self.model = mobilenet_v3_large(weights=None)
                
                # 修改分类头
                self.model.classifier[-1] = nn.Linear(
                    self.model.classifier[-1].in_features, 
                    num_classes
                )
                
                # 加载权重
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'labels' in checkpoint:
                        self.labels = checkpoint['labels']
                else:
                    self.model.load_state_dict(checkpoint)
                    
                print(f"[Local Model] ✅ 已加载自定义模型: {MODEL_PATH}")
            else:
                # 使用预训练的 MobileNetV3-Large
                from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
                self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
                
                # 修改分类头
                self.model.classifier[-1] = nn.Linear(
                    self.model.classifier[-1].in_features,
                    num_classes
                )
                
                print(f"[Local Model] ⚠️ 使用 MobileNetV3-Large 预训练权重（未针对垃圾分类微调）")
                print(f"[Local Model] 请运行 python download_model.py 下载专用权重")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"[Local Model] 🚀 模型: MobileNetV3-Large (5.4M 参数)")
            print(f"[Local Model] 运行设备: {self.device}")
            print(f"[Local Model] 分类数量: {num_classes} 类")
            print(f"[Local Model] 类别列表: {self.labels}")
            
        except Exception as e:
            print(f"[Local Model] ❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def predict(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """
        对图片进行多分类预测
        
        Args:
            image: PIL 图片
            top_k: 返回置信度最高的 k 个结果
        
        Returns:
            包含 top_k 预测结果的字典
        """
        if self.model is None:
            raise Exception("模型未加载，请检查模型文件")
        
        # 预处理
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # 获取 top_k 结果
            top_probs, top_indices = torch.topk(probs, min(top_k, len(self.labels)), dim=1)
        
        # 构建结果
        predictions = []
        for i in range(top_probs.size(1)):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            label = self.labels[idx] if idx < len(self.labels) else 'trash'
            
            # 使用新的映射逻辑
            chinese_name, category = get_mapped_info(label)
            
            predictions.append({
                'label': label,
                'label_cn': chinese_name,
                'category': category,
                'confidence': prob
            })
        
        # 主预测结果
        top1 = predictions[0]
        
        return {
            'top1': top1,
            'top_k': predictions,
            'all_probs': {self.labels[i]: probs[0][i].item() for i in range(len(self.labels))}
        }

    def _get_disposal_tips(self, category):
        """获取投放建议"""
        tips = {
            'Recyclable': [
                '请投放至蓝色可回收物垃圾桶',
                '保持物品干燥清洁',
                '纸类请折叠整齐，塑料瓶请压扁',
                '玻璃制品请注意防碎'
            ],
            'Hazardous': [
                '请投放至红色有害垃圾桶',
                '电池、灯泡等请轻拿轻放',
                '药品请保留原包装',
                '切勿与其他垃圾混合'
            ],
            'Wet': [
                '请投放至绿色厨余垃圾桶',
                '沥干水分后投放',
                '去除包装袋、牙签等杂物',
                '大骨头属于干垃圾'
            ],
            'Dry': [
                '请投放至灰色其他垃圾桶',
                '尽量沥干水分',
                '难以辨别的垃圾可投放此类',
                '注意不要混入有害垃圾'
            ],
            'Unknown': [
                '建议咨询当地垃圾分类指南',
                '或使用在线AI模型进行识别'
            ]
        }
        return tips.get(category, tips['Unknown'])


# 全局模型实例（单例模式）
_classifier_instance: Optional[WasteClassifier] = None


def get_classifier() -> WasteClassifier:
    """获取分类器单例"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = WasteClassifier()
    return _classifier_instance


def analyze_waste_local(input_data: str, is_image: bool = True) -> Dict[str, Any]:
    """
    本地模型推理入口（多分类）
    
    Args:
        input_data: Base64 图片数据 或 文本描述
        is_image: 是否为图片输入
    
    Returns:
        分类结果字典，包含细分类和四大类
    """
    classifier = get_classifier()
    
    if is_image:
        # 解码 Base64 图片
        try:
            image_bytes = base64.b64decode(input_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return {
                'itemName': '图片解析失败',
                'category': 'Unknown',
                'confidence': 0,
                'explanation': f'无法解析图片数据: {str(e)}',
                'disposalTips': ['建议咨询当地垃圾分类指南']
            }
        
        # 进行预测
        try:
            result = classifier.predict(image, top_k=3)
            
            top1 = result['top1']
            top_k = result['top_k']
            
            category = top1['category']
            confidence = top1['confidence']
            item_name = top1['label_cn']
            raw_label = top1['label']
            
            # 生成解释（包含 top-3 结果）
            # 重新获取中文类别名称
            category_cn_map = {
                'Recyclable': '可回收物',
                'Hazardous': '有害垃圾',
                'Wet': '厨余垃圾',
                'Dry': '其他垃圾',
                'Unknown': '未知类别'
            }
            category_cn_text = category_cn_map.get(category, '未知类别')
            
            explanation = f"经本地AI模型分析，该物品最可能是「{item_name}」，属于{category_cn_text}。"
            
            if len(top_k) > 1:
                other_preds = [f"{p['label_cn']}({p['confidence']*100:.1f}%)" for p in top_k[1:]]
                explanation += f"\n其他可能：{', '.join(other_preds)}"
            
            if confidence < 0.6:
                explanation += "\n⚠️ 置信度较低，建议使用在线大模型进行二次确认。"
            
            return {
                'itemName': item_name,
                'category': category,
                'confidence': round(confidence, 3),
                'explanation': explanation,
                'explanation': explanation,
                # 重新定义 DISPOSAL_TIPS (因为之前删除了)
                'disposalTips': classifier._get_disposal_tips(category),
                'modelType': 'local',
                'rawLabel': raw_label,
                # 额外返回多分类结果供前端展示
                'multiClassResults': [
                    {
                        'name': p['label_cn'],
                        'category': p['category'],
                        'confidence': round(p['confidence'], 3)
                    }
                    for p in top_k
                ]
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'itemName': '识别失败',
                'category': 'Unknown', 
                'confidence': 0,
                'explanation': f'本地模型推理出错: {str(e)}',
                'disposalTips': ['建议咨询当地垃圾分类指南']
            }
    else:
        # 文本输入 - 本地模型不支持
        return {
            'itemName': input_data,
            'category': 'Unknown',
            'confidence': 0,
            'explanation': '本地模型仅支持图片识别，文本查询请使用在线模型。',
            'disposalTips': ['建议切换到在线模型进行文本查询']
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 50)
    print("测试本地多分类模型服务")
    print("=" * 50)
    classifier = get_classifier()
    print(f"\n模型加载状态: {'✅ 成功' if classifier.model else '❌ 失败'}")
    print(f"支持类别数: {len(classifier.labels) if classifier.labels else 0}")
    print(f"类别列表: {classifier.labels}")
