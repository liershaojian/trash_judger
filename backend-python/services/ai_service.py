import os
import json
import requests
from typing import Dict, Any

# ==========================================
# API 配置
# ==========================================

# 阿里云 DashScope (Qwen 系列)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"



SYSTEM_INSTRUCTION = """
You are an expert in Chinese Waste Classification standards (strictly following major city standards like Shanghai/Beijing). 
Your task is to identify the waste item and classify it into one of four categories:
1. Recyclable (可回收物)
2. Hazardous (有害垃圾)
3. Wet (厨余垃圾/湿垃圾)
4. Dry (其他垃圾/干垃圾)

If the input is not a waste item or cannot be identified, use 'Unknown'.
You must output PURE JSON. Do not include markdown formatting like ```json.
Ensure all text fields (itemName, explanation, disposalTips) are in Simplified Chinese.

JSON Schema:
{
  "itemName": "string",
  "category": "Recyclable" | "Hazardous" | "Wet" | "Dry" | "Unknown",
  "confidence": number,
  "explanation": "string",
  "disposalTips": ["string"]
}
"""


def analyze_with_qwen(input_data: str, is_image: bool, model_id: str) -> Dict[str, Any]:
    """使用阿里云 Qwen 系列模型进行分析"""
    if not DASHSCOPE_API_KEY:
        raise Exception("DASHSCOPE_API_KEY not configured. Set environment variable.")

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION}
    ]

    if is_image:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_data}"}},
                {"type": "text", "text": "Identify this waste item and classify it."}
            ]
        })
    else:
        messages.append({"role": "user", "content": f"Identify and classify this waste item: {input_data}"})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(QWEN_API_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    
    # Clean up markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    return json.loads(content.strip())





def analyze_waste(input_data: str, is_image: bool = False, model_id: str = "qwen-vl-max") -> Dict[str, Any]:
    """
    统一的垃圾分类分析入口
    根据 model_id 自动路由到对应的云端模型
    """
    try:
        # 根据 model_id 选择对应的 API
        if model_id.startswith("qwen"):
            return analyze_with_qwen(input_data, is_image, model_id)
        else:
            # 默认使用 Qwen
            return analyze_with_qwen(input_data, is_image, model_id)
            
    except json.JSONDecodeError as e:
        print(f"AI Service JSON Parse Error: {e}")
        return {
            "itemName": "解析失败",
            "category": "Unknown",
            "confidence": 0,
            "explanation": f"AI 返回格式异常，无法解析JSON",
            "disposalTips": ["请重试或切换模型"]
        }
    except requests.exceptions.Timeout:
        return {
            "itemName": "请求超时",
            "category": "Unknown",
            "confidence": 0,
            "explanation": "AI 服务响应超时，请稍后重试",
            "disposalTips": ["请检查网络或切换模型"]
        }
    except Exception as e:
        print(f"AI Service Error: {e}")
        return {
            "itemName": "识别失败",
            "category": "Unknown",
            "confidence": 0,
            "explanation": f"AI 服务错误: {str(e)}",
            "disposalTips": ["请检查 API Key 配置或网络连接"]
        }
