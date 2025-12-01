from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schemas import WasteAnalysisRequest, WasteAnalysisResult
from services.ai_service import analyze_waste as analyze_waste_cloud
from services.local_model_service import analyze_waste_local
import models


router = APIRouter(
    prefix="/api/waste",
    tags=["waste"]
)

# 本地模型 ID 列表
LOCAL_MODEL_IDS = ['local-mobilenet', 'local-resnet']


@router.post("/analyze", response_model=WasteAnalysisResult)
async def analyze_waste_item(request: WasteAnalysisRequest, db: Session = Depends(get_db)):
    """
    垃圾分类识别接口
    支持本地模型和云端大模型两种模式
    """
    
    # 根据 modelId 选择推理方式
    if request.modelId in LOCAL_MODEL_IDS:
        # 本地模型推理
        result = analyze_waste_local(request.input, request.isImage)
    else:
        # 云端大模型推理 (Qwen/Gemini)
        result = analyze_waste_cloud(request.input, request.isImage, request.modelId)
    
    # Save record to database
    try:
        # Avoid saving large base64 strings to TEXT column
        image_data = "Base64 Image Data" if request.isImage else None
        
        db_record = models.IdentificationRecord(
            waste_name=result['itemName'],
            category=result['category'],
            confidence=result['confidence'],
            explanation=result['explanation'],
            image_url=image_data,
            user_id=None # Anonymous for now
        )
        db.add(db_record)
        db.commit()
    except Exception as e:
        print(f"Warning: Failed to save record to DB: {e}")
        # Continue even if DB save fails
    
    return result
