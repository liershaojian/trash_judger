from pydantic import BaseModel
from typing import List, Optional

class WasteAnalysisRequest(BaseModel):
    input: str
    isImage: bool = False
    modelId: str = "qwen-vl-max"

class WasteAnalysisResult(BaseModel):
    itemName: str
    category: str
    confidence: float
    explanation: str
    disposalTips: List[str]
