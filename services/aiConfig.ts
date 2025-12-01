import { WasteCategory } from "../types";

// ==========================================
// 支持的模型列表
// ==========================================

export const SUPPORTED_MODELS = {
  // 本地模型（离线推理，无需联网）
  LOCAL: [
    { id: "local-mobilenet", name: "📱 MobileNetV3-Large (本地)" },
  ],
  // 云端大模型（在线推理）

  QWEN: [
    { id: "qwen-vl-max", name: "👁️ Qwen-VL-Max (图像识别)" },
    { id: "qwen-plus", name: "💬 Qwen-Plus (纯文本)" }
  ]
};

// 默认使用本地模型
export const DEFAULT_MODEL_ID = "local-mobilenet";
