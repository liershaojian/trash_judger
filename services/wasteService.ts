import { WasteAnalysisResult, WasteCategory } from "../types";
import { DEFAULT_MODEL_ID } from "./aiConfig";

const BACKEND_URL = "http://localhost:8000/api/waste/analyze";

export const analyzeWaste = async (
  input: string,
  isImage: boolean = false,
  modelId: string = DEFAULT_MODEL_ID
): Promise<WasteAnalysisResult> => {
  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input,
        isImage,
        modelId
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend Error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    return data as WasteAnalysisResult;

  } catch (error: any) {
    console.error("Analysis Error:", error);
    // 抛出错误让调用方处理，而不是返回假数据
    throw new Error(error.message || "无法连接到后端服务，请确保后端已启动");
  }
};
