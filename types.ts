export enum WasteCategory {
  Recyclable = "Recyclable", // 可回收物
  Hazardous = "Hazardous",   // 有害垃圾
  Wet = "Wet",               // 厨余/湿垃圾
  Dry = "Dry",               // 其他/干垃圾
  Unknown = "Unknown"        // 未知/非垃圾
}

export interface WasteAnalysisResult {
  itemName: string;
  category: WasteCategory;
  confidence: number;
  explanation: string;
  disposalTips: string[];
}

export interface HistoryItem extends WasteAnalysisResult {
  id: string;
  timestamp: number;
  imageUrl?: string; // Base64 string if available
}