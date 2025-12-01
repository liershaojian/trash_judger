import React from 'react';
import { WasteCategory, WasteAnalysisResult } from '../types';

interface ResultCardProps {
  result: WasteAnalysisResult;
  onReset: () => void;
  userImage?: string;
}

// Color and Icon mapping based on category
// 注意：Tailwind 需要完整类名，不支持动态拼接如 bg-${color}-500
const getCategoryStyle = (category: WasteCategory) => {
  switch (category) {
    case WasteCategory.Recyclable:
      return {
        bg: "bg-blue-50",
        border: "border-blue-200",
        text: "text-blue-800",
        icon: "♻️",
        label: "可回收物 (Recyclable)",
        dotColor: "bg-blue-500",
        btnColor: "bg-blue-600 hover:bg-blue-700"
      };
    case WasteCategory.Hazardous:
      return {
        bg: "bg-red-50",
        border: "border-red-200",
        text: "text-red-800",
        icon: "☣️",
        label: "有害垃圾 (Hazardous)",
        dotColor: "bg-red-500",
        btnColor: "bg-red-600 hover:bg-red-700"
      };
    case WasteCategory.Wet:
      return {
        bg: "bg-emerald-50",
        border: "border-emerald-200",
        text: "text-emerald-800",
        icon: "🍂",
        label: "厨余/湿垃圾 (Wet)",
        dotColor: "bg-emerald-500",
        btnColor: "bg-emerald-600 hover:bg-emerald-700"
      };
    case WasteCategory.Dry:
      return {
        bg: "bg-gray-100",
        border: "border-gray-300",
        text: "text-gray-800",
        icon: "🗑️",
        label: "其他/干垃圾 (Dry)",
        dotColor: "bg-gray-500",
        btnColor: "bg-gray-600 hover:bg-gray-700"
      };
    default:
      return {
        bg: "bg-slate-50",
        border: "border-slate-200",
        text: "text-slate-800",
        icon: "❓",
        label: "未知类别 (Unknown)",
        dotColor: "bg-slate-500",
        btnColor: "bg-slate-600 hover:bg-slate-700"
      };
  }
};

export const ResultCard: React.FC<ResultCardProps> = ({ result, onReset, userImage }) => {
  const style = getCategoryStyle(result.category);

  return (
    <div className="w-full max-w-md mx-auto animate-fade-in-up">
      <div className={`relative overflow-hidden rounded-3xl shadow-2xl border-2 ${style.border} bg-white`}>
        
        {/* Header / Image Section */}
        <div className={`h-48 ${style.bg} flex flex-col items-center justify-center relative overflow-hidden`}>
          {userImage ? (
            <img src={`data:image/jpeg;base64,${userImage}`} alt="User upload" className="w-full h-full object-cover opacity-90" />
          ) : (
            <div className="text-8xl transform hover:scale-110 transition-transform duration-300">{style.icon}</div>
          )}
          
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/60 to-transparent text-white backdrop-blur-[2px]">
            <h2 className="text-3xl font-bold drop-shadow-sm">{result.itemName}</h2>
          </div>
        </div>

        {/* Content Section */}
        <div className="p-6 space-y-6">
          
          {/* Badge */}
          <div className="flex items-center justify-between">
            <span className={`px-4 py-2 rounded-full text-sm font-bold uppercase tracking-wide ${style.bg} ${style.text} border ${style.border}`}>
              {style.label}
            </span>
            {result.confidence > 0 && (
              <span className="text-xs text-slate-400">置信度: {(result.confidence * 100).toFixed(0)}%</span>
            )}
          </div>

          {/* Explanation */}
          <div>
            <h3 className="text-sm font-semibold text-slate-500 uppercase mb-2">分类原因</h3>
            <p className="text-slate-700 leading-relaxed text-lg">
              {result.explanation}
            </p>
          </div>

          {/* Tips */}
          <div>
            <h3 className="text-sm font-semibold text-slate-500 uppercase mb-2">投放建议</h3>
            <ul className="space-y-2">
              {result.disposalTips.map((tip, idx) => (
                <li key={idx} className="flex items-start">
                  <span className={`mr-2 mt-1.5 w-1.5 h-1.5 rounded-full ${style.dotColor} flex-shrink-0`}></span>
                  <span className="text-slate-600">{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Actions */}
          <div className="pt-4">
            <button
              onClick={onReset}
              className={`w-full py-4 rounded-xl font-bold text-white shadow-lg hover:shadow-xl transition active:scale-[0.98] ${style.btnColor}`}
            >
              识别下一个
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};