import React, { useState, useEffect, useCallback } from 'react';
import { analyzeWaste } from './services/wasteService';
import { SUPPORTED_MODELS, DEFAULT_MODEL_ID } from './services/aiConfig';
import { WasteAnalysisResult, HistoryItem } from './types';
import { CameraCapture } from './components/CameraCapture';
import { ResultCard } from './components/ResultCard';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [result, setResult] = useState<WasteAnalysisResult | null>(null);
  const [userImage, setUserImage] = useState<string | null>(null); // Base64 without prefix
  const [searchText, setSearchText] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);

  // Load history from local storage on mount
  useEffect(() => {
    const saved = localStorage.getItem('waste_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load history", e);
      }
    }
  }, []);

  const saveToHistory = useCallback((res: WasteAnalysisResult, img?: string) => {
    const newItem: HistoryItem = {
      ...res,
      id: Date.now().toString(),
      timestamp: Date.now(),
      imageUrl: img // Store small thumbnail or ref? Storing full base64 in localstorage is heavy, but okay for demo.
    };

    setHistory(prev => {
      const updated = [newItem, ...prev].slice(0, 10); // Keep last 10
      localStorage.setItem('waste_history', JSON.stringify(updated));
      return updated;
    });
  }, []);

  const handleAnalysis = async (input: string, isImage: boolean) => {
    setIsLoading(true);
    setResult(null);
    if (isImage) setUserImage(input);
    else setUserImage(null);

    try {
      // If user selects a text-only model for image, warn them or switch automatically?
      // For now, let's rely on the user selecting Qwen-VL for images if using Alibaba.
      if (isImage && modelId === 'qwen-plus') {
        // qwen-plus is text only, switch to qwen-vl-max ideally, or let API fail.
        // Let's just proceed, the service will try to send image and Qwen might reject it.
        console.warn("Warning: Using text model for image.");
      }

      const analysis = await analyzeWaste(input, isImage, modelId);
      setResult(analysis);
      saveToHistory(analysis, isImage ? input : undefined);
    } catch (error) {
      console.error("Analysis failed", error);
      alert("识别失败，请检查网络或 API Key 配置");
    } finally {
      setIsLoading(false);
      setShowCamera(false);
      setSearchText("");
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchText.trim()) return;
    handleAnalysis(searchText, false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = (reader.result as string).split(',')[1];
        handleAnalysis(base64String, true);
      };
      reader.readAsDataURL(file);
    }
  };

  const resetApp = () => {
    setResult(null);
    setUserImage(null);
    setSearchText("");
  };

  if (showCamera) {
    return (
      <CameraCapture
        onCapture={(base64) => handleAnalysis(base64, true)}
        onClose={() => setShowCamera(false)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 pb-10">
      {/* Header */}
      <header className="bg-emerald-600 text-white pt-12 pb-24 px-6 rounded-b-[2.5rem] shadow-lg relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
          <div className="absolute top-[-10%] right-[-10%] w-64 h-64 bg-emerald-400 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob"></div>
          <div className="absolute top-[20%] left-[-10%] w-64 h-64 bg-teal-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000"></div>
        </div>

        <div className="max-w-md mx-auto relative z-10">
          <div className="flex items-center justify-between mb-4">
            <span className="bg-emerald-800/30 px-3 py-1 rounded-full text-xs font-semibold tracking-wider uppercase backdrop-blur-sm border border-emerald-500/30">EcoSort AI</span>

            {/* Model Selector (Dynamically Loaded) */}
            <div className="relative group max-w-[200px]">
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full appearance-none bg-emerald-800/30 border border-emerald-500/30 text-white text-xs font-semibold rounded-full pl-3 pr-8 py-1 focus:outline-none focus:ring-2 focus:ring-emerald-400 cursor-pointer hover:bg-emerald-800/40 transition truncate"
              >
                <optgroup label="🏠 本地模型 (离线)">
                  {SUPPORTED_MODELS.LOCAL.map(model => (
                    <option key={model.id} value={model.id} className="text-slate-900">{model.name}</option>
                  ))}
                </optgroup>

                <optgroup label="☁️ 阿里云百炼 (在线)">
                  {SUPPORTED_MODELS.QWEN.map(model => (
                    <option key={model.id} value={model.id} className="text-slate-900">{model.name}</option>
                  ))}
                </optgroup>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-emerald-100">
                <svg className="fill-current h-3 w-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" /></svg>
              </div>
            </div>
          </div>
          <h1 className="text-4xl font-bold mb-2 tracking-tight">智能垃圾分类</h1>
          <p className="text-emerald-100 text-lg">拍一拍，或搜一搜，轻松搞定分类难题。</p>
        </div>
      </header>

      <main className="max-w-md mx-auto px-6 -mt-16 relative z-20 space-y-6">

        {/* Main Interaction Area */}
        {isLoading ? (
          <div className="bg-white rounded-3xl p-8 shadow-xl text-center py-16 border border-emerald-100">
            <div className="inline-block relative">
              <div className="h-16 w-16 border-4 border-emerald-200 border-t-emerald-500 rounded-full animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl">♻️</span>
              </div>
            </div>
            <p className="mt-6 text-slate-500 font-medium animate-pulse">AI 正在分析您的物品...</p>
            <p className="text-xs text-slate-400 mt-2 font-mono">Model: {modelId}</p>
          </div>
        ) : result ? (
          <ResultCard result={result} onReset={resetApp} userImage={userImage || undefined} />
        ) : (
          <>
            {/* Search Box */}
            <div className="bg-white p-2 rounded-2xl shadow-xl border border-slate-100 flex items-center">
              <form onSubmit={handleTextSubmit} className="flex-1 flex items-center">
                <span className="pl-4 text-slate-400">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                </span>
                <input
                  type="text"
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  placeholder="输入物品名称 (例如: 电池)"
                  className="w-full p-4 outline-none text-slate-700 placeholder-slate-400 bg-transparent"
                />
                <button type="submit" disabled={!searchText} className="bg-emerald-600 text-white p-3 rounded-xl hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
              </form>
            </div>

            {/* Camera / Upload Actions */}
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setShowCamera(true)}
                className="group relative overflow-hidden bg-gradient-to-br from-emerald-500 to-teal-600 rounded-3xl p-6 shadow-lg shadow-emerald-500/20 text-left hover:shadow-xl transition active:scale-95"
              >
                <div className="relative z-10">
                  <div className="bg-white/20 w-12 h-12 rounded-2xl flex items-center justify-center mb-4 backdrop-blur-sm group-hover:scale-110 transition text-white">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
                  </div>
                  <h3 className="text-white font-bold text-lg">拍照识别</h3>
                  <p className="text-emerald-100 text-sm mt-1">AI 视觉分析</p>
                </div>
                <div className="absolute -bottom-4 -right-4 text-emerald-400/20">
                  <svg width="100" height="100" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10"></circle></svg>
                </div>
              </button>

              <label className="cursor-pointer group relative overflow-hidden bg-white rounded-3xl p-6 shadow-lg border border-slate-100 text-left hover:bg-slate-50 transition active:scale-95">
                <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                <div className="relative z-10">
                  <div className="bg-indigo-100 w-12 h-12 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition text-indigo-600">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                  </div>
                  <h3 className="text-slate-800 font-bold text-lg">相册上传</h3>
                  <p className="text-slate-500 text-sm mt-1">选择现有图片</p>
                </div>
              </label>
            </div>

            {/* History */}
            {history.length > 0 && (
              <div className="pt-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold text-slate-700">最近记录</h3>
                  <button onClick={() => { setHistory([]); localStorage.removeItem('waste_history') }} className="text-xs text-slate-400 hover:text-slate-600">清除</button>
                </div>
                <div className="space-y-3">
                  {history.map((item) => (
                    <div key={item.id} className="bg-white p-4 rounded-2xl shadow-sm border border-slate-100 flex items-center gap-4 animate-fade-in-up">
                      <div className={`w-2 h-12 rounded-full shrink-0 ${item.category === 'Recyclable' ? 'bg-blue-500' :
                        item.category === 'Hazardous' ? 'bg-red-500' :
                          item.category === 'Wet' ? 'bg-emerald-500' :
                            'bg-gray-500'
                        }`}></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex justify-between items-center">
                          <h4 className="font-bold text-slate-800 truncate">{item.itemName}</h4>
                          <span className="text-xs text-slate-400">{new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        </div>
                        <p className="text-sm text-slate-500 truncate">{item.explanation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
