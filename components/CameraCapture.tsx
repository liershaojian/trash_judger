import React, { useRef, useEffect, useState, useCallback } from 'react';

interface CameraCaptureProps {
  onCapture: (base64Image: string) => void;
  onClose: () => void;
}

export const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }, // Prefer back camera
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // wait for loadedmetadata to ensure video dimensions are known
        videoRef.current.onloadedmetadata = () => {
             setIsStreaming(true);
             if (videoRef.current) videoRef.current.play();
        }
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      setError("无法访问摄像头。请检查权限设置或使用文件上传。");
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => {
      // Cleanup stream
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [startCamera]);

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Set canvas size to match video stream
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      // Get base64 string (remove prefix for service)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
      const base64 = dataUrl.split(',')[1];
      onCapture(base64);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col">
      {/* Header */}
      <div className="relative px-4 py-4 flex justify-between items-center bg-black/50 backdrop-blur-sm z-10">
        <button onClick={onClose} className="text-white p-2 rounded-full bg-white/10 hover:bg-white/20 transition">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        </button>
        <span className="text-white font-medium">拍摄垃圾</span>
        <div className="w-10"></div> {/* Spacer */}
      </div>

      {/* Camera Viewport */}
      <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black">
        {error ? (
          <div className="text-white text-center p-6">
            <p className="mb-4 text-red-400">{error}</p>
            <button onClick={onClose} className="px-4 py-2 bg-white text-black rounded-lg">返回</button>
          </div>
        ) : (
          <>
            <video 
              ref={videoRef} 
              className="absolute w-full h-full object-cover" 
              playsInline 
              muted 
            />
            {/* Overlay Grid */}
            <div className="absolute inset-0 pointer-events-none opacity-30 grid grid-cols-3 grid-rows-3">
                <div className="border border-white/20"></div><div className="border border-white/20"></div><div className="border border-white/20"></div>
                <div className="border border-white/20"></div><div className="border border-white/20"></div><div className="border border-white/20"></div>
                <div className="border border-white/20"></div><div className="border border-white/20"></div><div className="border border-white/20"></div>
            </div>
            {!isStreaming && (
              <div className="absolute inset-0 flex items-center justify-center bg-black text-white">
                <svg className="animate-spin h-8 w-8 text-emerald-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
            )}
          </>
        )}
      </div>

      {/* Controls */}
      <div className="h-32 bg-black flex items-center justify-center relative">
         <canvas ref={canvasRef} className="hidden" />
         {!error && (
           <button 
             onClick={handleCapture}
             disabled={!isStreaming}
             className="w-20 h-20 rounded-full border-4 border-white flex items-center justify-center bg-white/20 hover:bg-white/40 transition active:scale-95"
           >
             <div className="w-16 h-16 bg-white rounded-full"></div>
           </button>
         )}
      </div>
    </div>
  );
};