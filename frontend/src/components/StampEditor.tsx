import { useCallback, useEffect, useRef, useState } from "react";

interface StampEditorProps {
  imageUrl: string;
  onResult: (blob: Blob) => void;
}

export function StampEditor({ imageUrl, onResult }: StampEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [brushSize, setBrushSize] = useState(40);
  const [source, setSource] = useState<{ x: number; y: number } | null>(null);
  const painting = useRef(false);
  const altHeld = useRef(false);

  const drawImage = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete) return;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0);
  }, []);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    if (img.complete) drawImage();
    else img.onload = drawImage;
  }, [imageUrl, drawImage]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      altHeld.current = e.altKey;
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, []);

  const toCanvasCoords = (e: React.PointerEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const stamp = (x: number, y: number) => {
    if (!source) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const half = brushSize / 2;
    ctx.drawImage(canvas, source.x - half, source.y - half, brushSize, brushSize, x - half, y - half, brushSize, brushSize);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const pt = toCanvasCoords(e);
    if (e.altKey || altHeld.current) {
      setSource(pt);
      return;
    }
    if (!source) return;
    painting.current = true;
    stamp(pt.x, pt.y);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!painting.current || !source) return;
    const pt = toCanvasCoords(e);
    stamp(pt.x, pt.y);
  };

  const onPointerUp = () => {
    painting.current = false;
  };

  const apply = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) onResult(blob);
    }, "image/jpeg", 0.95);
  };

  return (
    <div className="flex flex-col gap-3 h-full">
      <p className="text-xs text-gray-500 shrink-0">
        Alt+click to set source · Paint to clone · Brush size: {brushSize}px
        {source && <span className="text-accent ml-2">Source set</span>}
      </p>
      <div className="flex-1 min-h-0 overflow-auto rounded-xl border border-surface-border checkerboard">
        <canvas
          ref={canvasRef}
          className="max-w-full cursor-crosshair touch-none"
          style={{ display: "block", margin: "0 auto" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />
        <img ref={imgRef} src={imageUrl} alt="" className="hidden" crossOrigin="anonymous" />
      </div>
      <div className="flex items-center gap-3 shrink-0">
        <input
          type="range"
          min={10}
          max={120}
          value={brushSize}
          onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
          className="flex-1 accent-accent"
        />
        <button type="button" onClick={apply} disabled={!source} className="px-4 py-2 rounded-lg bg-accent text-sm font-semibold disabled:opacity-40">
          Apply stamp
        </button>
      </div>
    </div>
  );
}
