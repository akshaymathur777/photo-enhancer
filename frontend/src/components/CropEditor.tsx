import { useCallback, useEffect, useRef, useState } from "react";
import { FlipHorizontal, FlipVertical, Loader2, RotateCcw, RotateCw } from "lucide-react";

interface CropEditorProps {
  imageUrl: string;
  file: File;
  onResult: (blob: Blob) => void;
  onError: (msg: string) => void;
}

export function CropEditor({ imageUrl, file, onResult, onError }: CropEditorProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  const [angle, setAngle] = useState(0);
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [crop, setCrop] = useState({ x: 0.1, y: 0.1, w: 0.8, h: 0.8 });
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const dragging = useRef<string | null>(null);
  const dragStart = useRef({ x: 0, y: 0, crop: crop });

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const onLoad = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    if (img.complete) onLoad();
    else img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  }, [imageUrl]);

  const applyCrop = async () => {
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("left", String(Math.round(crop.x * imgSize.w)));
      form.append("top", String(Math.round(crop.y * imgSize.h)));
      form.append("width", String(Math.round(crop.w * imgSize.w)));
      form.append("height", String(Math.round(crop.h * imgSize.h)));
      form.append("angle", String(angle));
      form.append("flip_h", String(flipH));
      form.append("flip_v", String(flipV));
      const res = await fetch("/api/crop-rotate", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      onResult(await res.blob());
    } catch (err) {
      onError(err instanceof Error ? err.message : "Crop failed");
    } finally {
      setLoading(false);
    }
  };

  const onPointerDown = (handle: string) => (e: React.PointerEvent) => {
    e.stopPropagation();
    dragging.current = handle;
    dragStart.current = { x: e.clientX, y: e.clientY, crop: { ...crop } };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const dx = (e.clientX - dragStart.current.x) / rect.width;
    const dy = (e.clientY - dragStart.current.y) / rect.height;
    const c = dragStart.current.crop;
    if (dragging.current === "move") {
      setCrop({
        ...c,
        x: Math.max(0, Math.min(1 - c.w, c.x + dx)),
        y: Math.max(0, Math.min(1 - c.h, c.y + dy)),
      });
    } else if (dragging.current === "se") {
      setCrop({
        ...c,
        w: Math.max(0.1, Math.min(1 - c.x, c.w + dx)),
        h: Math.max(0.1, Math.min(1 - c.y, c.h + dy)),
      });
    }
  }, []);

  const onPointerUp = () => {
    dragging.current = null;
  };

  return (
    <div className="flex flex-col gap-3 h-full">
      <div
        ref={containerRef}
        className="relative flex-1 min-h-[200px] rounded-xl border border-surface-border overflow-hidden checkerboard"
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Crop"
          className="w-full h-full object-contain"
          draggable={false}
          style={{ transform: `${flipH ? "scaleX(-1)" : ""} ${flipV ? "scaleY(-1)" : ""}`.trim() || undefined }}
        />
        <div
          className="absolute border-2 border-accent bg-accent/10 cursor-move"
          style={{
            left: `${crop.x * 100}%`,
            top: `${crop.y * 100}%`,
            width: `${crop.w * 100}%`,
            height: `${crop.h * 100}%`,
          }}
          onPointerDown={onPointerDown("move")}
        >
          <div
            className="absolute bottom-0 right-0 w-4 h-4 bg-accent cursor-se-resize"
            onPointerDown={onPointerDown("se")}
          />
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0 flex-wrap">
        <button type="button" onClick={() => setAngle((a) => a - 90)} className="p-2 rounded-lg bg-surface border border-surface-border" title="Rotate left">
          <RotateCcw className="w-4 h-4" />
        </button>
        <span className="text-sm text-gray-400 min-w-[3rem] text-center">{angle}°</span>
        <button type="button" onClick={() => setAngle((a) => a + 90)} className="p-2 rounded-lg bg-surface border border-surface-border" title="Rotate right">
          <RotateCw className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={() => setFlipH((f) => !f)}
          className={`p-2 rounded-lg border ${flipH ? "bg-accent/20 border-accent/40" : "bg-surface border-surface-border"}`}
          title="Flip horizontal"
        >
          <FlipHorizontal className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={() => setFlipV((f) => !f)}
          className={`p-2 rounded-lg border ${flipV ? "bg-accent/20 border-accent/40" : "bg-surface border-surface-border"}`}
          title="Flip vertical"
        >
          <FlipVertical className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={applyCrop}
          disabled={loading}
          className="px-4 py-2 rounded-lg bg-accent hover:bg-accent-hover text-sm font-semibold disabled:opacity-40"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Apply crop"}
        </button>
      </div>
    </div>
  );
}
