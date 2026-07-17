import {
  Brush,
  Loader2,
  MousePointer2,
  RotateCcw,
  Trash2,
  Users,
  Wand2,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { AiMode } from "../api";
import {
  autoRemovePeople,
  autoSegmentPeople,
  ClickPoint,
  InpaintMode,
  removeObjects,
  segmentPreview,
  SelectionMode,
} from "../objectRemovalApi";

interface ImageLayout {
  offsetX: number;
  offsetY: number;
  drawW: number;
  drawH: number;
  naturalW: number;
  naturalH: number;
}

interface ObjectRemovalEditorProps {
  imageUrl: string;
  file: File;
  mode?: AiMode;
  onResult: (blob: Blob) => void;
  onError: (message: string | null) => void;
  onWorkingFileChange: (file: File) => void;
}

function computeLayout(img: HTMLImageElement, container: HTMLDivElement): ImageLayout | null {
  if (!img.naturalWidth) return null;
  const rect = container.getBoundingClientRect();
  const naturalW = img.naturalWidth;
  const naturalH = img.naturalHeight;
  const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
  const drawW = naturalW * scale;
  const drawH = naturalH * scale;
  return {
    offsetX: (rect.width - drawW) / 2,
    offsetY: (rect.height - drawH) / 2,
    drawW,
    drawH,
    naturalW,
    naturalH,
  };
}

function defaultInpaintMode(globalMode: AiMode): InpaintMode {
  return globalMode === "fast" ? "quick" : "ai";
}

export function ObjectRemovalEditor({
  imageUrl,
  file,
  mode = "balanced",
  onResult,
  onError,
  onWorkingFileChange,
}: ObjectRemovalEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const brushOverlayRef = useRef<HTMLCanvasElement>(null);
  const paintingRef = useRef(false);

  const [points, setPoints] = useState<ClickPoint[]>([]);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<"preview" | "remove" | null>(null);
  const [subtractMode, setSubtractMode] = useState(false);
  const [edgeSoftness, setEdgeSoftness] = useState(1.0);
  const [layout, setLayout] = useState<ImageLayout | null>(null);
  const [selectionMode, setSelectionMode] = useState<SelectionMode>("smart");
  const [inpaintMode, setInpaintMode] = useState<InpaintMode>(() => defaultInpaintMode(mode));
  const [inpaintTouched, setInpaintTouched] = useState(false);
  const [brushSize, setBrushSize] = useState(24);
  const [hasBrushMask, setHasBrushMask] = useState(false);

  useEffect(() => {
    if (!inpaintTouched) {
      setInpaintMode(defaultInpaintMode(mode));
    }
  }, [mode, inpaintTouched]);

  const refreshLayout = useCallback(() => {
    const img = imgRef.current;
    const container = containerRef.current;
    if (img && container) {
      setLayout(computeLayout(img, container));
    }
  }, []);

  const initMaskCanvas = useCallback((w: number, h: number) => {
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    maskCanvasRef.current = canvas;
    setHasBrushMask(false);
  }, []);

  const getImageCoords = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } | null => {
      const container = containerRef.current;
      if (!layout || !container) return null;

      const rect = container.getBoundingClientRect();
      const x = ((clientX - rect.left - layout.offsetX) / layout.drawW) * layout.naturalW;
      const y = ((clientY - rect.top - layout.offsetY) / layout.drawH) * layout.naturalH;

      if (x < 0 || y < 0 || x > layout.naturalW || y > layout.naturalH) return null;
      return { x, y };
    },
    [layout]
  );

  const paintBrush = useCallback(
    (clientX: number, clientY: number) => {
      const coords = getImageCoords(clientX, clientY);
      const maskCanvas = maskCanvasRef.current;
      const overlay = brushOverlayRef.current;
      if (!coords || !maskCanvas || !layout) return;

      const ctx = maskCanvas.getContext("2d");
      const octx = overlay?.getContext("2d");
      if (!ctx) return;

      const radius = brushSize / 2;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(coords.x, coords.y, radius, 0, Math.PI * 2);
      ctx.fill();
      setHasBrushMask(true);

      if (octx && overlay) {
        const sx = (coords.x / layout.naturalW) * layout.drawW;
        const sy = (coords.y / layout.naturalH) * layout.drawH;
        const sr = (brushSize / layout.naturalW) * layout.drawW;
        octx.fillStyle = "rgba(239, 68, 68, 0.45)";
        octx.beginPath();
        octx.arc(sx, sy, sr / 2, 0, Math.PI * 2);
        octx.fill();
      }
    },
    [brushSize, getImageCoords, layout]
  );

  const clearBrushMask = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const overlay = brushOverlayRef.current;
    if (maskCanvas) {
      const ctx = maskCanvas.getContext("2d");
      ctx?.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
    if (overlay) {
      const octx = overlay.getContext("2d");
      octx?.clearRect(0, 0, overlay.width, overlay.height);
    }
    setHasBrushMask(false);
  }, []);

  const clearSelection = useCallback(() => {
    setPoints([]);
    setOverlayUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    clearBrushMask();
  }, [clearBrushMask]);

  const updatePreview = useCallback(
    async (pts: ClickPoint[], currentFile: File) => {
      if (pts.length === 0) {
        setOverlayUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return null;
        });
        return;
      }
      setLoading("preview");
      try {
        const blob = await segmentPreview(currentFile, pts);
        const url = URL.createObjectURL(blob);
        setOverlayUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      } catch (err) {
        onError(err instanceof Error ? err.message : "Preview failed");
      } finally {
        setLoading(null);
      }
    },
    [onError]
  );

  const handleSmartClick = (e: React.MouseEvent) => {
    if (selectionMode !== "smart" || loading) return;
    const coords = getImageCoords(e.clientX, e.clientY);
    if (!coords) return;
    const pt: ClickPoint = {
      x: Math.round(coords.x),
      y: Math.round(coords.y),
      label: subtractMode ? 0 : 1,
    };
    const next = [...points, pt];
    setPoints(next);
    updatePreview(next, file);
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (selectionMode !== "brush" || loading) return;
    paintingRef.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    paintBrush(e.clientX, e.clientY);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!paintingRef.current || selectionMode !== "brush") return;
    paintBrush(e.clientX, e.clientY);
  };

  const handlePointerUp = () => {
    paintingRef.current = false;
  };

  const undoLast = () => {
    if (selectionMode === "smart") {
      const next = points.slice(0, -1);
      setPoints(next);
      updatePreview(next, file);
    } else {
      clearBrushMask();
    }
  };

  const detectPeople = async () => {
    setLoading("preview");
    onError(null);
    try {
      const blob = await autoSegmentPeople(file);
      const url = URL.createObjectURL(blob);
      setOverlayUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      setSelectionMode("auto-people");
    } catch (err) {
      onError(err instanceof Error ? err.message : "People detection failed");
    } finally {
      setLoading(null);
    }
  };

  const maskToBlob = (): Promise<Blob> => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return Promise.reject(new Error("No brush mask"));
    return new Promise((resolve, reject) => {
      maskCanvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Could not export mask"));
      }, "image/png");
    });
  };

  const applyResult = (blob: Blob) => {
    onResult(blob);
    const newFile = new File([blob], file.name, { type: "image/jpeg" });
    onWorkingFileChange(newFile);
    clearSelection();
    setSelectionMode("smart");
  };

  const handleRemove = async () => {
    const canRemoveSmart = selectionMode === "smart" && points.length > 0;
    const canRemoveBrush = selectionMode === "brush" && hasBrushMask;
    const canRemovePeople = selectionMode === "auto-people";

    if (!canRemoveSmart && !canRemoveBrush && !canRemovePeople) return;

    setLoading("remove");
    onError(null);
    try {
      let blob: Blob;
      if (canRemovePeople) {
        blob = await autoRemovePeople(file, inpaintMode, edgeSoftness, mode);
      } else if (canRemoveBrush) {
        const maskFile = await maskToBlob();
        blob = await removeObjects(file, { maskFile, inpaintMode, edgeSoftness, mode });
      } else {
        blob = await removeObjects(file, { points, inpaintMode, edgeSoftness, mode });
      }
      applyResult(blob);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Removal failed");
    } finally {
      setLoading(null);
    }
  };

  const switchSelectionMode = (next: SelectionMode) => {
    if (next === selectionMode) return;
    clearSelection();
    setSelectionMode(next);
    if (next === "auto-people") {
      void detectPeople();
    }
  };

  useEffect(() => {
    return () => {
      if (overlayUrl) URL.revokeObjectURL(overlayUrl);
    };
  }, [overlayUrl]);

  useEffect(() => {
    clearSelection();
    setSelectionMode("smart");
  }, [imageUrl, clearSelection]);

  useEffect(() => {
    const img = imgRef.current;
    if (img?.naturalWidth && img.naturalHeight) {
      initMaskCanvas(img.naturalWidth, img.naturalHeight);
    }
  }, [imageUrl, initMaskCanvas]);

  useEffect(() => {
    refreshLayout();
    window.addEventListener("resize", refreshLayout);
    return () => window.removeEventListener("resize", refreshLayout);
  }, [refreshLayout, overlayUrl, imageUrl, selectionMode]);

  useEffect(() => {
    const overlay = brushOverlayRef.current;
    if (!overlay || !layout) return;
    overlay.width = Math.round(layout.drawW);
    overlay.height = Math.round(layout.drawH);
    const octx = overlay.getContext("2d");
    octx?.clearRect(0, 0, overlay.width, overlay.height);
  }, [layout, selectionMode, imageUrl]);

  const displaySrc = selectionMode === "brush" ? imageUrl : overlayUrl ?? imageUrl;
  const canRemove =
    (selectionMode === "smart" && points.length > 0) ||
    (selectionMode === "brush" && hasBrushMask) ||
    selectionMode === "auto-people";

  const canClear =
    (selectionMode === "smart" && points.length > 0) ||
    (selectionMode === "brush" && hasBrushMask) ||
    (selectionMode === "auto-people" && overlayUrl !== null);

  const toScreen = (pt: ClickPoint) => {
    if (!layout) return { left: 0, top: 0 };
    return {
      left: layout.offsetX + (pt.x / layout.naturalW) * layout.drawW,
      top: layout.offsetY + (pt.y / layout.naturalH) * layout.drawH,
    };
  };

  const cursorClass =
    selectionMode === "brush" ? "cursor-crosshair" : selectionMode === "smart" ? "cursor-crosshair" : "cursor-default";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => switchSelectionMode("smart")}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
            selectionMode === "smart"
              ? "bg-accent/20 text-accent-hover border border-accent/40"
              : "border border-surface-border text-gray-400 hover:text-white"
          }`}
        >
          <MousePointer2 className="w-4 h-4" />
          AI Select
        </button>
        <button
          type="button"
          onClick={() => switchSelectionMode("brush")}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
            selectionMode === "brush"
              ? "bg-accent/20 text-accent-hover border border-accent/40"
              : "border border-surface-border text-gray-400 hover:text-white"
          }`}
        >
          <Brush className="w-4 h-4" />
          Brush
        </button>
        <button
          type="button"
          onClick={() => switchSelectionMode("auto-people")}
          disabled={loading !== null}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
            selectionMode === "auto-people"
              ? "bg-accent/20 text-accent-hover border border-accent/40"
              : "border border-surface-border text-gray-400 hover:text-white"
          }`}
        >
          <Users className="w-4 h-4" />
          Remove All People
        </button>
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs text-gray-500 uppercase tracking-wide">Inpaint</span>
        <button
          type="button"
          onClick={() => {
            setInpaintTouched(true);
            setInpaintMode("ai");
          }}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
            inpaintMode === "ai"
              ? "bg-purple-600/20 text-purple-300 border border-purple-600/40"
              : "border border-surface-border text-gray-400 hover:text-white"
          }`}
        >
          <Wand2 className="w-3.5 h-3.5" />
          AI Remove
        </button>
        <button
          type="button"
          onClick={() => {
            setInpaintTouched(true);
            setInpaintMode("quick");
          }}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
            inpaintMode === "quick"
              ? "bg-amber-600/20 text-amber-300 border border-amber-600/40"
              : "border border-surface-border text-gray-400 hover:text-white"
          }`}
        >
          <Zap className="w-3.5 h-3.5" />
          Quick
        </button>
      </div>

      {inpaintMode === "quick" && (
        <p className="text-xs text-amber-400/90 bg-amber-950/30 border border-amber-700/30 rounded-lg px-3 py-2">
          Quick mode uses fast LaMa (512px) — lower quality. Switch to AI Remove for generative Best Quality fill.
        </p>
      )}
      {inpaintMode === "ai" && (
        <p className="text-xs text-purple-300/90 bg-purple-950/30 border border-purple-700/30 rounded-lg px-3 py-2">
          AI Remove uses SDXL Inpaint (~7 GB) on first use with CUDA. Falls back to SD1.5 / LaMa if the GPU path is unavailable.
        </p>
      )}

      <div
        ref={containerRef}
        className={`relative rounded-xl overflow-hidden border border-surface-border bg-black/20 select-none min-h-[300px] ${cursorClass}`}
        onClick={handleSmartClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        <img
          ref={imgRef}
          src={displaySrc}
          alt="Edit"
          onLoad={refreshLayout}
          className="w-full max-h-[60vh] object-contain mx-auto pointer-events-none"
          draggable={false}
        />
        {selectionMode === "brush" && layout && (
          <canvas
            ref={brushOverlayRef}
            className="absolute pointer-events-none z-10"
            style={{
              left: layout.offsetX,
              top: layout.offsetY,
              width: layout.drawW,
              height: layout.drawH,
            }}
          />
        )}
        {selectionMode === "smart" &&
          points.map((pt, i) => {
            const pos = toScreen(pt);
            return (
              <span
                key={i}
                className="absolute w-3 h-3 rounded-full border-2 border-white shadow -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10"
                style={{
                  left: pos.left,
                  top: pos.top,
                  backgroundColor: pt.label === 1 ? "#22c55e" : "#ef4444",
                }}
              />
            );
          })}
        {loading && (
          <div className="absolute inset-0 bg-black/40 flex items-center justify-center z-20">
            <Loader2 className="w-8 h-8 animate-spin text-white" />
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {selectionMode === "smart" && (
          <>
            <button
              type="button"
              onClick={() => setSubtractMode(false)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                !subtractMode
                  ? "bg-green-600/20 text-green-400 border border-green-600/40"
                  : "border border-surface-border text-gray-400 hover:text-white"
              }`}
            >
              Select (+)
            </button>
            <button
              type="button"
              onClick={() => setSubtractMode(true)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                subtractMode
                  ? "bg-red-600/20 text-red-400 border border-red-600/40"
                  : "border border-surface-border text-gray-400 hover:text-white"
              }`}
            >
              Exclude (−)
            </button>
          </>
        )}
        {selectionMode === "brush" && (
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <span className="whitespace-nowrap">Brush size</span>
            <input
              type="range"
              min={8}
              max={80}
              step={2}
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
              disabled={loading !== null}
              className="w-28 accent-accent"
            />
            <span className="w-6 text-gray-500 tabular-nums">{brushSize}</span>
          </label>
        )}
        <button
          type="button"
          onClick={undoLast}
          disabled={
            loading !== null ||
            (selectionMode === "smart" && points.length === 0) ||
            (selectionMode === "brush" && !hasBrushMask)
          }
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border border-surface-border text-gray-400 hover:text-white disabled:opacity-40"
        >
          <RotateCcw className="w-4 h-4" />
          Undo
        </button>
        <button
          type="button"
          onClick={clearSelection}
          disabled={loading !== null || !canClear}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border border-surface-border text-gray-400 hover:text-white disabled:opacity-40"
        >
          Clear
        </button>
        <label className="flex items-center gap-2 text-sm text-gray-400 ml-auto mr-2">
          <span className="whitespace-nowrap">Edge softness</span>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.1}
            value={edgeSoftness}
            onChange={(e) => setEdgeSoftness(parseFloat(e.target.value))}
            disabled={loading !== null}
            className="w-24 accent-accent"
          />
          <span className="w-8 text-gray-500 tabular-nums">{edgeSoftness.toFixed(1)}</span>
        </label>
        <button
          type="button"
          onClick={handleRemove}
          disabled={!canRemove || loading !== null}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-accent hover:bg-accent-hover disabled:opacity-40"
        >
          {loading === "remove" ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Wand2 className="w-4 h-4" />
          )}
          {selectionMode === "auto-people" ? "Remove people" : "Remove selected"}
        </button>
      </div>

      <p className="text-sm text-gray-500">
        <Trash2 className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
        {selectionMode === "smart" &&
          "Click objects to select (red highlight). Use Exclude to refine. AI Remove uses SDXL; Quick is faster."}
        {selectionMode === "brush" && "Paint over areas to remove. Use AI Remove for best quality or Quick for speed."}
        {selectionMode === "auto-people" &&
          "All detected people are highlighted. Click Remove people to erase them in one step."}
      </p>
    </div>
  );
}
