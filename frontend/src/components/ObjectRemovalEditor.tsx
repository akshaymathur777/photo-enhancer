import {
  Brush,
  Loader2,
  Lock,
  MousePointer2,
  Redo2,
  RotateCcw,
  Trash2,
  Undo2,
  UserMinus,
  Users,
  Wand2,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { AiMode, GenerativeRemovalStatus } from "../api";
import { fetchGenerativeRemovalStatus } from "../api";
import {
  autoRemoveOthers,
  autoRemovePeople,
  autoSegmentPeople,
  ClickPoint,
  InpaintMode,
  previewRemoveOthers,
  removeObjects,
  segmentPreview,
  SelectionMode,
  warmRemovalModels,
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

function engineLabel(engine: string): string {
  switch (engine) {
    case "powerpaint":
      return "PowerPaint erase";
    case "sdxl":
      return "SDXL inpaint";
    case "sd15":
      return "SD1.5 inpaint";
    case "lama":
      return "LaMa";
    case "classical":
      return "Quick LaMa";
    case "water_exemplar":
      return "Texture fill";
    default:
      return engine || "inpaint";
  }
}

const GENERATIVE_MODEL_LABELS: Record<string, string> = {
  powerpaint: "PowerPaint",
  sdxl_inpaint: "SDXL Inpaint",
};

function missingWeightsLabel(generative: GenerativeRemovalStatus): string {
  return generative.models
    .filter((m) => !m.present)
    .map(
      (m) =>
        `${GENERATIVE_MODEL_LABELS[m.id] ?? m.name} (~${(m.size_estimate_mb / 1000).toFixed(1)} GB)`
    )
    .join(" and ");
}

interface GenerativeHint {
  tone: "info" | "ready" | "warn";
  badge: string;
  text: string;
}

function generativeHint(
  mode: AiMode,
  generative: GenerativeRemovalStatus | null
): GenerativeHint {
  if (mode !== "best_quality") {
    return {
      tone: "info",
      badge: "",
      text: "Balanced and Fast fill water and indoor scenes with the classical LaMa / PatchMatch path. Switch to Best Quality for the CUDA PowerPaint / SDXL erase.",
    };
  }
  if (!generative) {
    return {
      tone: "ready",
      badge: "",
      text: "Best Quality runs a Quick preview, then the CUDA erase when the PowerPaint / SDXL weights are downloaded.",
    };
  }
  if (generative.ready) {
    return {
      tone: "ready",
      badge: "PowerPaint + SDXL ready",
      text: "Best Quality runs a Quick preview then the CUDA erase (PowerPaint for indoor, SDXL for water). The generative fill keeps the hole — classical post-fill only runs on Fast/Balanced or if erase falls back.",
    };
  }
  if (!generative.cuda) {
    return {
      tone: "warn",
      badge: "No CUDA — classical fill",
      text: "No CUDA GPU detected, so Best Quality uses the classical LaMa / PatchMatch fill. That is often a good result — just not the generative erase.",
    };
  }
  const missing = missingWeightsLabel(generative);
  if (generative.scenes.water || generative.scenes.structured) {
    const readyScenes = generative.scenes.water ? "water and sky" : "indoor and structured";
    const fallbackScenes = generative.scenes.water ? "indoor and structured" : "water and sky";
    return {
      tone: "warn",
      badge: "Partly ready",
      text: `Best Quality reaches the generative erase on ${readyScenes} scenes only. ${fallbackScenes} scenes use the classical fill until you download ${missing} in Models.`,
    };
  }
  return {
    tone: "warn",
    badge: "Weights missing — classical fill",
    text: `Best Quality will use the classical LaMa / PatchMatch fill: ${missing} are not downloaded. Download them in Models to enable the generative erase — nothing downloads mid-edit.`,
  };
}

const HINT_STYLES: Record<GenerativeHint["tone"], string> = {
  info: "text-gray-400 bg-surface/60 border border-surface-border",
  ready: "text-purple-300/90 bg-purple-950/30 border border-purple-700/30",
  warn: "text-amber-300/90 bg-amber-950/30 border border-amber-700/30",
};

const BADGE_STYLES: Record<GenerativeHint["tone"], string> = {
  info: "text-gray-400 border-surface-border bg-surface/60",
  ready: "text-purple-300 border-purple-700/40 bg-purple-950/30",
  warn: "text-amber-300 border-amber-700/40 bg-amber-950/30",
};

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
  const strokeUndoRef = useRef<ImageData[]>([]);
  const strokeRedoRef = useRef<ImageData[]>([]);
  const strokeBaselineRef = useRef<ImageData | null>(null);

  const [points, setPoints] = useState<ClickPoint[]>([]);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<"preview" | "remove" | null>(null);
  const [progressLabel, setProgressLabel] = useState<string | null>(null);
  const [subtractMode, setSubtractMode] = useState(false);
  const [edgeSoftness, setEdgeSoftness] = useState(1.0);
  const [layout, setLayout] = useState<ImageLayout | null>(null);
  const [selectionMode, setSelectionMode] = useState<SelectionMode>("smart");
  const [inpaintMode, setInpaintMode] = useState<InpaintMode>(() => defaultInpaintMode(mode));
  const [inpaintTouched, setInpaintTouched] = useState(false);
  const [brushSize, setBrushSize] = useState(16);
  const [hasBrushMask, setHasBrushMask] = useState(false);
  const [lockSubject, setLockSubject] = useState(true);
  const [lockPoint, setLockPoint] = useState<{ x: number; y: number } | null>(null);
  const [strokeTick, setStrokeTick] = useState(0);
  const [lastEngine, setLastEngine] = useState<string | null>(null);
  const [generative, setGenerative] = useState<GenerativeRemovalStatus | null>(null);

  useEffect(() => {
    if (!inpaintTouched) {
      setInpaintMode(defaultInpaintMode(mode));
    }
  }, [mode, inpaintTouched]);

  // Warm models once when the tool opens (Best path feels continuous).
  useEffect(() => {
    void warmRemovalModels().catch(() => {
      /* optional — ignore warm failures */
    });
    void fetchGenerativeRemovalStatus()
      .then(setGenerative)
      .catch(() => {
        /* readiness is a hint only — stay quiet if the probe fails */
      });
  }, []);

  const refreshLayout = useCallback(() => {
    const img = imgRef.current;
    const container = containerRef.current;
    if (img && container) {
      setLayout(computeLayout(img, container));
    }
  }, []);

  const syncBrushOverlay = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const overlay = brushOverlayRef.current;
    if (!maskCanvas || !overlay || !layout) return;
    const octx = overlay.getContext("2d");
    if (!octx) return;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    octx.save();
    octx.globalAlpha = 0.45;
    octx.drawImage(maskCanvas, 0, 0, overlay.width, overlay.height);
    octx.globalCompositeOperation = "source-in";
    octx.fillStyle = "rgb(239, 68, 68)";
    octx.fillRect(0, 0, overlay.width, overlay.height);
    octx.restore();
  }, [layout]);

  const initMaskCanvas = useCallback((w: number, h: number) => {
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    maskCanvasRef.current = canvas;
    strokeUndoRef.current = [];
    strokeRedoRef.current = [];
    strokeBaselineRef.current = null;
    setHasBrushMask(false);
    setStrokeTick((t) => t + 1);
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
      if (!coords || !maskCanvas) return;

      const ctx = maskCanvas.getContext("2d");
      if (!ctx) return;

      const radius = brushSize / 2;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(coords.x, coords.y, radius, 0, Math.PI * 2);
      ctx.fill();
      setHasBrushMask(true);
      syncBrushOverlay();
    },
    [brushSize, getImageCoords, syncBrushOverlay]
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
    strokeUndoRef.current = [];
    strokeRedoRef.current = [];
    strokeBaselineRef.current = null;
    setHasBrushMask(false);
    setStrokeTick((t) => t + 1);
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
      setProgressLabel("Selecting…");
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
        setProgressLabel(null);
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
    if (lockSubject && !lockPoint && pt.label === 1) {
      setLockPoint({ x: pt.x, y: pt.y });
    }
    const next = [...points, pt];
    setPoints(next);
    updatePreview(next, file);
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (selectionMode !== "brush" || loading) return;
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas?.getContext("2d");
    if (maskCanvas && ctx) {
      strokeBaselineRef.current = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    }
    paintingRef.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    paintBrush(e.clientX, e.clientY);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!paintingRef.current || selectionMode !== "brush") return;
    paintBrush(e.clientX, e.clientY);
  };

  const handlePointerUp = () => {
    if (!paintingRef.current) return;
    paintingRef.current = false;
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas?.getContext("2d");
    const baseline = strokeBaselineRef.current;
    if (maskCanvas && ctx && baseline) {
      strokeUndoRef.current.push(baseline);
      if (strokeUndoRef.current.length > 40) strokeUndoRef.current.shift();
      strokeRedoRef.current = [];
      strokeBaselineRef.current = null;
      setStrokeTick((t) => t + 1);
    }
  };

  const undoStroke = () => {
    if (selectionMode === "smart") {
      const next = points.slice(0, -1);
      setPoints(next);
      updatePreview(next, file);
      return;
    }
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas?.getContext("2d");
    if (!maskCanvas || !ctx || strokeUndoRef.current.length === 0) return;
    const current = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const prev = strokeUndoRef.current.pop()!;
    strokeRedoRef.current.push(current);
    ctx.putImageData(prev, 0, 0);
    const data = prev.data;
    let any = false;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i]! > 0) {
        any = true;
        break;
      }
    }
    setHasBrushMask(any);
    syncBrushOverlay();
    setStrokeTick((t) => t + 1);
  };

  const redoStroke = () => {
    if (selectionMode !== "brush") return;
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas?.getContext("2d");
    if (!maskCanvas || !ctx || strokeRedoRef.current.length === 0) return;
    const current = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const next = strokeRedoRef.current.pop()!;
    strokeUndoRef.current.push(current);
    ctx.putImageData(next, 0, 0);
    setHasBrushMask(true);
    syncBrushOverlay();
    setStrokeTick((t) => t + 1);
  };

  const detectPeople = async () => {
    setLoading("preview");
    setProgressLabel("Detecting people…");
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
      setProgressLabel(null);
    }
  };

  const detectOthers = async () => {
    setLoading("preview");
    setProgressLabel("Finding bystanders…");
    onError(null);
    try {
      const blob = await previewRemoveOthers(file, lockSubject ? lockPoint : null);
      const url = URL.createObjectURL(blob);
      setOverlayUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      setSelectionMode("remove-others");
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not find other people");
    } finally {
      setLoading(null);
      setProgressLabel(null);
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

  const applyResult = (blob: Blob, stayInTool = true) => {
    onResult(blob);
    const newFile = new File([blob], file.name, { type: "image/jpeg" });
    onWorkingFileChange(newFile);
    clearSelection();
    if (stayInTool) {
      // Apple-like: stay ready for next person after each remove.
      setSelectionMode("smart");
    }
  };

  const effectiveLock = lockSubject ? lockPoint : null;

  const runRemoval = async (
    run: (inpaint: InpaintMode) => Promise<{ blob: Blob; engine: string }>
  ) => {
    setLoading("remove");
    onError(null);
    try {
      const progressive = inpaintMode === "ai" && mode === "best_quality";
      if (progressive) {
        setProgressLabel("Quick preview (LaMa)…");
        const quick = await run("quick");
        onResult(quick.blob);
        setLastEngine(quick.engine);
        setProgressLabel(
          generative && !generative.ready
            ? "Best refine…"
            : `Best refine (${engineLabel("powerpaint")}/SDXL)…`
        );
        const best = await run("ai");
        setLastEngine(best.engine);
        setProgressLabel(`Finishing (${engineLabel(best.engine)})…`);
        applyResult(best.blob);
      } else {
        setProgressLabel(inpaintMode === "quick" ? "Quick LaMa…" : "AI Remove…");
        const result = await run(inpaintMode);
        setLastEngine(result.engine);
        setProgressLabel(`Done — ${engineLabel(result.engine)}`);
        applyResult(result.blob);
      }
    } catch (err) {
      onError(err instanceof Error ? err.message : "Removal failed");
    } finally {
      setLoading(null);
      setTimeout(() => setProgressLabel(null), 1200);
    }
  };

  const handleRemove = async () => {
    const canRemoveSmart = selectionMode === "smart" && points.length > 0;
    const canRemoveBrush = selectionMode === "brush" && hasBrushMask;
    const canRemovePeople = selectionMode === "auto-people";
    const canRemoveOthers = selectionMode === "remove-others";

    if (!canRemoveSmart && !canRemoveBrush && !canRemovePeople && !canRemoveOthers) return;

    if (canRemoveOthers) {
      await runRemoval((inpaint) =>
        autoRemoveOthers(file, {
          inpaintMode: inpaint,
          edgeSoftness,
          mode,
          lockPoint: effectiveLock,
        })
      );
      return;
    }

    if (canRemovePeople) {
      await runRemoval((inpaint) => autoRemovePeople(file, inpaint, edgeSoftness, mode));
      return;
    }

    if (canRemoveBrush) {
      const maskFile = await maskToBlob();
      await runRemoval((inpaint) =>
        removeObjects(file, {
          maskFile,
          inpaintMode: inpaint,
          edgeSoftness,
          mode,
          lockPoint: effectiveLock,
        })
      );
      return;
    }

    await runRemoval((inpaint) =>
      removeObjects(file, {
        points,
        inpaintMode: inpaint,
        edgeSoftness,
        mode,
        lockPoint: effectiveLock,
      })
    );
  };

  const switchSelectionMode = (next: SelectionMode) => {
    if (next === selectionMode) return;
    clearSelection();
    setSelectionMode(next);
    if (next === "auto-people") {
      void detectPeople();
    } else if (next === "remove-others") {
      void detectOthers();
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
    setLockPoint(null);
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
    syncBrushOverlay();
  }, [layout, selectionMode, imageUrl, syncBrushOverlay, strokeTick]);

  const displaySrc = selectionMode === "brush" ? imageUrl : overlayUrl ?? imageUrl;
  const canRemove =
    (selectionMode === "smart" && points.length > 0) ||
    (selectionMode === "brush" && hasBrushMask) ||
    selectionMode === "auto-people" ||
    selectionMode === "remove-others";

  const canClear =
    (selectionMode === "smart" && points.length > 0) ||
    (selectionMode === "brush" && hasBrushMask) ||
    ((selectionMode === "auto-people" || selectionMode === "remove-others") && overlayUrl !== null);

  const canUndoBrush = selectionMode === "brush" && strokeUndoRef.current.length > 0;
  const canRedoBrush = selectionMode === "brush" && strokeRedoRef.current.length > 0;
  const canUndoSmart = selectionMode === "smart" && points.length > 0;

  const toScreen = (pt: ClickPoint) => {
    if (!layout) return { left: 0, top: 0 };
    return {
      left: layout.offsetX + (pt.x / layout.naturalW) * layout.drawW,
      top: layout.offsetY + (pt.y / layout.naturalH) * layout.drawH,
    };
  };

  const cursorClass =
    selectionMode === "brush" ? "cursor-crosshair" : selectionMode === "smart" ? "cursor-crosshair" : "cursor-default";

  const hint = generativeHint(mode, generative);

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
          onClick={() => switchSelectionMode("remove-others")}
          disabled={loading !== null}
          className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
            selectionMode === "remove-others"
              ? "bg-accent/20 text-accent-hover border border-accent/40"
              : "border border-surface-border text-gray-400 hover:text-white"
          }`}
        >
          <UserMinus className="w-4 h-4" />
          Remove others
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
          Remove all people
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
        <button
          type="button"
          onClick={() => setLockSubject((v) => !v)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
            lockSubject
              ? "bg-emerald-600/20 text-emerald-300 border border-emerald-600/40"
              : "border border-surface-border text-gray-400 hover:text-white"
          }`}
          title="Lock the first selected / largest person so later strokes do not erase them"
        >
          <Lock className="w-3.5 h-3.5" />
          Lock subject{lockPoint ? " ✓" : ""}
        </button>
        {inpaintMode === "ai" && hint.badge && (
          <span className={`ml-auto text-[11px] px-2 py-1 rounded-md border ${BADGE_STYLES[hint.tone]}`}>
            {hint.badge}
          </span>
        )}
      </div>

      {inpaintMode === "quick" && (
        <p className="text-xs text-amber-400/90 bg-amber-950/30 border border-amber-700/30 rounded-lg px-3 py-2">
          Quick = LaMa only (classical, fastest). Switch to AI Remove for the Best Quality erase path.
        </p>
      )}
      {inpaintMode === "ai" && (
        <p className={`text-xs rounded-lg px-3 py-2 ${HINT_STYLES[hint.tone]}`}>{hint.text}</p>
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
        {lockSubject && lockPoint && layout && selectionMode === "smart" && (
          <span
            className="absolute w-4 h-4 rounded-full border-2 border-emerald-300 bg-emerald-500/40 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10"
            style={{
              left: layout.offsetX + (lockPoint.x / layout.naturalW) * layout.drawW,
              top: layout.offsetY + (lockPoint.y / layout.naturalH) * layout.drawH,
            }}
            title="Locked subject"
          />
        )}
        {loading && (
          <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center gap-2 z-20">
            <Loader2 className="w-8 h-8 animate-spin text-white" />
            {progressLabel && <p className="text-sm text-white/90">{progressLabel}</p>}
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
            <span className="whitespace-nowrap">Brush</span>
            <input
              type="range"
              min={6}
              max={64}
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
          onClick={undoStroke}
          disabled={loading !== null || (!canUndoSmart && !canUndoBrush)}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border border-surface-border text-gray-400 hover:text-white disabled:opacity-40"
        >
          <Undo2 className="w-4 h-4" />
          Undo
        </button>
        {selectionMode === "brush" && (
          <button
            type="button"
            onClick={redoStroke}
            disabled={loading !== null || !canRedoBrush}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border border-surface-border text-gray-400 hover:text-white disabled:opacity-40"
          >
            <Redo2 className="w-4 h-4" />
            Redo
          </button>
        )}
        <button
          type="button"
          onClick={clearSelection}
          disabled={loading !== null || !canClear}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border border-surface-border text-gray-400 hover:text-white disabled:opacity-40"
        >
          <RotateCcw className="w-4 h-4" />
          Clear
        </button>
        <label className="flex items-center gap-2 text-sm text-gray-400 ml-auto mr-2">
          <span className="whitespace-nowrap">Feather</span>
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
          {selectionMode === "auto-people"
            ? "Remove people"
            : selectionMode === "remove-others"
              ? "Remove others"
              : "Remove selected"}
        </button>
      </div>

      <p className="text-sm text-gray-500">
        <Trash2 className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
        {selectionMode === "smart" &&
          "Click one person/object at a time. Lock subject protects the hero. After remove, select the next person."}
        {selectionMode === "brush" &&
          "Paint one region, then Remove. Undo/redo strokes. Prefer AI Remove for quality."}
        {selectionMode === "remove-others" &&
          "Keeps the locked / largest person; removes other detected people (library/coastal pattern)."}
        {selectionMode === "auto-people" &&
          "All detected people are highlighted. Prefer Remove others when you want to keep one subject."}
        {lastEngine && !loading && (
          <span className="ml-2 text-gray-600">Last engine: {engineLabel(lastEngine)}</span>
        )}
      </p>
    </div>
  );
}
