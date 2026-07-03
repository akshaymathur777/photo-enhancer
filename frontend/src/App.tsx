import {
  Columns2,
  FolderOpen,
  HardDrive,
  ImagePlus,
  Loader2,
  Redo2,
  Save,
  Sparkles,
  Undo2,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AiMode,
  blobFromUrl,
  CANVAS_TOOLS,
  CategoryId,
  defaultOptions,
  downloadBlob,
  fetchModes,
  fetchModelsStatus,
  fetchRelightPresets,
  fetchStyles,
  isImageFile,
  maxUpscaleScale,
  openImageFile,
  processImage,
  RelightPreset,
  StyleInfo,
  ToolId,
  ToolOptions,
  upscaleScaleHint,
} from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { CompareSlider } from "./components/CompareSlider";
import { CropEditor } from "./components/CropEditor";
import { FilterGallery } from "./components/FilterGallery";
import { ModelDownloadPanel, shouldShowOnboarding } from "./components/ModelDownloadPanel";
import { ObjectRemovalEditor } from "./components/ObjectRemovalEditor";
import { StampEditor } from "./components/StampEditor";
import { useHistory } from "./hooks/useHistory";

type ToolDef = { id: ToolId; label: string; desc: string };

const CATEGORIES: { id: CategoryId; label: string; tools: ToolDef[] }[] = [
  {
    id: "edit",
    label: "Edit",
    tools: [
      { id: "crop", label: "Crop", desc: "Crop & rotate" },
      { id: "blur", label: "Blur", desc: "Global or selective" },
      { id: "bokeh", label: "Bokeh", desc: "Background blur" },
      { id: "eraser", label: "Eraser", desc: "Quick object removal" },
      { id: "stamp", label: "Stamp", desc: "Clone stamp" },
      { id: "prism", label: "Prism", desc: "Rainbow refraction" },
      { id: "vignette", label: "Vignette", desc: "Edge darkening" },
      { id: "grain", label: "Grain", desc: "Film grain" },
      { id: "relight", label: "Relight", desc: "Lighting presets" },
      { id: "ai-repair", label: "AI Repair", desc: "Fix scratches" },
      { id: "photo-restore", label: "Restore", desc: "Denoise + sharpen" },
    ],
  },
  {
    id: "portrait",
    label: "Portrait",
    tools: [
      { id: "portrait-magic", label: "Magic", desc: "One-tap portrait" },
      { id: "skin-smooth", label: "Skin", desc: "Face-aware smooth" },
      { id: "fix-red-eye", label: "Red Eye", desc: "Fix flash eyes" },
      { id: "whiten-teeth", label: "Teeth", desc: "Whiten smile" },
      { id: "hair-tint", label: "Hair Tint", desc: "Simple recolor" },
      { id: "lip-blush", label: "Lip/Blush", desc: "Basic overlay" },
    ],
  },
  {
    id: "looks",
    label: "Looks",
    tools: [{ id: "ai-styles", label: "Filters", desc: "30+ presets" }],
  },
  {
    id: "enhance",
    label: "Enhance",
    tools: [
      { id: "remove-bg", label: "Remove BG", desc: "BiRefNet Massive" },
      { id: "replace-bg", label: "Replace BG", desc: "Color, blur, image" },
      { id: "remove-objects", label: "Remove Objects", desc: "SAM + LaMa" },
      { id: "auto-enhance", label: "Auto Enhance", desc: "One-click polish" },
      { id: "adjust", label: "Adjust", desc: "Sliders" },
      { id: "denoise", label: "Denoise", desc: "Reduce noise" },
      { id: "sharpen", label: "Sharpen", desc: "Crisp details" },
      { id: "upscale", label: "Upscale", desc: "Real-ESRGAN 2×/4×" },
    ],
  },
  {
    id: "batch",
    label: "Batch",
    tools: [{ id: "batch", label: "Batch", desc: "Folder processing" }],
  },
];

const MODES: { id: AiMode; label: string; desc: string }[] = [
  { id: "best_quality", label: "Best Quality", desc: "Heaviest models, best results" },
  { id: "balanced", label: "Balanced", desc: "Default speed/quality" },
  { id: "fast", label: "Fast", desc: "Lite models, quicker" },
];

const MODE_BG_MODEL: Record<AiMode, ToolOptions["model"]> = {
  best_quality: "massive",
  balanced: "general",
  fast: "lite",
};

const MODELS = [
  { id: "massive", label: "Massive", desc: "Best quality" },
  { id: "bria", label: "BRIA RMBG 2.0", desc: "Top-tier edges" },
  { id: "general", label: "General", desc: "Balanced" },
  { id: "portrait", label: "Portrait", desc: "Hair & skin" },
  { id: "matting", label: "Matting", desc: "Soft edges" },
  { id: "anime", label: "Anime/Art", desc: "Illustrations" },
  { id: "lite", label: "Lite", desc: "Faster" },
] as const;

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="flex justify-between text-sm mb-1.5">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300 font-mono text-xs">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent"
      />
    </label>
  );
}

function extForTool(tool: ToolId, options: ToolOptions): string {
  if (tool === "remove-bg" || tool === "upscale" || options.bgType === "transparent") return "png";
  return "jpg";
}

export default function App() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [workingFile, setWorkingFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [category, setCategory] = useState<CategoryId>("enhance");
  const [tool, setTool] = useState<ToolId>("remove-bg");
  const [options, setOptions] = useState<ToolOptions>(defaultOptions);
  const [loading, setLoading] = useState(false);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [styles, setStyles] = useState<StyleInfo[]>([]);
  const [relightPresets, setRelightPresets] = useState<RelightPreset[]>([]);
  const [modes, setModes] = useState(MODES);
  const [compareMode, setCompareMode] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showModelsPanel, setShowModelsPanel] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const bgImageRef = useRef<HTMLInputElement>(null);
  const history = useHistory();

  const activeTools = CATEGORIES.find((c) => c.id === category)?.tools ?? [];

  useEffect(() => {
    void fetchModes().then((m) => {
      if (m.length) setModes(m.map((item) => ({ id: item.id, label: item.name, desc: item.description })));
    });
    void fetchStyles().then((s) => {
      if (s.length) setStyles(s);
    });
    void fetchRelightPresets().then(setRelightPresets);
    void fetchModelsStatus()
      .then((s) => {
        if (shouldShowOnboarding(s)) setShowOnboarding(true);
      })
      .catch(() => {/* offline or server not ready yet */});
  }, []);

  useEffect(() => {
    if (activeTools.length && !activeTools.some((t) => t.id === tool)) {
      setTool(activeTools[0].id);
    }
  }, [category, activeTools, tool]);

  const isDirty = !!imageUrl && !!originalUrl && imageUrl !== originalUrl;

  const revokeIfNot = (url: string | null, keep: string | null) => {
    if (url && url !== keep) URL.revokeObjectURL(url);
  };

  const commitDocument = useCallback(
    (blob: Blob, mime = "image/jpeg", trackHistory = true) => {
      const name = fileName ?? "photo.jpg";
      const next = new File([blob], name, { type: mime });
      const url = trackHistory ? history.push(blob) : URL.createObjectURL(blob);
      setWorkingFile(next);
      setImageUrl((prev) => {
        revokeIfNot(prev, originalUrl);
        return url;
      });
    },
    [fileName, originalUrl, history]
  );

  const openFile = useCallback(
    async (f: File) => {
      if (!isImageFile(f)) {
        setError(`Unsupported file: ${f.name}`);
        return;
      }
      setOpening(true);
      setError(null);
      setInfo(null);
      try {
        const { blob, file: ready } = await openImageFile(f);
        const url = URL.createObjectURL(blob);
        history.reset(blob);
        const dims = await new Promise<{ w: number; h: number }>((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
          img.onerror = () => reject(new Error("Could not read image dimensions"));
          img.src = url;
        });
        setImageSize(dims);
        const maxScale = maxUpscaleScale(dims.w, dims.h, defaultOptions.mode);
        setOptions((o) => (maxScale !== null && o.scale > maxScale ? { ...o, scale: maxScale } : o));
        setFileName(ready.name);
        setWorkingFile(ready);
        setOriginalFile(ready);
        setImageUrl(url);
        setOriginalUrl(url);
        setCompareMode(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : `Could not open ${f.name}`);
      } finally {
        setOpening(false);
      }
    },
    [history]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f && isImageFile(f)) void openFile(f);
    },
    [openFile]
  );

  const runTool = async () => {
    if (!workingFile || CANVAS_TOOLS.includes(tool)) return;
    setLoading(true);
    setError(null);
    setInfo(null);
    try {
      const { blob, warning, effectiveScale } = await processImage(workingFile, tool, options);
      if (tool !== "batch") {
        commitDocument(blob, blob.type || "image/jpeg");
      }
      if (warning) setInfo(warning);
      if (tool === "upscale" && imageSize) {
        const scale = effectiveScale ?? options.scale;
        setImageSize({ w: imageSize.w * scale, h: imageSize.h * scale });
        if (effectiveScale && effectiveScale !== options.scale) {
          setOptions((o) => ({ ...o, scale: effectiveScale as 2 | 4 }));
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Processing failed");
    } finally {
      setLoading(false);
    }
  };

  const saveImage = async (asNew = false) => {
    if (!imageUrl || !fileName) return;
    const blob = await blobFromUrl(imageUrl);
    const ext = fileName.includes(".") ? fileName.split(".").pop() : extForTool(tool, options);
    const base = fileName.replace(/\.[^.]+$/, "") || "photo";
    downloadBlob(blob, asNew ? `${base}-edited.${ext}` : fileName);
  };

  const revert = () => {
    if (!originalFile || !originalUrl) return;
    setWorkingFile(originalFile);
    setImageUrl((prev) => {
      revokeIfNot(prev, originalUrl);
      return originalUrl;
    });
    history.reset(originalFile);
    setError(null);
    setCompareMode(false);
  };

  const handleUndo = () => {
    const entry = history.undo();
    if (!entry) return;
    const f = new File([entry.blob], fileName ?? "photo.jpg", { type: entry.blob.type || "image/jpeg" });
    setWorkingFile(f);
    setImageUrl((prev) => {
      revokeIfNot(prev, originalUrl);
      return entry.url;
    });
  };

  const handleRedo = () => {
    const entry = history.redo();
    if (!entry) return;
    const f = new File([entry.blob], fileName ?? "photo.jpg", { type: entry.blob.type || "image/jpeg" });
    setWorkingFile(f);
    setImageUrl((prev) => {
      revokeIfNot(prev, originalUrl);
      return entry.url;
    });
  };

  const setMode = (mode: AiMode) => {
    setOptions((o) => {
      const maxScale = imageSize ? maxUpscaleScale(imageSize.w, imageSize.h, mode) : 4;
      return {
        ...o,
        mode,
        model: MODE_BG_MODEL[mode],
        scale: maxScale !== null && o.scale > maxScale ? maxScale : o.scale,
      };
    });
  };

  const setOpt = <K extends keyof ToolOptions>(key: K, value: ToolOptions[K]) => {
    setOptions((o) => ({ ...o, [key]: value }));
  };

  const showCanvasEditor = CANVAS_TOOLS.includes(tool) || tool === "eraser";

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <header className="border-b border-surface-border bg-surface-raised shrink-0">
        <div className="flex items-center h-12 px-4 gap-1">
          <div className="flex items-center gap-2.5 pr-4 mr-2 border-r border-surface-border">
            <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4" />
            </div>
            <span className="font-semibold text-sm hidden sm:block">Photo Enhancer</span>
          </div>
          <button type="button" onClick={() => inputRef.current?.click()} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5">
            <FolderOpen className="w-4 h-4" /> Open
          </button>
          <button type="button" onClick={() => saveImage(false)} disabled={!workingFile} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 disabled:opacity-40">
            <Save className="w-4 h-4" /> Save
          </button>
          <button type="button" onClick={() => saveImage(true)} disabled={!workingFile} className="px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 disabled:opacity-40">
            Save As…
          </button>
          <button type="button" onClick={handleUndo} disabled={!history.canUndo} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 disabled:opacity-40">
            <Undo2 className="w-4 h-4" /> Undo
          </button>
          <button type="button" onClick={handleRedo} disabled={!history.canRedo} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 disabled:opacity-40">
            <Redo2 className="w-4 h-4" /> Redo
          </button>
          <button
            type="button"
            onClick={() => setCompareMode((c) => !c)}
            disabled={!isDirty}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors disabled:opacity-40 ${
              compareMode ? "bg-accent/20 text-accent-hover" : "text-gray-300 hover:bg-white/5"
            }`}
          >
            <Columns2 className="w-4 h-4" /> Compare
          </button>
          <button type="button" onClick={revert} disabled={!isDirty} className="px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5 disabled:opacity-40">
            Revert
          </button>
          <button
            type="button"
            onClick={() => setShowModelsPanel(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-gray-300 hover:bg-white/5"
            title="Manage AI models"
          >
            <HardDrive className="w-4 h-4" />
            <span className="hidden sm:inline">Models</span>
          </button>
          <div className="ml-auto text-xs text-gray-500 truncate max-w-[30%]">
            {fileName ?? "No image open"}
            {isDirty && <span className="text-accent ml-1">• edited</span>}
          </div>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Category rail */}
        <nav className="w-16 shrink-0 border-r border-surface-border bg-surface-raised/80 flex flex-col items-center py-3 gap-1">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              type="button"
              onClick={() => setCategory(cat.id)}
              className={`w-12 py-2 rounded-lg text-[10px] font-semibold uppercase tracking-wide transition-colors ${
                category === cat.id ? "bg-accent/20 text-accent-hover" : "text-gray-500 hover:bg-white/5"
              }`}
            >
              {cat.label}
            </button>
          ))}
        </nav>

        {/* Tool sidebar */}
        <aside className="w-56 shrink-0 border-r border-surface-border bg-surface-raised/50 overflow-y-auto p-3 space-y-4">
          <section>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">AI Mode</h2>
            <select
              value={options.mode}
              onChange={(e) => setMode(e.target.value as AiMode)}
              className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm"
            >
              {modes.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </section>

          <section>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">{CATEGORIES.find((c) => c.id === category)?.label}</h2>
            <div className="space-y-0.5">
              {activeTools.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setTool(t.id)}
                  className={`w-full text-left px-2.5 py-2 rounded-lg text-sm transition-colors ${
                    tool === t.id ? "bg-accent/15 text-accent-hover border border-accent/30" : "hover:bg-white/5 text-gray-300 border border-transparent"
                  }`}
                >
                  <div className="font-medium">{t.label}</div>
                  <div className="text-[10px] text-gray-500">{t.desc}</div>
                </button>
              ))}
            </div>
          </section>

          <section className="space-y-3 pt-2 border-t border-surface-border">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500">Options</h2>

            {(tool === "remove-bg" || tool === "replace-bg" || tool === "bokeh" || tool === "blur") && (
              <>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Model</label>
                  <select value={options.model} onChange={(e) => setOpt("model", e.target.value as ToolOptions["model"])} className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm">
                    {MODELS.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </select>
                </div>
                {tool === "replace-bg" && (
                  <>
                    <select value={options.bgType} onChange={(e) => setOpt("bgType", e.target.value as ToolOptions["bgType"])} className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm">
                      <option value="color">Solid color</option>
                      <option value="blur">Blurred original</option>
                      <option value="image">Custom image</option>
                      <option value="transparent">Transparent PNG</option>
                    </select>
                    {options.bgType === "color" && (
                      <input type="color" value={options.bgColor} onChange={(e) => setOpt("bgColor", e.target.value)} className="w-full h-9 rounded-lg cursor-pointer" />
                    )}
                    {options.bgType === "blur" && (
                      <Slider label="Blur radius" value={options.blurRadius} min={5} max={50} step={1} onChange={(v) => setOpt("blurRadius", v)} />
                    )}
                    {options.bgType === "image" && (
                      <button type="button" onClick={() => bgImageRef.current?.click()} className="w-full py-1.5 text-sm rounded-lg border border-surface-border hover:border-accent/30">
                        {options.bgImageFile ? options.bgImageFile.name : "Choose background image"}
                      </button>
                    )}
                  </>
                )}
                {tool === "bokeh" && (
                  <>
                    <Slider label="Blur strength" value={options.blurRadius} min={5} max={60} step={1} onChange={(v) => setOpt("blurRadius", v)} />
                    <select value={options.bokehShape} onChange={(e) => setOpt("bokehShape", e.target.value as ToolOptions["bokehShape"])} className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm">
                      <option value="circle">Circle</option>
                      <option value="heart">Heart</option>
                      <option value="hexagon">Hexagon</option>
                    </select>
                    <label className="flex items-center gap-2 text-sm text-gray-400">
                      <input type="checkbox" checked={options.depthAware} onChange={(e) => setOpt("depthAware", e.target.checked)} className="accent-accent" />
                      Depth-aware falloff
                    </label>
                  </>
                )}
                {tool === "blur" && (
                  <>
                    <Slider label="Blur radius" value={options.blurAmount} min={3} max={50} step={1} onChange={(v) => setOpt("blurAmount", v)} />
                    <label className="flex items-center gap-2 text-sm text-gray-400">
                      <input type="checkbox" checked={options.selectiveBlur} onChange={(e) => setOpt("selectiveBlur", e.target.checked)} className="accent-accent" />
                      Blur background only
                    </label>
                  </>
                )}
              </>
            )}

            {tool === "ai-styles" && styles.length > 0 && (
              <FilterGallery styles={styles} selected={options.style} onSelect={(id) => setOpt("style", id)} />
            )}

            {tool === "relight" && relightPresets.length > 0 && (
              <select value={options.relightPreset} onChange={(e) => setOpt("relightPreset", e.target.value)} className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm">
                {relightPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            )}

            {["auto-enhance", "portrait-magic", "photo-restore", "prism", "vignette", "grain", "relight", "skin-smooth", "whiten-teeth"].includes(tool) && (
              <Slider label="Strength" value={options.strength} min={0.2} max={2} step={0.1} onChange={(v) => setOpt("strength", v)} />
            )}

            {tool === "ai-repair" && (
              <label className="flex items-center gap-2 text-sm text-gray-400">
                <input type="checkbox" checked={options.aiRepairAuto} onChange={(e) => setOpt("aiRepairAuto", e.target.checked)} className="accent-accent" />
                Auto-detect scratches
              </label>
            )}

            {tool === "hair-tint" && (
              <>
                <Slider label="Hue shift" value={options.hairHue} min={-30} max={30} step={1} onChange={(v) => setOpt("hairHue", v)} />
                <Slider label="Saturation" value={options.hairSaturation} min={0} max={1} step={0.05} onChange={(v) => setOpt("hairSaturation", v)} />
              </>
            )}

            {tool === "lip-blush" && (
              <>
                <label className="text-xs text-gray-400">Lip color<input type="color" value={options.lipColor} onChange={(e) => setOpt("lipColor", e.target.value)} className="w-full h-8 mt-1 rounded" /></label>
                <label className="text-xs text-gray-400">Blush color<input type="color" value={options.blushColor} onChange={(e) => setOpt("blushColor", e.target.value)} className="w-full h-8 mt-1 rounded" /></label>
                <Slider label="Intensity" value={options.makeupIntensity} min={0.1} max={0.8} step={0.05} onChange={(v) => setOpt("makeupIntensity", v)} />
              </>
            )}

            {tool === "adjust" && (
              <>
                <Slider label="Brightness" value={options.brightness} min={0.5} max={1.5} step={0.05} onChange={(v) => setOpt("brightness", v)} />
                <Slider label="Contrast" value={options.contrast} min={0.5} max={1.5} step={0.05} onChange={(v) => setOpt("contrast", v)} />
                <Slider label="Saturation" value={options.saturation} min={0} max={2} step={0.05} onChange={(v) => setOpt("saturation", v)} />
                <Slider label="Sharpness" value={options.sharpness} min={0.5} max={2} step={0.05} onChange={(v) => setOpt("sharpness", v)} />
              </>
            )}

            {tool === "denoise" && (
              <Slider label="Strength" value={options.denoiseStrength} min={3} max={21} step={2} onChange={(v) => setOpt("denoiseStrength", v)} />
            )}

            {tool === "sharpen" && (
              <Slider label="Amount" value={options.sharpenAmount} min={0.5} max={3} step={0.1} onChange={(v) => setOpt("sharpenAmount", v)} />
            )}

            {tool === "upscale" && (
              <div className="space-y-2">
                <div className="flex gap-2">
                  {([2, 4] as const).map((s) => {
                    const maxScale = imageSize ? maxUpscaleScale(imageSize.w, imageSize.h, options.mode) : 4;
                    const disabled = maxScale !== null && s > maxScale;
                    return (
                      <button key={s} type="button" disabled={disabled} onClick={() => setOpt("scale", s)} className={`flex-1 py-1.5 rounded-lg text-sm ${options.scale === s ? "bg-accent text-white" : "bg-surface border border-surface-border"} ${disabled ? "opacity-40" : ""}`}>
                        {s}×
                      </button>
                    );
                  })}
                </div>
                {imageSize && upscaleScaleHint(imageSize.w, imageSize.h, options.scale) && (
                  <p className="text-[11px] text-amber-400/90">{upscaleScaleHint(imageSize.w, imageSize.h, options.scale)}</p>
                )}
              </div>
            )}

            {tool === "batch" && (
              <>
                <input type="text" placeholder="Input folder path" value={options.batchInputDir} onChange={(e) => setOpt("batchInputDir", e.target.value)} className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm" />
                <input type="text" placeholder="Output folder path" value={options.batchOutputDir} onChange={(e) => setOpt("batchOutputDir", e.target.value)} className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm" />
                <select value={options.batchOperation} onChange={(e) => setOpt("batchOperation", e.target.value)} className="w-full bg-surface border border-surface-border rounded-lg px-2 py-1.5 text-sm">
                  <option value="auto_enhance">Auto Enhance</option>
                  <option value="portrait_magic">Portrait Magic</option>
                  <option value="photo_restore">Photo Restore</option>
                </select>
              </>
            )}

            {!showCanvasEditor && (
              <button
                type="button"
                onClick={runTool}
                disabled={!workingFile || loading || (tool === "upscale" && imageSize !== null && maxUpscaleScale(imageSize.w, imageSize.h, options.mode) === null)}
                className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-40 text-sm font-semibold"
              >
                {loading ? <><Loader2 className="w-4 h-4 animate-spin" /> Processing…</> : tool === "batch" ? "Run batch" : "Apply to image"}
              </button>
            )}
          </section>
        </aside>

        {/* Canvas */}
        <main className="flex-1 flex flex-col min-w-0 bg-surface p-4 relative">
          {opening && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
              <Loader2 className="w-8 h-8 animate-spin text-white" />
            </div>
          )}

          {!imageUrl ? (
            <div onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} onClick={() => inputRef.current?.click()} className="flex-1 border-2 border-dashed border-surface-border rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-accent/40">
              <ImagePlus className="w-10 h-10 text-accent mb-3" />
              <p className="font-medium">Open an image to get started</p>
              <p className="text-sm text-gray-500 mt-1">Drop a file or click Open</p>
            </div>
          ) : compareMode && originalUrl && imageUrl ? (
            <CompareSlider before={originalUrl} after={imageUrl} />
          ) : showCanvasEditor && workingFile ? (
            <div className="flex-1 min-h-0">
              {tool === "crop" && (
                <CropEditor imageUrl={imageUrl} file={workingFile} onResult={(blob) => commitDocument(blob)} onError={setError} />
              )}
              {tool === "stamp" && (
                <StampEditor imageUrl={imageUrl} onResult={(blob) => commitDocument(blob)} />
              )}
              {(tool === "remove-objects" || tool === "eraser") && (
                <ObjectRemovalEditor
                  imageUrl={imageUrl}
                  file={workingFile}
                  mode={options.mode}
                  onResult={(blob) => commitDocument(blob, "image/jpeg")}
                  onError={setError}
                  onWorkingFileChange={setWorkingFile}
                />
              )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center min-h-0 rounded-xl border border-surface-border checkerboard overflow-hidden relative">
              <img src={imageUrl} alt="Document" className="max-w-full max-h-full object-contain" draggable={false} />
              {loading && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                  <Loader2 className="w-10 h-10 animate-spin text-white" />
                </div>
              )}
            </div>
          )}

          {info && <div className="mt-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-200 px-3 py-2 text-sm">{info}</div>}
          {error && <div className="mt-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 px-3 py-2 text-sm">{error}</div>}
        </main>

        {workingFile && imageUrl && (
          <aside className="w-72 xl:w-80 shrink-0 hidden md:flex flex-col min-h-0 border-l border-surface-border">
            <ChatPanel
              file={workingFile}
              onApplyImage={(blob) => commitDocument(blob, blob.type || "image/jpeg")}
              onNeedsObjectRemoval={() => { setCategory("edit"); setTool("eraser"); }}
              onRevert={revert}
            />
          </aside>
        )}
      </div>

      {/* Horizontal tool strip */}
      {workingFile && (
        <div className="shrink-0 border-t border-surface-border bg-surface-raised/80 px-4 py-2 flex gap-2 overflow-x-auto">
          {activeTools.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTool(t.id)}
              className={`shrink-0 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                tool === t.id ? "bg-accent text-white" : "bg-surface border border-surface-border text-gray-400 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      <input ref={inputRef} type="file" accept="image/*,.heic,.heif" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) void openFile(f); e.target.value = ""; }} />
      <input ref={bgImageRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) setOpt("bgImageFile", f); e.target.value = ""; }} />

      {showOnboarding && (
        <ModelDownloadPanel
          mode="onboarding"
          onClose={() => setShowOnboarding(false)}
        />
      )}

      {showModelsPanel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-lg max-h-[90vh]">
            <ModelDownloadPanel mode="settings" onClose={() => setShowModelsPanel(false)} />
          </div>
        </div>
      )}
    </div>
  );
}
