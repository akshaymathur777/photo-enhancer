export type CategoryId = "edit" | "portrait" | "looks" | "enhance" | "batch";

export type ToolId =
  | "crop"
  | "blur"
  | "bokeh"
  | "eraser"
  | "stamp"
  | "prism"
  | "vignette"
  | "grain"
  | "relight"
  | "ai-repair"
  | "photo-restore"
  | "portrait-magic"
  | "skin-smooth"
  | "fix-red-eye"
  | "whiten-teeth"
  | "hair-tint"
  | "lip-blush"
  | "ai-styles"
  | "remove-bg"
  | "replace-bg"
  | "remove-objects"
  | "auto-enhance"
  | "adjust"
  | "denoise"
  | "sharpen"
  | "upscale"
  | "batch";

export type AiMode = "best_quality" | "balanced" | "fast";

export type AiStyleId = string;

export type BgModel =
  | "massive"
  | "general"
  | "portrait"
  | "matting"
  | "anime"
  | "bria"
  | "lite";

export interface ToolOptions {
  mode: AiMode;
  model: BgModel;
  style: AiStyleId;
  bgType: "transparent" | "color" | "blur" | "image";
  bgColor: string;
  bgImageFile: File | null;
  blurRadius: number;
  bokehShape: "circle" | "heart" | "hexagon";
  depthAware: boolean;
  strength: number;
  brightness: number;
  contrast: number;
  saturation: number;
  sharpness: number;
  denoiseStrength: number;
  sharpenAmount: number;
  scale: 2 | 4;
  blurAmount: number;
  selectiveBlur: boolean;
  relightPreset: string;
  hairHue: number;
  hairSaturation: number;
  lipColor: string;
  blushColor: string;
  makeupIntensity: number;
  batchInputDir: string;
  batchOutputDir: string;
  batchOperation: string;
  aiRepairAuto: boolean;
}

export const defaultOptions: ToolOptions = {
  mode: "balanced",
  model: "massive",
  style: "cinematic",
  bgType: "color",
  bgColor: "#ffffff",
  bgImageFile: null,
  blurRadius: 20,
  bokehShape: "circle",
  depthAware: false,
  strength: 1.0,
  brightness: 1.0,
  contrast: 1.0,
  saturation: 1.0,
  sharpness: 1.0,
  denoiseStrength: 10,
  sharpenAmount: 1.5,
  scale: 2,
  blurAmount: 15,
  selectiveBlur: false,
  relightPreset: "even_light",
  hairHue: 0,
  hairSaturation: 0.3,
  lipColor: "#cc4444",
  blushColor: "#ff88aa",
  makeupIntensity: 0.35,
  batchInputDir: "",
  batchOutputDir: "",
  batchOperation: "auto_enhance",
  aiRepairAuto: false,
};

const ENDPOINTS: Partial<Record<ToolId, string>> = {
  "remove-bg": "/api/remove-background",
  "replace-bg": "/api/replace-background",
  "ai-styles": "/api/apply-style",
  "auto-enhance": "/api/auto-enhance",
  adjust: "/api/adjust",
  denoise: "/api/denoise",
  sharpen: "/api/sharpen",
  upscale: "/api/upscale",
  bokeh: "/api/bokeh",
  blur: "/api/blur",
  prism: "/api/prism",
  vignette: "/api/vignette",
  grain: "/api/grain",
  relight: "/api/relight",
  "ai-repair": "/api/ai-repair",
  "photo-restore": "/api/photo-restore",
  "portrait-magic": "/api/portrait-magic",
  "skin-smooth": "/api/skin-smooth",
  "fix-red-eye": "/api/fix-red-eye",
  "whiten-teeth": "/api/whiten-teeth",
  "hair-tint": "/api/hair-tint",
  "lip-blush": "/api/lip-blush",
};

export const CANVAS_TOOLS: ToolId[] = ["crop", "stamp", "eraser", "remove-objects"];

export const MAX_OUTPUT_MP = 100_000_000;

export function maxUpscaleScale(width: number, height: number, mode: AiMode = "balanced"): 2 | 4 | null {
  const pixels = width * height;
  const modeCap = mode === "fast" ? 2 : 4;
  if (pixels * 16 <= MAX_OUTPUT_MP && modeCap >= 4) return 4;
  if (pixels * 4 <= MAX_OUTPUT_MP) return Math.min(2, modeCap) as 2 | 4;
  return null;
}

export function upscaleScaleHint(width: number, height: number, scale: 2 | 4): string | null {
  const max = maxUpscaleScale(width, height);
  if (max === null) {
    const srcMp = (width * height) / 1_000_000;
    return `Image is ~${srcMp.toFixed(0)}MP — too large to upscale within the 100MP output limit.`;
  }
  if (scale > max) {
    return `${scale}× exceeds the limit for this image; ${max}× will be used instead.`;
  }
  const outMp = (width * height * scale * scale) / 1_000_000;
  if (outMp > 80) {
    return `Output will be ~${outMp.toFixed(0)}MP — may use significant memory.`;
  }
  return null;
}

async function parseApiError(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const json = JSON.parse(text) as { detail?: string | { msg: string }[] };
    if (typeof json.detail === "string") return json.detail;
    if (Array.isArray(json.detail) && json.detail[0]?.msg) return json.detail[0].msg;
  } catch {
    /* plain text */
  }
  return text || `Request failed (${res.status})`;
}

export interface ProcessResult {
  blob: Blob;
  warning?: string;
  effectiveScale?: number;
}

export async function processImage(
  file: File,
  tool: ToolId,
  options: ToolOptions
): Promise<ProcessResult> {
  if (tool === "batch") {
    const form = new FormData();
    form.append("input_dir", options.batchInputDir);
    form.append("output_dir", options.batchOutputDir);
    form.append("operation", options.batchOperation);
    const res = await fetch("/api/batch/enhance", { method: "POST", body: form });
    if (!res.ok) throw new Error(await parseApiError(res));
    const data = (await res.json()) as { processed: number; errors: string[] };
    return {
      blob: file,
      warning: `Batch complete: ${data.processed} images processed${data.errors.length ? `, ${data.errors.length} errors` : ""}`,
    };
  }

  const endpoint = ENDPOINTS[tool];
  if (!endpoint) throw new Error(`No endpoint for tool: ${tool}`);

  const form = new FormData();
  form.append("file", file);
  form.append("mode", options.mode);

  switch (tool) {
    case "remove-bg":
    case "replace-bg":
    case "bokeh":
    case "blur":
      form.append("model", options.model);
      if (tool === "replace-bg") {
        form.append("bg_type", options.bgType);
        form.append("color", options.bgColor);
        form.append("blur_radius", String(options.blurRadius));
        if (options.bgType === "image" && options.bgImageFile) {
          form.append("bg_image", options.bgImageFile);
        }
      }
      if (tool === "bokeh") {
        form.append("blur_radius", String(options.blurRadius));
        form.append("shape", options.bokehShape);
        form.append("depth_aware", String(options.depthAware));
      }
      if (tool === "blur") {
        form.append("radius", String(options.blurAmount));
        form.append("selective", String(options.selectiveBlur));
      }
      break;
    case "ai-styles":
      form.append("style", options.style);
      break;
    case "auto-enhance":
    case "portrait-magic":
    case "photo-restore":
    case "prism":
    case "vignette":
    case "grain":
    case "relight":
    case "skin-smooth":
    case "whiten-teeth":
      form.append("strength", String(options.strength));
      if (tool === "relight") form.append("preset", options.relightPreset);
      break;
    case "adjust":
      form.append("brightness", String(options.brightness));
      form.append("contrast", String(options.contrast));
      form.append("saturation", String(options.saturation));
      form.append("sharpness", String(options.sharpness));
      break;
    case "denoise":
      form.append("strength", String(options.denoiseStrength));
      break;
    case "sharpen":
      form.append("amount", String(options.sharpenAmount));
      break;
    case "upscale":
      form.append("scale", String(options.scale));
      break;
    case "ai-repair":
      form.append("auto_detect", String(options.aiRepairAuto));
      break;
    case "hair-tint":
      form.append("hue_shift", String(options.hairHue));
      form.append("saturation", String(options.hairSaturation));
      break;
    case "lip-blush":
      form.append("lip_color", options.lipColor);
      form.append("blush_color", options.blushColor);
      form.append("intensity", String(options.makeupIntensity));
      break;
  }

  const res = await fetch(endpoint, { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseApiError(res));

  const blob = await res.blob();
  if (tool === "upscale") {
    const warning = res.headers.get("X-Upscale-Warning") ?? undefined;
    const scaleHeader = res.headers.get("X-Upscale-Scale");
    const effectiveScale = scaleHeader ? parseInt(scaleHeader, 10) : undefined;
    return { blob, warning, effectiveScale };
  }
  return { blob };
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function blobFromUrl(url: string): Promise<Blob> {
  return fetch(url).then((r) => r.blob());
}

const IMAGE_EXTENSIONS = new Set([
  "jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff", "avif", "heic", "heif", "hif",
]);

export function isImageFile(file: File): boolean {
  if (file.type.startsWith("image/")) return true;
  const ext = file.name.toLowerCase().split(".").pop();
  return !!ext && IMAGE_EXTENSIONS.has(ext);
}

export function needsServerPreview(file: File): boolean {
  const ext = file.name.toLowerCase().split(".").pop();
  return ext === "heic" || ext === "heif" || ext === "hif" || !file.type.startsWith("image/");
}

export async function openImageFile(file: File): Promise<{ blob: Blob; file: File }> {
  if (!needsServerPreview(file)) {
    return { blob: file, file };
  }
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/open", { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(await parseApiError(res));
  }
  const blob = await res.blob();
  const base = file.name.replace(/\.[^.]+$/, "") || "photo";
  const outFile = new File([blob], `${base}.jpg`, { type: blob.type || "image/jpeg" });
  return { blob, file: outFile };
}

export interface ModeInfo {
  id: AiMode;
  name: string;
  description: string;
}

export interface StyleInfo {
  id: AiStyleId;
  name: string;
  description: string;
  tags?: string[];
}

export interface RelightPreset {
  id: string;
  name: string;
  description: string;
}

export async function fetchModes(): Promise<ModeInfo[]> {
  const res = await fetch("/api/modes");
  if (!res.ok) return [];
  const data = (await res.json()) as { modes: ModeInfo[] };
  return data.modes ?? [];
}

export async function fetchStyles(): Promise<StyleInfo[]> {
  const res = await fetch("/api/styles");
  if (!res.ok) return [];
  const data = (await res.json()) as { styles: StyleInfo[] };
  return data.styles ?? [];
}

export async function fetchRelightPresets(): Promise<RelightPreset[]> {
  const res = await fetch("/api/relight/presets");
  if (!res.ok) return [];
  const data = (await res.json()) as { presets: RelightPreset[] };
  return data.presets ?? [];
}

export interface ModelInfo {
  id: string;
  name: string;
  size_estimate_mb: number;
  feature: string;
  status: "present" | "missing" | "downloading" | "error";
}

export interface ModelsStatus {
  models: ModelInfo[];
  downloaded_count: number;
  total_count: number;
  all_present: boolean;
  disk_used_bytes: number;
  disk_used_mb: number;
  total_estimate_mb: number;
  models_dir: string;
  download_in_progress: boolean;
}

export interface DownloadProgress {
  status: "idle" | "downloading" | "complete" | "error";
  current_model_id: string | null;
  current_model_name: string | null;
  bytes_downloaded: number;
  bytes_total: number;
  percent: number;
  model_statuses: Record<string, string>;
  error: string | null;
  message: string | null;
}

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  const res = await fetch("/api/models/status");
  if (!res.ok) throw new Error(await parseApiError(res));
  return (await res.json()) as ModelsStatus;
}

export async function fetchDownloadProgress(): Promise<DownloadProgress> {
  const res = await fetch("/api/models/download-progress");
  if (!res.ok) throw new Error(await parseApiError(res));
  return (await res.json()) as DownloadProgress;
}

export async function startDownloadAll(): Promise<{ started: boolean; message: string }> {
  const res = await fetch("/api/models/download-all", { method: "POST" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return (await res.json()) as { started: boolean; message: string };
}

export async function startDownloadModel(modelId: string): Promise<{ started: boolean; message: string }> {
  const res = await fetch(`/api/models/download/${modelId}`, { method: "POST" });
  if (!res.ok) throw new Error(await parseApiError(res));
  return (await res.json()) as { started: boolean; message: string };
}
