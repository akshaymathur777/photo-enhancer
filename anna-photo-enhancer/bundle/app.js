// Reliclear — local-first photo cleanup + browser Edit/Looks/Effects.
import { AnnaAppRuntime } from "/static/anna-apps/_sdk/latest/index.js";
import {
  LOOKS,
  adjustImage,
  applyLook,
  blurImage,
  cropImage,
  denoiseImage,
  grainImage,
  prismImage,
  restoreImage,
  sharpenImage,
  transformGeometry,
  vignetteImage,
} from "./canvas_ops.js";

const CATEGORIES = {
  clean: ["enhance", "remove_bg"],
  edit: ["crop", "rotate", "adjust", "sharpen", "denoise", "restore"],
  looks: ["looks"],
  effects: ["blur", "vignette", "grain", "prism"],
};

const MODES = {
  enhance: {
    label: "Enhance",
    help: "Improve clarity, color, and detail.",
    running: "Enhancing…",
    engine: "local_or_host",
    prompt:
      "Enhance this photograph: improve clarity, natural detail, color balance, and mild sharpening. Keep the subject, composition, and identity unchanged. Do not add or remove objects. Photorealistic output.",
  },
  remove_bg: {
    label: "Remove BG",
    help: "AI cutout of the subject; transparent when possible.",
    running: "Removing background…",
    engine: "local_or_host",
    prompt:
      "Remove the background completely. Keep the main subject with clean edges. Output a transparent PNG when possible; otherwise a clean cutout on a plain white background. Do not alter the subject.",
  },
  crop: {
    label: "Crop",
    help: "Trim edges with the sliders, then Process.",
    running: "Cropping…",
    engine: "canvas",
  },
  rotate: {
    label: "Rotate / Flip",
    help: "Rotate or flip instantly — updates the working photo.",
    running: "Transforming…",
    engine: "canvas_instant",
  },
  adjust: {
    label: "Adjust",
    help: "Brightness, contrast, saturation, exposure, sharpness.",
    running: "Adjusting…",
    engine: "canvas",
  },
  sharpen: {
    label: "Sharpen",
    help: "Crisp detail with unsharp masking.",
    running: "Sharpening…",
    engine: "canvas",
  },
  denoise: {
    label: "Denoise",
    help: "Soft noise reduction (browser lite).",
    running: "Denoising…",
    engine: "canvas",
  },
  restore: {
    label: "Restore",
    help: "Denoise + sharpen for older or soft shots.",
    running: "Restoring…",
    engine: "canvas",
  },
  looks: {
    label: "Looks",
    help: "Apply a classical look preset (browser).",
    running: "Applying look…",
    engine: "canvas",
  },
  blur: {
    label: "Blur",
    help: "Global soft blur.",
    running: "Blurring…",
    engine: "canvas",
  },
  vignette: {
    label: "Vignette",
    help: "Darken the edges.",
    running: "Applying vignette…",
    engine: "canvas",
  },
  grain: {
    label: "Grain",
    help: "Add subtle film grain.",
    running: "Adding grain…",
    engine: "canvas",
  },
  prism: {
    label: "Prism",
    help: "Rainbow refraction shift.",
    running: "Applying prism…",
    engine: "canvas",
  },
  // Kept for host-prompt compatibility / later unhide (desktop SAM+SDXL).
  remove_object: {
    label: "Remove object",
    help: "Desktop-only for now — needs generative inpaint models.",
    running: "Removing object…",
    engine: "deferred",
    promptFor: (text) =>
      `Remove the following from this photograph and inpaint the area so it looks natural and continuous with the surroundings: ${text}. Preserve everything else. Photorealistic, no artifacts.`,
  },
};

const SHIPPED_MODES = [
  "enhance",
  "remove_bg",
  "crop",
  "rotate",
  "adjust",
  "sharpen",
  "denoise",
  "restore",
  "looks",
  "blur",
  "vignette",
  "grain",
  "prism",
];

const DEFAULT_OPTS = {
  crop: { left: 0.05, top: 0.05, right: 0.05, bottom: 0.05 },
  adjust: { brightness: 1, contrast: 1, saturation: 1, exposure: 1, sharpness: 1 },
  sharpen: { amount: 1.5 },
  denoise: { strength: 10 },
  looks: { style: "cinematic" },
  blur: { radius: 8 },
  vignette: { strength: 0.5 },
  grain: { amount: 0.04 },
  prism: { strength: 0.6 },
};

const INLINE_SOFT_LIMIT = 7.5 * 1024 * 1024;
// Forum #400 interim: GPT Image 2 edit route broken until ~2026-10-30 host rollout.
const EDIT_MODEL = "fal-ai/nano-banana-2";
/** Longest edge for HEIC→JPEG; matches typical enhance/upload size and speeds WASM decode. */
const HEIC_MAX_EDGE = 2048;
const JPEG_QUALITY = 0.85;
const ACCEPTED_EXT = /\.(jpe?g|png|webp|hei[cf])$/i;
const HEIC_MIME = /^image\/hei[cf]$/i;
const HEIC_EXT = /\.hei[cf]$/i;

/**
 * Tool ids Anna ACL will accept for the required local executa.
 * Prefer declared bundled/logical handles + published mint map; never lead with tool-dev-*.
 */
function localToolCandidates() {
  const minted =
    typeof window !== "undefined" &&
    window.__ANNA_TOOL_IDS__ &&
    window.__ANNA_TOOL_IDS__["photo-enhancer-local"];
  const ids = [
    "bundled:photo-enhancer-local",
    "photo-enhancer-local",
  ];
  if (typeof minted === "string" && minted) ids.push(minted);
  // Local harness only — must stay last so production never ACL-blocks on undeclared tool-dev-*.
  ids.push("tool-dev-photo-enhancer-local");
  return [...new Set(ids)];
}

function isToolUnavailableError(err) {
  const msg = String(err?.message || "");
  const code = String(err?.code || err?.name || "");
  const blob = `${code} ${msg}`;
  return /DENIED|NOT_FOUND|unknown|invalid.*tool|APP_NOT_GRANTED|ACL|not a declared|permission_denied/i.test(
    blob
  );
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function fileToBase64(file) {
  const buf = await file.arrayBuffer();
  return bytesToBase64(new Uint8Array(buf));
}

function fileNameLower(file) {
  return String(file?.name || "").toLowerCase();
}

/** HEIC/HEIF by MIME or extension (Windows often leaves MIME empty). */
function isHeicFile(file) {
  if (!file) return false;
  const type = String(file.type || "").toLowerCase();
  if (HEIC_MIME.test(type)) return true;
  return HEIC_EXT.test(fileNameLower(file));
}

/** Accept JPG/PNG/WEBP/HEIC even when the OS reports an empty MIME type. */
function isAcceptedImageFile(file) {
  if (!file) return false;
  const type = String(file.type || "").toLowerCase();
  if (type.startsWith("image/")) return true;
  // Windows HEIC (and occasionally JPEG) often arrives as "" or octet-stream.
  if (!type || type === "application/octet-stream") {
    return ACCEPTED_EXT.test(fileNameLower(file));
  }
  return false;
}

let heic2anyLoader = null;
/** Cache HEIC→JPEG by original File identity so preview + process never convert twice. */
const heicJpegCache = new WeakMap();

/** Lazy-load vendored heic2any (UMD → window.heic2any) only when needed. */
function loadHeic2Any() {
  if (typeof window !== "undefined" && typeof window.heic2any === "function") {
    return Promise.resolve(window.heic2any);
  }
  if (heic2anyLoader) return heic2anyLoader;
  heic2anyLoader = new Promise((resolve, reject) => {
    const existing = document.querySelector("script[data-reliclear-heic2any]");
    if (existing) {
      if (typeof window.heic2any === "function") {
        resolve(window.heic2any);
        return;
      }
      existing.addEventListener("load", () => {
        if (typeof window.heic2any === "function") resolve(window.heic2any);
        else reject(new Error("HEIC converter loaded but unavailable."));
      });
      existing.addEventListener("error", () =>
        reject(new Error("Failed to load HEIC converter."))
      );
      return;
    }
    const script = document.createElement("script");
    script.src = new URL("./vendor/heic2any.min.js", import.meta.url).href;
    script.async = true;
    script.dataset.reliclearHeic2any = "1";
    script.onload = () => {
      if (typeof window.heic2any !== "function") {
        reject(new Error("HEIC converter loaded but unavailable."));
        return;
      }
      resolve(window.heic2any);
    };
    script.onerror = () => reject(new Error("Failed to load HEIC converter."));
    document.head.appendChild(script);
  }).catch((err) => {
    heic2anyLoader = null;
    throw err;
  });
  return heic2anyLoader;
}

/** Warm heic2any in the background so the first HEIC pick skips the ~1.3MB download wait. */
function preloadHeic2Any() {
  try {
    loadHeic2Any().catch(() => {});
  } catch {
    /* preload is best-effort */
  }
}

function fitWithinMaxEdge(width, height, maxEdge = HEIC_MAX_EDGE) {
  const w = Math.max(1, width | 0);
  const h = Math.max(1, height | 0);
  const longest = Math.max(w, h);
  if (longest <= maxEdge) return { width: w, height: h };
  const scale = maxEdge / longest;
  return {
    width: Math.max(1, Math.round(w * scale)),
    height: Math.max(1, Math.round(h * scale)),
  };
}

async function bitmapToJpegFile(source, nameHint) {
  const bitmap = await createImageBitmap(source);
  try {
    const size = fitWithinMaxEdge(bitmap.width, bitmap.height);
    const canvas = document.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas unavailable for JPEG re-encode.");
    ctx.drawImage(bitmap, 0, 0, size.width, size.height);
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("JPEG encode failed."))),
        "image/jpeg",
        JPEG_QUALITY
      );
    });
    const base =
      String(nameHint || "photo").replace(/\.[^.]+$/, "") || "photo";
    return new File([blob], `${base}.jpg`, {
      type: "image/jpeg",
      lastModified: Date.now(),
    });
  } finally {
    bitmap.close?.();
  }
}

async function heicToJpegFile(file) {
  // Start script fetch immediately; decode + canvas downscale runs after.
  const heic2anyPromise = loadHeic2Any();
  const heic2any = await heic2anyPromise;
  let converted = await heic2any({
    blob: file,
    toType: "image/jpeg",
    quality: JPEG_QUALITY,
  });
  if (Array.isArray(converted)) converted = converted[0];
  // heic2any returns full-res; downscale via canvas for faster encode + smaller uploads.
  const base = String(file.name || "photo").replace(/\.hei[cf]$/i, "") || "photo";
  try {
    return await bitmapToJpegFile(converted, base);
  } catch {
    return new File([converted], `${base}.jpg`, {
      type: "image/jpeg",
      lastModified: Date.now(),
    });
  }
}

/**
 * Convert HEIC/HEIF → JPEG in the UI so host upload + local tools never see HEIC.
 * Prefer native decode (Safari); otherwise use vendored heic2any (Chrome/Windows).
 * Non-HEIC files pass through unchanged — no binary rebuild required.
 * Results are cached on the original File so pick + process never convert twice.
 */
async function ensureProcessableImage(file) {
  if (!isHeicFile(file)) return file;
  const cached = heicJpegCache.get(file);
  if (cached) return cached;

  const converting = (async () => {
    try {
      return await bitmapToJpegFile(file, file.name);
    } catch {
      /* Most Chromium builds cannot decode HEIC natively. */
    }
    try {
      return await heicToJpegFile(file);
    } catch {
      throw new Error(
        "Could not convert this HEIC photo. Try exporting as JPG or PNG, then upload again."
      );
    }
  })();

  heicJpegCache.set(file, converting);
  try {
    const ready = await converting;
    heicJpegCache.set(file, ready);
    return ready;
  } catch (err) {
    heicJpegCache.delete(file);
    throw err;
  }
}

/** Re-encode any image File as PNG base64 for local classical tools. */
async function fileToPngBase64(file) {
  // Always go through bitmap→canvas so truncated / odd PNGs become PIL-safe.
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas unavailable for PNG re-encode.");
    ctx.drawImage(bitmap, 0, 0);
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("PNG encode failed."))),
        "image/png"
      );
    });
    return fileToBase64(blob);
  } finally {
    bitmap.close?.();
  }
}

function setStage(el, src, alt) {
  el.replaceChildren();
  if (!src) {
    const p = document.createElement("div");
    p.className = "placeholder";
    p.textContent = alt || "—";
    el.appendChild(p);
    return;
  }
  const img = document.createElement("img");
  img.src = src;
  img.alt = alt || "";
  el.appendChild(img);
}

function setStatus(el, text, isError = false) {
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function hostErrorBlob(err) {
  const raw = String(err?.message || err || "");
  const code = String(err?.code || err?.name || "");
  return { raw, code, blob: `${code} ${raw}` };
}

function isHostEditUnavailable(err) {
  const { blob, raw } = hostErrorBlob(err);
  return (
    /\b502\b/i.test(blob) ||
    /Bad Gateway/i.test(blob) ||
    /APP_PROVIDER_ERROR/i.test(blob) ||
    /cloudflare/i.test(blob) ||
    /<!DOCTYPE|<html[\s>]/i.test(raw)
  );
}

function formatHostError(err) {
  const { raw, code, blob } = hostErrorBlob(err);
  if (isHostEditUnavailable(err) || /cloudflare/i.test(blob)) {
    return (
      "Platform image edit is temporarily unavailable (upload succeeded). " +
      "Please try again later."
    );
  }
  if (/APP_NOT_GRANTED|not granted|missing.?grant|permission denied|ACL/i.test(blob)) {
    return (
      "Reliclear needs image-edit permission. " +
      "Enable grants for this app in Anna, then try again."
    );
  }
  if (/APP_INVALID_REQUEST|unsupported.*(image|type|format)|mime/i.test(blob)) {
    return "That image type is not supported. Try JPG, PNG, WEBP, or HEIC.";
  }
  if (/too large|oversized|payload|size.?limit|413\b/i.test(blob)) {
    return "That photo is too large. Try a smaller file (under ~8 MB works best).";
  }
  if (/expired|presign|URL.*expir|403\b.*url/i.test(blob)) {
    return "The upload link expired. Upload the photo again, then process.";
  }
  if (/timeout|ETIMEDOUT|timed out/i.test(blob)) {
    return "Processing timed out. Try again with a smaller photo.";
  }
  // Avoid dumping Cloudflare HTML / giant blobs into the status line.
  if (/<!DOCTYPE|<html/i.test(raw) || raw.length > 280) {
    const short = raw.replace(/\s+/g, " ").replace(/<[^>]+>/g, " ").slice(0, 160).trim();
    if (!short || /cloudflare|bad gateway|502/i.test(short)) {
      return "Something went wrong talking to Anna. Please try again later.";
    }
    return `Error: ${short}${raw.length > 160 ? "…" : ""}`;
  }
  if (code && code !== "Error" && !raw.includes(code)) {
    return `Error: ${raw} (${code})`;
  }
  return `Error: ${raw}`;
}

function extractLocalPngB64(out) {
  if (!out || typeof out !== "object") return null;
  if (out.success === false) return null;
  // Harness may return plugin payload flat, nested under data, or under result.
  const layers = [out, out.data, out.result, out.data?.data, out.result?.data].filter(
    (x) => x && typeof x === "object"
  );
  for (const layer of layers) {
    const b64 = layer.image_png_b64 || layer.png_b64;
    if (typeof b64 === "string" && b64.trim()) return b64.trim();
  }
  return null;
}

async function invokeLocalTool(anna, method, args) {
  if (!anna?.tools?.invoke) {
    throw new Error("Local tools.invoke is not available in this host.");
  }
  let lastErr = null;
  for (const tool_id of localToolCandidates()) {
    try {
      const out = await anna.tools.invoke({ tool_id, method, args });
      if (out && out.success === false) {
        lastErr = new Error(out.error || `${method} failed`);
        // Undeclared/ACL rejects often arrive as success:false — try next handle.
        continue;
      }
      return out;
    } catch (e) {
      lastErr = e;
      // Try next declared/minted id on ACL / unknown-tool rejects.
      if (isToolUnavailableError(e)) continue;
      throw e;
    }
  }
  throw lastErr || new Error(`Local tool ${method} unavailable.`);
}

async function uploadImage(anna, file) {
  if (file.size <= INLINE_SOFT_LIMIT) {
    const content_b64 = await fileToBase64(file);
    // Prefer documented detailed shape; fall back to alternate arg names if needed.
    try {
      return await anna.upload.inline({
        filename: file.name || "photo.jpg",
        mime_type: file.type || "image/jpeg",
        content_b64,
        purpose: "image_input",
      });
    } catch (e) {
      const msg = String(e?.message || e);
      if (!/invalid|unknown|required|APP_INVALID/i.test(msg)) throw e;
      return await anna.upload.inline({
        filename: file.name || "photo.jpg",
        mime: file.type || "image/jpeg",
        bytes_b64: content_b64,
        purpose: "image_input",
      });
    }
  }

  const negotiated = await anna.upload.negotiate({
    filename: file.name || "photo.jpg",
    mime_type: file.type || "image/jpeg",
    purpose: "image_input",
  }).catch(async () =>
    anna.upload.negotiate({
      filename: file.name || "photo.jpg",
      mime: file.type || "image/jpeg",
      purpose: "image_input",
      size_bytes: file.size,
    })
  );

  const putUrl = negotiated.put_url || negotiated.upload_url;
  const headers = { ...(negotiated.headers || {}) };
  if (!headers["Content-Type"] && !headers["content-type"]) {
    headers["Content-Type"] = file.type || "image/jpeg";
  }
  const put = await fetch(putUrl, { method: "PUT", headers, body: file });
  if (!put.ok) {
    throw new Error(`Upload PUT failed (${put.status})`);
  }
  const key = negotiated.r2_key || negotiated.file_ref;
  return await anna.upload.confirm({ r2_key: key }).catch(() =>
    anna.upload.confirm({ file_ref: key })
  );
}

function uploadUrl(up) {
  return up.download_url || up.url;
}

function downloadResult(url, mode) {
  if (!url) return;
  if (url.startsWith("data:")) {
    const a = document.createElement("a");
    a.href = url;
    a.download = `photo-enhancer-${mode || "result"}-${Date.now()}.png`;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    return;
  }
  window.open(url, "_blank", "noopener,noreferrer");
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read result file."));
    reader.readAsDataURL(file);
  });
}

async function runCanvasMode(mode, file, opts) {
  switch (mode) {
    case "crop":
      return cropImage(file, opts.crop);
    case "adjust":
      return adjustImage(file, opts.adjust);
    case "sharpen":
      return sharpenImage(file, opts.sharpen.amount);
    case "denoise":
      return denoiseImage(file, opts.denoise.strength);
    case "restore":
      return restoreImage(file);
    case "looks":
      return applyLook(file, opts.looks.style);
    case "blur":
      return blurImage(file, opts.blur.radius);
    case "vignette":
      return vignetteImage(file, opts.vignette.strength);
    case "grain":
      return grainImage(file, opts.grain.amount);
    case "prism":
      return prismImage(file, opts.prism.strength);
    default:
      throw new Error(`Unknown canvas mode: ${mode}`);
  }
}

async function main() {
  const status = document.getElementById("status");
  const drop = document.getElementById("drop");
  const fileInput = document.getElementById("file-input");
  const runBtn = document.getElementById("run-btn");
  const downloadBtn = document.getElementById("download-btn");
  const beforeStage = document.getElementById("before-stage");
  const afterStage = document.getElementById("after-stage");
  const modeHelp = document.getElementById("mode-help");
  const modesEl = document.getElementById("modes");
  const optionsEl = document.getElementById("options");
  const catBtns = [...document.querySelectorAll(".cat")];

  let anna = null;
  let category = "clean";
  let mode = "enhance";
  let file = null;
  let localPreviewUrl = null;
  let resultUrl = null;
  let opts = structuredClone(DEFAULT_OPTS);
  /** Canvas-only modes work without Anna host; enhance/rembg need tools.invoke. */
  let hostOptional = false;

  function canProcess() {
    if (!file) return false;
    const eng = MODES[mode]?.engine;
    if (eng === "canvas" || eng === "canvas_instant") return true;
    return Boolean(anna);
  }

  function renderModes() {
    const ids = CATEGORIES[category] || [];
    modesEl.replaceChildren();
    for (const id of ids) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "mode";
      btn.dataset.mode = id;
      btn.textContent = MODES[id].label;
      btn.title = MODES[id].help;
      btn.setAttribute("aria-pressed", String(id === mode));
      btn.addEventListener("click", () => {
        if (!SHIPPED_MODES.includes(id)) return;
        mode = id;
        syncModeUi();
      });
      modesEl.appendChild(btn);
    }
  }

  function slider(rowParent, label, value, min, max, step, onInput) {
    const row = document.createElement("div");
    row.className = "opt-row";
    const lab = document.createElement("label");
    lab.className = "opt-label";
    lab.textContent = label;
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    const val = document.createElement("span");
    val.className = "opt-val";
    const fmt = (v) => (step < 1 ? Number(v).toFixed(2) : String(Math.round(Number(v))));
    val.textContent = fmt(value);
    input.addEventListener("input", () => {
      val.textContent = fmt(input.value);
      onInput(Number(input.value));
    });
    row.append(lab, input, val);
    rowParent.appendChild(row);
  }

  function renderOptions() {
    optionsEl.replaceChildren();
    const eng = MODES[mode]?.engine;
    if (eng === "canvas_instant") {
      optionsEl.hidden = false;
      const row = document.createElement("div");
      row.className = "quick-row";
      const actions = [
        ["rotate_ccw", "↺ 90°"],
        ["rotate_cw", "↻ 90°"],
        ["flip_h", "Flip H"],
        ["flip_v", "Flip V"],
      ];
      for (const [op, label] of actions) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "quick";
        b.textContent = label;
        b.disabled = !file;
        b.addEventListener("click", () => applyInstantTransform(op));
        row.appendChild(b);
      }
      optionsEl.appendChild(row);
      return;
    }
    if (mode === "crop") {
      optionsEl.hidden = false;
      for (const edge of ["left", "top", "right", "bottom"]) {
        slider(optionsEl, edge[0].toUpperCase() + edge.slice(1), opts.crop[edge], 0, 0.4, 0.01, (v) => {
          opts.crop[edge] = v;
        });
      }
      return;
    }
    if (mode === "adjust") {
      optionsEl.hidden = false;
      for (const [key, label, min, max] of [
        ["brightness", "Bright", 0.5, 1.5],
        ["contrast", "Contrast", 0.5, 1.5],
        ["saturation", "Sat", 0.5, 1.5],
        ["exposure", "Exposure", 0.5, 1.5],
        ["sharpness", "Sharp", 0.5, 1.5],
      ]) {
        slider(optionsEl, label, opts.adjust[key], min, max, 0.01, (v) => {
          opts.adjust[key] = v;
        });
      }
      return;
    }
    if (mode === "sharpen") {
      optionsEl.hidden = false;
      slider(optionsEl, "Amount", opts.sharpen.amount, 0.2, 3, 0.1, (v) => {
        opts.sharpen.amount = v;
      });
      return;
    }
    if (mode === "denoise") {
      optionsEl.hidden = false;
      slider(optionsEl, "Strength", opts.denoise.strength, 3, 21, 1, (v) => {
        opts.denoise.strength = v;
      });
      return;
    }
    if (mode === "looks") {
      optionsEl.hidden = false;
      const row = document.createElement("div");
      row.className = "chip-row";
      for (const look of LOOKS) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = look.label;
        chip.setAttribute("aria-pressed", String(look.id === opts.looks.style));
        chip.addEventListener("click", () => {
          opts.looks.style = look.id;
          renderOptions();
        });
        row.appendChild(chip);
      }
      optionsEl.appendChild(row);
      return;
    }
    if (mode === "blur") {
      optionsEl.hidden = false;
      slider(optionsEl, "Radius", opts.blur.radius, 1, 30, 1, (v) => {
        opts.blur.radius = v;
      });
      return;
    }
    if (mode === "vignette") {
      optionsEl.hidden = false;
      slider(optionsEl, "Strength", opts.vignette.strength, 0.05, 1, 0.05, (v) => {
        opts.vignette.strength = v;
      });
      return;
    }
    if (mode === "grain") {
      optionsEl.hidden = false;
      slider(optionsEl, "Amount", opts.grain.amount, 0.01, 0.12, 0.01, (v) => {
        opts.grain.amount = v;
      });
      return;
    }
    if (mode === "prism") {
      optionsEl.hidden = false;
      slider(optionsEl, "Strength", opts.prism.strength, 0.1, 1.2, 0.05, (v) => {
        opts.prism.strength = v;
      });
      return;
    }
    optionsEl.hidden = true;
  }

  function syncModeUi() {
    for (const btn of catBtns) {
      btn.setAttribute("aria-selected", String(btn.dataset.cat === category));
    }
    if (!CATEGORIES[category]?.includes(mode)) {
      mode = CATEGORIES[category][0];
    }
    renderModes();
    if (modeHelp) modeHelp.textContent = MODES[mode].help;
    renderOptions();
    const eng = MODES[mode]?.engine;
    runBtn.textContent = eng === "canvas_instant" ? "Use buttons above" : "Process photo";
    runBtn.disabled = !canProcess() || eng === "canvas_instant";
  }

  async function setWorkingFile(next, { note = "", keepAfter = false } = {}) {
    file = next;
    if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
    localPreviewUrl = URL.createObjectURL(file);
    setStage(beforeStage, localPreviewUrl, "Original");
    if (!keepAfter) {
      setStage(afterStage, null, "Result appears here");
      resultUrl = null;
      downloadBtn.disabled = true;
    }
    setStatus(
      status,
      `${file.name} · ${(file.size / 1024).toFixed(0)} KB ready.${note}`
    );
    runBtn.disabled = !canProcess() || MODES[mode]?.engine === "canvas_instant";
    renderOptions();
  }

  async function applyInstantTransform(op) {
    if (!file) return;
    runBtn.disabled = true;
    try {
      setStatus(status, "Transforming…");
      const out = await transformGeometry(file, op);
      await setWorkingFile(out, { note: ` (${op.replace("_", " ")})` });
      setStatus(status, `Applied ${op.replace("_", " ")}. Working photo updated.`);
    } catch (e) {
      setStatus(status, formatHostError(e), true);
    } finally {
      runBtn.disabled = !canProcess() || MODES[mode]?.engine === "canvas_instant";
    }
  }

  async function pickFile(next) {
    if (!isAcceptedImageFile(next)) {
      setStatus(
        status,
        "Please choose an image file (JPG, PNG, WEBP, or HEIC).",
        true
      );
      return;
    }
    runBtn.disabled = true;
    downloadBtn.disabled = true;
    try {
      let ready = next;
      if (isHeicFile(next)) {
        setStatus(
          status,
          "Converting HEIC to JPEG… (large iPhone photos can take a few seconds)"
        );
        ready = await ensureProcessableImage(next);
      }
      const note = ready !== next ? " (converted from HEIC)" : "";
      await setWorkingFile(ready, { note });
    } catch (e) {
      file = null;
      setStage(beforeStage, null, "No photo yet");
      setStage(afterStage, null, "Result appears here");
      resultUrl = null;
      setStatus(status, formatHostError(e), true);
    } finally {
      runBtn.disabled = !canProcess() || MODES[mode]?.engine === "canvas_instant";
    }
  }

  for (const btn of catBtns) {
    btn.addEventListener("click", () => {
      category = btn.dataset.cat;
      syncModeUi();
    });
  }
  syncModeUi();

  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("dragover");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const f = e.dataTransfer?.files?.[0];
    if (f) pickFile(f);
  });
  fileInput.addEventListener("change", () => {
    const f = fileInput.files?.[0];
    if (f) pickFile(f);
  });

  downloadBtn.addEventListener("click", () => {
    downloadResult(resultUrl, mode);
  });

  // Prefetch HEIC WASM while connecting so first HEIC pick is not cold-start.
  preloadHeic2Any();

  try {
    anna = await AnnaAppRuntime.connect();
    hostOptional = false;
    setStatus(status, "Ready — upload a photo to begin.");
    runBtn.disabled = !canProcess() || MODES[mode]?.engine === "canvas_instant";
    Promise.resolve()
      .then(() => anna.window.set_title({ title: "Reliclear" }))
      .then(() => anna.window.ready?.({}))
      .catch(() => {});
  } catch {
    hostOptional = true;
    setStatus(
      status,
      "Standalone preview — Edit / Looks / Effects work offline. Enhance & Remove BG need `anna-app dev`.",
      true
    );
    runBtn.disabled = !canProcess() || MODES[mode]?.engine === "canvas_instant";
  }

  runBtn.addEventListener("click", async () => {
    if (!file) return;
    if (!SHIPPED_MODES.includes(mode)) {
      setStatus(status, "That tool is not available in this release.", true);
      return;
    }

    const eng = MODES[mode]?.engine;
    if (eng === "canvas_instant") return;

    runBtn.disabled = true;
    downloadBtn.disabled = true;

    try {
      // Browser-local A-tier tools — no Executa / host required.
      if (eng === "canvas") {
        setStatus(status, MODES[mode].running);
        const outFile = await runCanvasMode(mode, file, opts);
        resultUrl = await fileToDataUrl(outFile);
        setStage(afterStage, resultUrl, "Result");
        downloadBtn.disabled = false;
        try {
          await anna?.storage?.set?.({
            key: "photo-enhancer:last",
            value: { mode, at: Date.now(), path: "canvas", engine: "browser" },
          });
        } catch {
          /* storage is best-effort */
        }
        setStatus(
          status,
          `Done — ${MODES[mode].label.toLowerCase()} in-browser. You can download the result.`
        );
        return;
      }

      if (!anna) {
        throw new Error("Anna host required for Enhance / Remove background. Run with `anna-app dev`.");
      }

      // Required local executa first (declared in manifest). Host image.edit is optional.
      if (mode === "enhance" || mode === "remove_bg") {
        const localMethod = mode === "enhance" ? "local_enhance" : "local_remove_bg";
        const localArgs =
          mode === "enhance" ? { strength: 1.0 } : { model_key: "lite" };
        try {
          setStatus(
            status,
            mode === "enhance" ? "Enhancing locally…" : "Removing background locally…"
          );
          const image_png_b64 = await fileToPngBase64(file);
          const localOut = await invokeLocalTool(anna, localMethod, {
            image_png_b64,
            ...localArgs,
          });
          const pngB64 = extractLocalPngB64(localOut);
          if (pngB64) {
            resultUrl = `data:image/png;base64,${pngB64}`;
            setStage(afterStage, resultUrl, "Result");
            downloadBtn.disabled = false;
            try {
              await anna.storage.set({
                key: "photo-enhancer:last",
                value: {
                  mode,
                  at: Date.now(),
                  path: localMethod,
                  engine:
                    localOut?.data?.engine ||
                    (mode === "enhance" ? "classical_enhance" : "rembg"),
                },
              });
            } catch {
              /* storage is best-effort */
            }
            setStatus(
              status,
              mode === "enhance"
                ? "Done — enhanced locally. You can download the result."
                : "Done — background removed locally. You can download the result."
            );
            return;
          }
        } catch (localErr) {
          setStatus(status, "Local tool unavailable — trying Anna photo-edit…");
          void localErr;
        }
      }

      const prompt = MODES[mode].prompt;
      if (!prompt) {
        throw new Error(`${MODES[mode].label} needs the local Reliclear tool.`);
      }

      setStatus(status, "Uploading photo…");
      const uploaded = await uploadImage(anna, file);
      const imageUrl = uploadUrl(uploaded);
      if (!imageUrl) throw new Error("Upload succeeded but no download URL was returned.");

      setStatus(status, MODES[mode].running);
      let edited;
      try {
        edited = await anna.image.edit({
          prompt,
          image_url: imageUrl,
          n: 1,
          model: EDIT_MODEL,
        });
      } catch (editErr) {
        if (mode === "remove_bg" && isHostEditUnavailable(editErr)) {
          throw new Error(
            "Platform image edit is temporarily unavailable, and local rembg is not available. " +
              'Install with: uv pip install "rembg[cpu]" in executas/photo-enhancer-local'
          );
        }
        if (isHostEditUnavailable(editErr)) {
          throw new Error(
            "Local Reliclear tools and Anna photo-edit are both unavailable right now. Please try again later."
          );
        }
        throw editErr;
      }

      const images = edited.images || [];
      const first = images[0];
      resultUrl = first?.url;
      if (!resultUrl) throw new Error("No edited image returned.");

      setStage(afterStage, resultUrl, "Result");
      downloadBtn.disabled = false;

      try {
        await anna.storage.set({
          key: "photo-enhancer:last",
          value: {
            mode,
            at: Date.now(),
            r2_key: first.r2_key || first.r2Key || null,
            model: edited.model || null,
          },
        });
      } catch {
        /* storage is best-effort */
      }

      setStatus(
        status,
        `Done — ${MODES[mode].label.toLowerCase()} via Anna. You can download the result.`
      );
    } catch (e) {
      setStatus(status, formatHostError(e), true);
    } finally {
      runBtn.disabled = !canProcess() || MODES[mode]?.engine === "canvas_instant";
    }
  });

  void hostOptional;
}

main();
