// Reliclear Option A — host upload + image.edit (no local CUDA).
import { AnnaAppRuntime } from "/static/anna-apps/_sdk/latest/index.js";

// v0.1 App Review: UI ships Enhance + Remove background only.
// remove_object kept for a later unhide when host image.edit recovers.
const MODES = {
  enhance: {
    label: "Enhance",
    help: "Improve clarity, color, and detail.",
    running: "Enhancing…",
    prompt:
      "Enhance this photograph: improve clarity, natural detail, color balance, and mild sharpening. Keep the subject, composition, and identity unchanged. Do not add or remove objects. Photorealistic output.",
  },
  remove_bg: {
    label: "Remove background",
    help: "AI cutout of the subject; transparent when possible.",
    running: "Removing background…",
    prompt:
      "Remove the background completely. Keep the main subject with clean edges. Output a transparent PNG when possible; otherwise a clean cutout on a plain white background. Do not alter the subject.",
  },
  remove_object: {
    label: "Remove object",
    help: "Not available in this release — needs Anna photo-edit.",
    running: "Removing object…",
    promptFor: (text) =>
      `Remove the following from this photograph and inpaint the area so it looks natural and continuous with the surroundings: ${text}. Preserve everything else. Photorealistic, no artifacts.`,
  },
};

const SHIPPED_MODES = ["enhance", "remove_bg"];

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

async function main() {
  const status = document.getElementById("status");
  const drop = document.getElementById("drop");
  const fileInput = document.getElementById("file-input");
  const runBtn = document.getElementById("run-btn");
  const downloadBtn = document.getElementById("download-btn");
  const beforeStage = document.getElementById("before-stage");
  const afterStage = document.getElementById("after-stage");
  const modeHelp = document.getElementById("mode-help");
  const modeBtns = [...document.querySelectorAll(".mode")].filter((btn) =>
    SHIPPED_MODES.includes(btn.dataset.mode)
  );

  let anna = null;
  let mode = "enhance";
  let file = null;
  let localPreviewUrl = null;
  let resultUrl = null;

  function syncModeUi() {
    for (const btn of modeBtns) {
      btn.setAttribute("aria-pressed", String(btn.dataset.mode === mode));
    }
    if (modeHelp) modeHelp.textContent = MODES[mode].help;
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
      file = ready;
      if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
      localPreviewUrl = URL.createObjectURL(file);
      setStage(beforeStage, localPreviewUrl, "Original");
      setStage(afterStage, null, "Result appears here");
      resultUrl = null;
      const note =
        ready !== next ? " (converted from HEIC)" : "";
      setStatus(
        status,
        `${file.name} · ${(file.size / 1024).toFixed(0)} KB ready.${note}`
      );
    } catch (e) {
      file = null;
      setStage(beforeStage, null, "No photo yet");
      setStage(afterStage, null, "Result appears here");
      resultUrl = null;
      setStatus(status, formatHostError(e), true);
    } finally {
      runBtn.disabled = !file || !anna;
    }
  }

  for (const btn of modeBtns) {
    btn.addEventListener("click", () => {
      const next = btn.dataset.mode;
      if (!SHIPPED_MODES.includes(next)) return;
      mode = next;
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
    setStatus(status, "Ready — upload a photo to begin.");
    runBtn.disabled = !file;
    // Title / ready are best-effort — never block the editor on them.
    Promise.resolve()
      .then(() => anna.window.set_title({ title: "Reliclear" }))
      .then(() => anna.window.ready?.({}))
      .catch(() => {});
  } catch {
    setStatus(
      status,
      "Standalone preview (no Anna host). Run with `anna-app dev` for real edits.",
      true
    );
    runBtn.disabled = true;
  }

  runBtn.addEventListener("click", async () => {
    if (!anna || !file) return;

    if (!SHIPPED_MODES.includes(mode)) {
      setStatus(
        status,
        "Remove object is not available in this release. Use Enhance or Remove background.",
        true
      );
      return;
    }

    const prompt = MODES[mode].prompt;

    runBtn.disabled = true;
    downloadBtn.disabled = true;

    try {
      // Required local executa first (declared in manifest). Host image.edit is optional.
      if (mode === "enhance" || mode === "remove_bg") {
        const localMethod = mode === "enhance" ? "local_enhance" : "local_remove_bg";
        const localArgs =
          mode === "enhance"
            ? { strength: 1.0 }
            : { model_key: "lite" };
        try {
          setStatus(
            status,
            mode === "enhance"
              ? "Enhancing locally…"
              : "Removing background locally…"
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
          const detail = String(localErr?.message || localErr || "");
          if (
            mode === "remove_bg" &&
            /rembg|No module named|not installed|ImportError|onnxruntime/i.test(detail)
          ) {
            // Fall through to optional host edit; if that also fails, surface rembg hint.
          } else if (!isToolUnavailableError(localErr) && !isHostEditUnavailable(localErr)) {
            // Real local failure (not "tool missing") — still try optional host before giving up.
          }
          // Continue to optional host image.edit below.
          setStatus(status, "Local tool unavailable — trying Anna photo-edit…");
        }
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
      runBtn.disabled = !file || !anna;
    }
  });
}

main();
