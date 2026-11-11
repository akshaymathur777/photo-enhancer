// Reliclear Option A — host upload + image.edit (no local CUDA).
import { AnnaAppRuntime } from "/static/anna-apps/_sdk/latest/index.js";

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
    help: "Cut out the subject; transparent when possible.",
    running: "Removing background…",
    prompt:
      "Remove the background completely. Keep the main subject with clean edges. Output a transparent PNG when possible; otherwise a clean cutout on a plain white background. Do not alter the subject.",
  },
  remove_object: {
    label: "Remove object",
    help: "Needs Anna photo-edit. Describe what to erase — e.g. power line, watermark, person on the left.",
    running: "Removing object…",
    promptFor: (text) =>
      `Remove the following from this photograph and inpaint the area so it looks natural and continuous with the surroundings: ${text}. Preserve everything else. Photorealistic, no artifacts.`,
  },
};

const INLINE_SOFT_LIMIT = 7.5 * 1024 * 1024;
// Forum #400 interim: GPT Image 2 edit route broken until ~2026-10-30 host rollout.
const EDIT_MODEL = "fal-ai/nano-banana-2";
const LOCAL_TOOL_CANDIDATES = [
  "tool-dev-photo-enhancer-local",
  "bundled:photo-enhancer-local",
  "photo-enhancer-local",
];

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
    return "That image type is not supported. Try JPG, PNG, or WEBP.";
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
  for (const tool_id of LOCAL_TOOL_CANDIDATES) {
    try {
      const out = await anna.tools.invoke({ tool_id, method, args });
      if (out && out.success === false) {
        lastErr = new Error(out.error || `${method} failed`);
        continue;
      }
      return out;
    } catch (e) {
      lastErr = e;
      const msg = String(e?.message || e?.code || e);
      // Try next tool_id spelling on ACL / unknown-tool rejects.
      if (/DENIED|NOT_FOUND|unknown|invalid.*tool|APP_NOT_GRANTED|ACL/i.test(msg)) {
        continue;
      }
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
  const objectPrompt = document.getElementById("object-prompt");
  const objectText = document.getElementById("object-text");
  const modeHelp = document.getElementById("mode-help");
  const modeBtns = [...document.querySelectorAll(".mode")];

  let anna = null;
  let mode = "enhance";
  let file = null;
  let localPreviewUrl = null;
  let resultUrl = null;

  function syncModeUi() {
    for (const btn of modeBtns) {
      btn.setAttribute("aria-pressed", String(btn.dataset.mode === mode));
    }
    objectPrompt.classList.toggle("visible", mode === "remove_object");
    if (modeHelp) modeHelp.textContent = MODES[mode].help;
  }

  function pickFile(next) {
    if (!next || !next.type.startsWith("image/")) {
      setStatus(status, "Please choose an image file (JPG, PNG, or WEBP).", true);
      return;
    }
    file = next;
    if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
    localPreviewUrl = URL.createObjectURL(file);
    setStage(beforeStage, localPreviewUrl, "Original");
    setStage(afterStage, null, "Result appears here");
    resultUrl = null;
    downloadBtn.disabled = true;
    runBtn.disabled = !anna;
    setStatus(status, `${file.name} · ${(file.size / 1024).toFixed(0)} KB ready.`);
  }

  for (const btn of modeBtns) {
    btn.addEventListener("click", () => {
      mode = btn.dataset.mode;
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

    let prompt;
    if (mode === "remove_object") {
      const text = (objectText.value || "").trim();
      if (!text) {
        setStatus(
          status,
          "Describe what to remove — for example, the power line or watermark.",
          true
        );
        objectText.focus();
        return;
      }
      prompt = MODES.remove_object.promptFor(text);
    } else {
      prompt = MODES[mode].prompt;
    }

    runBtn.disabled = true;
    downloadBtn.disabled = true;
    setStatus(status, "Uploading photo…");

    try {
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
        if (!isHostEditUnavailable(editErr)) throw editErr;

        // Host edit 502 / provider outage — graceful local classical fallback.
        if (mode === "enhance") {
          setStatus(status, "Host edit unavailable — trying local classical enhance…");
          const image_png_b64 = await fileToPngBase64(file);
          const localOut = await invokeLocalTool(anna, "local_enhance", {
            image_png_b64,
            strength: 1.0,
          });
          const pngB64 = extractLocalPngB64(localOut);
          if (!pngB64) throw new Error("Local enhance returned no image.");
          resultUrl = `data:image/png;base64,${pngB64}`;
          setStage(afterStage, resultUrl, "Result");
          downloadBtn.disabled = false;
          try {
            await anna.storage.set({
              key: "photo-enhancer:last",
              value: {
                mode,
                at: Date.now(),
                path: "local_enhance",
                engine: localOut?.data?.engine || "classical_enhance",
              },
            });
          } catch {
            /* storage is best-effort */
          }
          setStatus(
            status,
            "Done — enhanced locally (Anna photo-edit unavailable). You can download the result."
          );
          return;
        }

        if (mode === "remove_bg") {
          // Optional rembg path — succeed if installed; else clear platform + rembg hint.
          let localErr = null;
          try {
            setStatus(status, "Host edit unavailable — trying local remove background…");
            const image_png_b64 = await fileToPngBase64(file);
            const localOut = await invokeLocalTool(anna, "local_remove_bg", {
              image_png_b64,
              model_key: "lite",
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
                    path: "local_remove_bg",
                    engine: localOut?.data?.engine || "rembg",
                  },
                });
              } catch {
                /* storage is best-effort */
              }
              setStatus(
                status,
                "Done — background removed locally (Anna photo-edit unavailable). You can download the result."
              );
              return;
            }
            localErr = new Error(
              localOut?.error || "Local remove_bg returned no image."
            );
          } catch (e) {
            localErr = e;
          }
          const detail = String(localErr?.message || localErr || "");
          if (/rembg|No module named|not installed|ImportError|onnxruntime/i.test(detail)) {
            throw new Error(
              "Platform image edit is temporarily unavailable, and local rembg is not available. " +
                'Install with: uv pip install "rembg[cpu]" in executas/photo-enhancer-local'
            );
          }
          throw editErr;
        }

        // remove_object: host edit down and UI has no mask → clear message only (no fake inpaint).
        throw new Error(
          "Remove object needs Anna's photo-edit service, which is temporarily unavailable. " +
            "Try Enhance or Remove background instead, or try Remove object again later."
        );
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
