/**
 * Browser-local image ops for Reliclear (no CUDA / no Executa round-trip).
 * Mirrors desktop A-tier: crop, rotate, flip, adjust, sharpen, denoise-lite,
 * blur, vignette, grain, prism, and a Looks subset.
 */

export const LOOKS = [
  { id: "cinematic", label: "Cinematic" },
  { id: "vintage", label: "Vintage" },
  { id: "vivid", label: "Vivid" },
  { id: "bw_dramatic", label: "B&W" },
  { id: "golden_hour", label: "Golden hour" },
  { id: "cool_moody", label: "Cool moody" },
  { id: "sepia", label: "Sepia" },
  { id: "faded", label: "Faded" },
  { id: "punchy", label: "Punchy" },
  { id: "soft_glow", label: "Soft glow" },
  { id: "matte", label: "Matte" },
  { id: "warm_film", label: "Warm film" },
];

function loadBitmap(file) {
  return createImageBitmap(file);
}

async function canvasToFile(canvas, nameHint = "photo.png", type = "image/png", quality) {
  const blob = await new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("Canvas encode failed."))),
      type,
      quality
    );
  });
  const base = String(nameHint || "photo").replace(/\.[^.]+$/, "") || "photo";
  const ext = type === "image/jpeg" ? "jpg" : "png";
  return new File([blob], `${base}.${ext}`, { type, lastModified: Date.now() });
}

async function withImageData(file, mutate) {
  const bitmap = await loadBitmap(file);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Canvas unavailable.");
    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    mutate(imageData, canvas, ctx);
    ctx.putImageData(imageData, 0, 0);
    const base = String(file.name || "photo").replace(/\.[^.]+$/, "") || "photo";
    return canvasToFile(canvas, base, "image/png");
  } finally {
    bitmap.close?.();
  }
}

function clampByte(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

function luma(r, g, b) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/** Rotate 90° clockwise / counterclockwise, or flip. */
export async function transformGeometry(file, op) {
  const bitmap = await loadBitmap(file);
  try {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas unavailable.");
    const w = bitmap.width;
    const h = bitmap.height;
    if (op === "rotate_cw" || op === "rotate_ccw") {
      canvas.width = h;
      canvas.height = w;
      if (op === "rotate_cw") {
        ctx.translate(h, 0);
        ctx.rotate(Math.PI / 2);
      } else {
        ctx.translate(0, w);
        ctx.rotate(-Math.PI / 2);
      }
      ctx.drawImage(bitmap, 0, 0);
    } else if (op === "flip_h") {
      canvas.width = w;
      canvas.height = h;
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(bitmap, 0, 0);
    } else if (op === "flip_v") {
      canvas.width = w;
      canvas.height = h;
      ctx.translate(0, h);
      ctx.scale(1, -1);
      ctx.drawImage(bitmap, 0, 0);
    } else {
      throw new Error(`Unknown transform: ${op}`);
    }
    const base = String(file.name || "photo").replace(/\.[^.]+$/, "") || "photo";
    return canvasToFile(canvas, base, "image/png");
  } finally {
    bitmap.close?.();
  }
}

/**
 * Crop using normalized insets 0–0.45 from each edge (left/top/right/bottom).
 */
export async function cropImage(file, insets = {}) {
  const left = Math.max(0, Math.min(0.45, Number(insets.left) || 0));
  const top = Math.max(0, Math.min(0.45, Number(insets.top) || 0));
  const right = Math.max(0, Math.min(0.45, Number(insets.right) || 0));
  const bottom = Math.max(0, Math.min(0.45, Number(insets.bottom) || 0));
  const bitmap = await loadBitmap(file);
  try {
    const w = bitmap.width;
    const h = bitmap.height;
    const x0 = Math.floor(w * left);
    const y0 = Math.floor(h * top);
    const x1 = Math.ceil(w * (1 - right));
    const y1 = Math.ceil(h * (1 - bottom));
    const cw = Math.max(1, x1 - x0);
    const ch = Math.max(1, y1 - y0);
    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas unavailable.");
    ctx.drawImage(bitmap, x0, y0, cw, ch, 0, 0, cw, ch);
    const base = String(file.name || "photo").replace(/\.[^.]+$/, "") || "photo";
    return canvasToFile(canvas, base, "image/png");
  } finally {
    bitmap.close?.();
  }
}

/**
 * Adjust: brightness/contrast/saturation/exposure/sharpness multipliers (~0.5–1.5).
 */
export async function adjustImage(file, opts = {}) {
  const brightness = Number(opts.brightness ?? 1);
  const contrast = Number(opts.contrast ?? 1);
  const saturation = Number(opts.saturation ?? 1);
  const exposure = Number(opts.exposure ?? 1);
  const sharpness = Number(opts.sharpness ?? 1);

  let out = await withImageData(file, (imageData) => {
    const d = imageData.data;
    const gamma = Math.max(0.25, Math.min(4, 1 / Math.max(0.25, exposure)));
    for (let i = 0; i < d.length; i += 4) {
      let r = d[i] / 255;
      let g = d[i + 1] / 255;
      let b = d[i + 2] / 255;
      // Exposure as gamma
      r = Math.pow(r, gamma);
      g = Math.pow(g, gamma);
      b = Math.pow(b, gamma);
      // Brightness
      r *= brightness;
      g *= brightness;
      b *= brightness;
      // Contrast around mid-gray
      r = (r - 0.5) * contrast + 0.5;
      g = (g - 0.5) * contrast + 0.5;
      b = (b - 0.5) * contrast + 0.5;
      // Saturation
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      r = y + (r - y) * saturation;
      g = y + (g - y) * saturation;
      b = y + (b - y) * saturation;
      d[i] = clampByte(r * 255);
      d[i + 1] = clampByte(g * 255);
      d[i + 2] = clampByte(b * 255);
    }
  });

  if (Math.abs(sharpness - 1) > 0.02) {
    out = await sharpenImage(out, Math.max(0, sharpness - 1) * 2.2);
  }
  return out;
}

export async function sharpenImage(file, amount = 1.5) {
  const a = Math.max(0, Math.min(4, Number(amount) || 0));
  if (a < 0.05) {
    // Pass-through copy
    return withImageData(file, () => {});
  }
  return withImageData(file, (imageData) => {
    const { data, width, height } = imageData;
    const src = new Uint8ClampedArray(data);
    const blur = boxBlur(src, width, height, 1);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = clampByte(src[i] + a * (src[i] - blur[i]));
      data[i + 1] = clampByte(src[i + 1] + a * (src[i + 1] - blur[i + 1]));
      data[i + 2] = clampByte(src[i + 2] + a * (src[i + 2] - blur[i + 2]));
    }
  });
}

/** Light denoise via small bilateral-ish blur (not OpenCV NLMeans). */
export async function denoiseImage(file, strength = 10) {
  const s = Math.max(1, Math.min(21, Number(strength) || 10));
  const radius = s <= 8 ? 1 : s <= 14 ? 2 : 3;
  return withImageData(file, (imageData) => {
    const { data, width, height } = imageData;
    const src = new Uint8ClampedArray(data);
    const blurred = boxBlur(src, width, height, radius);
    const mix = 0.35 + Math.min(s, 21) / 40;
    for (let i = 0; i < data.length; i += 4) {
      data[i] = clampByte(src[i] * (1 - mix) + blurred[i] * mix);
      data[i + 1] = clampByte(src[i + 1] * (1 - mix) + blurred[i + 1] * mix);
      data[i + 2] = clampByte(src[i + 2] * (1 - mix) + blurred[i + 2] * mix);
    }
  });
}

export async function restoreImage(file) {
  const denoised = await denoiseImage(file, 10);
  return sharpenImage(denoised, 1.4);
}

export async function blurImage(file, radius = 8) {
  const r = Math.max(1, Math.min(40, Math.round(Number(radius) || 8)));
  return withImageData(file, (imageData) => {
    const { data, width, height } = imageData;
    const src = new Uint8ClampedArray(data);
    const blurred = boxBlur(src, width, height, r);
    data.set(blurred);
  });
}

export async function vignetteImage(file, strength = 0.5) {
  const s = Math.max(0, Math.min(1, Number(strength) || 0));
  return withImageData(file, (imageData) => {
    const { data, width, height } = imageData;
    const cx = width / 2;
    const cy = height / 2;
    const maxD = Math.sqrt(cx * cx + cy * cy) || 1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        const v = 1 - Math.pow(dist / maxD, 1.8) * s;
        data[i] = clampByte(data[i] * v);
        data[i + 1] = clampByte(data[i + 1] * v);
        data[i + 2] = clampByte(data[i + 2] * v);
      }
    }
  });
}

export async function grainImage(file, amount = 0.04) {
  const a = Math.max(0, Math.min(0.2, Number(amount) || 0));
  return withImageData(file, (imageData) => {
    const { data } = imageData;
    // Deterministic-ish LCG so results are stable across runs
    let seed = 42;
    const next = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xffffffff;
    };
    for (let i = 0; i < data.length; i += 4) {
      const n = (next() * 2 - 1) * a * 255;
      data[i] = clampByte(data[i] + n);
      data[i + 1] = clampByte(data[i + 1] + n);
      data[i + 2] = clampByte(data[i + 2] + n);
    }
  });
}

export async function prismImage(file, strength = 0.6) {
  const s = Math.max(0.1, Math.min(1.5, Number(strength) || 0.6));
  return withImageData(file, (imageData) => {
    const { data, width, height } = imageData;
    const src = new Uint8ClampedArray(data);
    const shift = Math.max(2, Math.floor(width * 0.008 * s));
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const xr = Math.min(width - 1, x + shift);
        const xb = Math.max(0, x - shift);
        const ir = (y * width + xr) * 4;
        const ib = (y * width + xb) * 4;
        const r = src[ir];
        const g = src[i + 1];
        const b = src[ib + 2];
        data[i] = clampByte(src[i] * (1 - s * 0.5) + r * (s * 0.5));
        data[i + 1] = clampByte(src[i + 1] * (1 - s * 0.5) + g * (s * 0.5));
        data[i + 2] = clampByte(src[i + 2] * (1 - s * 0.5) + b * (s * 0.5));
      }
    }
  });
}

export async function applyLook(file, styleId) {
  const id = String(styleId || "cinematic");
  switch (id) {
    case "vintage":
      return withImageData(file, (imageData) => {
        const d = imageData.data;
        for (let i = 0; i < d.length; i += 4) {
          let r = d[i] * 1.06 + 12;
          let g = d[i + 1];
          let b = d[i + 2] * 0.88;
          r = r * 0.92 + 18;
          g = g * 0.92 + 18;
          b = b * 0.92 + 18;
          d[i] = clampByte(r);
          d[i + 1] = clampByte(g);
          d[i + 2] = clampByte(b);
        }
      }).then((f) => grainImage(f, 0.035));
    case "vivid":
      return adjustImage(file, { saturation: 1.45, contrast: 1.15, sharpness: 1.2 });
    case "bw_dramatic":
      return withImageData(file, (imageData) => {
        const d = imageData.data;
        for (let i = 0; i < d.length; i += 4) {
          let y = luma(d[i], d[i + 1], d[i + 2]);
          y = clampByte((y - 128) * 1.35 + 128);
          d[i] = d[i + 1] = d[i + 2] = y;
        }
      });
    case "golden_hour":
      return withImageData(file, (imageData) => {
        const d = imageData.data;
        for (let i = 0; i < d.length; i += 4) {
          d[i] = clampByte(d[i] * 1.12 + 10);
          d[i + 1] = clampByte(d[i + 1] * 1.06 + 6);
          d[i + 2] = clampByte(d[i + 2] * 0.92);
        }
      }).then((f) => adjustImage(f, { brightness: 1.08, sharpness: 1.1 }));
    case "cool_moody":
      return withImageData(file, (imageData) => {
        const d = imageData.data;
        for (let i = 0; i < d.length; i += 4) {
          d[i] = clampByte(d[i] * 0.9);
          d[i + 2] = clampByte(d[i + 2] * 1.14 + 8);
          d[i] = clampByte(d[i] * 0.94);
          d[i + 1] = clampByte(d[i + 1] * 0.94);
          d[i + 2] = clampByte(d[i + 2] * 0.94);
        }
      }).then((f) => adjustImage(f, { contrast: 1.12 }));
    case "sepia":
      return withImageData(file, (imageData) => {
        const d = imageData.data;
        for (let i = 0; i < d.length; i += 4) {
          const r = d[i];
          const g = d[i + 1];
          const b = d[i + 2];
          d[i] = clampByte(r * 0.393 + g * 0.769 + b * 0.189);
          d[i + 1] = clampByte(r * 0.349 + g * 0.686 + b * 0.168);
          d[i + 2] = clampByte(r * 0.272 + g * 0.534 + b * 0.131);
        }
      });
    case "faded":
      return adjustImage(file, { saturation: 0.85, contrast: 0.82 });
    case "punchy":
      return adjustImage(file, { contrast: 1.25, saturation: 1.3, sharpness: 1.3 });
    case "soft_glow":
      return blurImage(file, 2).then((f) => adjustImage(f, { brightness: 1.06 }));
    case "matte":
      return adjustImage(file, { brightness: 1.06, contrast: 0.78, saturation: 0.92 });
    case "warm_film":
      return withImageData(file, (imageData) => {
        const d = imageData.data;
        for (let i = 0; i < d.length; i += 4) {
          d[i] = clampByte(d[i] * 1.08);
          d[i + 1] = clampByte(d[i + 1] * 1.02);
          d[i + 2] = clampByte(d[i + 2] * 0.92);
        }
      }).then((f) => grainImage(f, 0.02));
    case "cinematic":
    default:
      return withImageData(file, (imageData) => {
        const d = imageData.data;
        for (let i = 0; i < d.length; i += 4) {
          let r = d[i] / 255;
          let g = d[i + 1] / 255;
          let b = d[i + 2] / 255;
          const lum = 0.299 * r + 0.587 * g + 0.114 * b;
          const shadow = Math.max(0, Math.min(1, 1 - lum * 1.4));
          const highlight = Math.max(0, Math.min(1, (lum - 0.45) * 2.2));
          r = Math.max(0, Math.min(1, r + highlight * 0.12 - shadow * 0.04));
          g = Math.max(0, Math.min(1, g + highlight * 0.06 - shadow * 0.1));
          b = Math.max(0, Math.min(1, b + highlight * 0.02 + shadow * 0.08));
          d[i] = clampByte(r * 255);
          d[i + 1] = clampByte(g * 255);
          d[i + 2] = clampByte(b * 255);
        }
      }).then((f) => adjustImage(f, { contrast: 1.22, saturation: 1.08 }));
  }
}

function boxBlur(src, width, height, radius) {
  if (radius < 1) return new Uint8ClampedArray(src);
  const out = new Uint8ClampedArray(src.length);
  const tmp = new Uint8ClampedArray(src.length);
  const w = width;
  const h = height;
  const r = radius | 0;

  // Horizontal
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let rs = 0;
      let gs = 0;
      let bs = 0;
      let as = 0;
      let count = 0;
      for (let k = -r; k <= r; k++) {
        const xx = Math.min(w - 1, Math.max(0, x + k));
        const i = (y * w + xx) * 4;
        rs += src[i];
        gs += src[i + 1];
        bs += src[i + 2];
        as += src[i + 3];
        count++;
      }
      const o = (y * w + x) * 4;
      tmp[o] = rs / count;
      tmp[o + 1] = gs / count;
      tmp[o + 2] = bs / count;
      tmp[o + 3] = as / count;
    }
  }
  // Vertical
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let rs = 0;
      let gs = 0;
      let bs = 0;
      let as = 0;
      let count = 0;
      for (let k = -r; k <= r; k++) {
        const yy = Math.min(h - 1, Math.max(0, y + k));
        const i = (yy * w + x) * 4;
        rs += tmp[i];
        gs += tmp[i + 1];
        bs += tmp[i + 2];
        as += tmp[i + 3];
        count++;
      }
      const o = (y * w + x) * 4;
      out[o] = rs / count;
      out[o + 1] = gs / count;
      out[o + 2] = bs / count;
      out[o + 3] = as / count;
    }
  }
  return out;
}
