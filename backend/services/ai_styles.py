"""Creative look presets applied via image processing pipelines."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

AI_STYLES: dict[str, dict[str, Any]] = {
    "cinematic": {
        "id": "cinematic",
        "name": "Cinematic",
        "description": "High contrast with teal shadows and warm highlights.",
        "tags": ["film", "movie", "teal", "orange"],
    },
    "vintage": {
        "id": "vintage",
        "name": "Vintage",
        "description": "Faded tones, warm cast, and subtle film grain.",
        "tags": ["retro", "old", "film", "faded"],
    },
    "vivid": {
        "id": "vivid",
        "name": "Vivid",
        "description": "Boosted saturation and clarity.",
        "tags": ["saturated", "colorful", "pop", "vibrant"],
    },
    "portrait": {
        "id": "portrait",
        "name": "Portrait",
        "description": "Skin softening with gentle background blur.",
        "tags": ["face", "skin", "soft", "bokeh"],
    },
    "bw_dramatic": {
        "id": "bw_dramatic",
        "name": "B&W Dramatic",
        "description": "Monochrome with punchy contrast.",
        "tags": ["black and white", "monochrome", "noir", "dramatic"],
    },
    "golden_hour": {
        "id": "golden_hour",
        "name": "Golden Hour",
        "description": "Warm sunset glow and soft highlights.",
        "tags": ["warm", "sunset", "golden", "glow"],
    },
    "cool_moody": {
        "id": "cool_moody",
        "name": "Cool Moody",
        "description": "Blue-toned shadows with subdued warmth.",
        "tags": ["cool", "blue", "moody", "cold"],
    },
    "faded": {"id": "faded", "name": "Faded", "description": "Soft, washed-out film look.", "tags": ["fade", "soft"]},
    "noir": {"id": "noir", "name": "Noir", "description": "Dark cinematic noir tones.", "tags": ["noir", "dark"]},
    "sepia": {"id": "sepia", "name": "Sepia", "description": "Classic warm brown tone.", "tags": ["sepia", "brown"]},
    "pastel": {"id": "pastel", "name": "Pastel", "description": "Soft desaturated pastels.", "tags": ["pastel", "soft"]},
    "arctic": {"id": "arctic", "name": "Arctic", "description": "Crisp cool whites and blues.", "tags": ["cold", "winter"]},
    "sunset": {"id": "sunset", "name": "Sunset", "description": "Rich orange and pink sky tones.", "tags": ["sunset", "warm"]},
    "emerald": {"id": "emerald", "name": "Emerald", "description": "Deep green cinematic grade.", "tags": ["green", "forest"]},
    "rose_gold": {"id": "rose_gold", "name": "Rose Gold", "description": "Pink-gold luxury tones.", "tags": ["rose", "gold"]},
    "matte": {"id": "matte", "name": "Matte", "description": "Lifted blacks, muted contrast.", "tags": ["matte", "flat"]},
    "high_key": {"id": "high_key", "name": "High Key", "description": "Bright airy high-key look.", "tags": ["bright", "airy"]},
    "low_key": {"id": "low_key", "name": "Low Key", "description": "Moody dark low-key lighting.", "tags": ["dark", "moody"]},
    "cross_process": {"id": "cross_process", "name": "Cross Process", "description": "Cross-processed color shift.", "tags": ["cross", "film"]},
    "bleach_bypass": {"id": "bleach_bypass", "name": "Bleach Bypass", "description": "Desaturated high-contrast film.", "tags": ["bleach", "gritty"]},
    "warm_film": {"id": "warm_film", "name": "Warm Film", "description": "Kodak-style warm film stock.", "tags": ["film", "warm"]},
    "cool_film": {"id": "cool_film", "name": "Cool Film", "description": "Fuji-style cool film stock.", "tags": ["film", "cool"]},
    "neon_pop": {"id": "neon_pop", "name": "Neon Pop", "description": "Vibrant neon saturation boost.", "tags": ["neon", "vivid"]},
    "dusty": {"id": "dusty", "name": "Dusty", "description": "Muted dusty desert tones.", "tags": ["dust", "desert"]},
    "editorial": {"id": "editorial", "name": "Editorial", "description": "Clean magazine editorial grade.", "tags": ["magazine", "clean"]},
    "soft_glow": {"id": "soft_glow", "name": "Soft Glow", "description": "Dreamy soft glow highlights.", "tags": ["glow", "dreamy"]},
    "punchy": {"id": "punchy", "name": "Punchy", "description": "Bold contrast and color pop.", "tags": ["bold", "pop"]},
    "earthy": {"id": "earthy", "name": "Earthy", "description": "Natural earth-tone palette.", "tags": ["earth", "natural"]},
    "aqua": {"id": "aqua", "name": "Aqua", "description": "Teal aqua coastal tones.", "tags": ["aqua", "teal"]},
    "lavender": {"id": "lavender", "name": "Lavender", "description": "Soft purple lavender cast.", "tags": ["purple", "lavender"]},
    "retro_pop": {"id": "retro_pop", "name": "Retro Pop", "description": "80s retro color pop.", "tags": ["retro", "80s"]},
    "ink": {"id": "ink", "name": "Ink", "description": "Deep ink-blue shadows.", "tags": ["ink", "blue"]},
    "honey": {"id": "honey", "name": "Honey", "description": "Warm honey amber tones.", "tags": ["honey", "amber"]},
    "slate": {"id": "slate", "name": "Slate", "description": "Desaturated slate gray grade.", "tags": ["gray", "slate"]},
    "velvia": {"id": "velvia", "name": "Velvia", "description": "Ultra-vivid landscape colors.", "tags": ["velvia", "landscape"]},
    "twilight": {"id": "twilight", "name": "Twilight", "description": "Purple-blue twilight hour.", "tags": ["twilight", "dusk"]},
}

DEFAULT_STYLE = "cinematic"


def list_styles() -> list[dict[str, Any]]:
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "description": s["description"],
            "tags": s.get("tags", []),
        }
        for s in AI_STYLES.values()
    ]


def normalize_style(style: str | None) -> str:
    if not style:
        return DEFAULT_STYLE
    key = style.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "cinematic": "cinematic",
        "vintage": "vintage",
        "vivid": "vivid",
        "portrait": "portrait",
        "bw": "bw_dramatic",
        "bw_dramatic": "bw_dramatic",
        "black_and_white": "bw_dramatic",
        "black_white": "bw_dramatic",
        "monochrome": "bw_dramatic",
        "golden_hour": "golden_hour",
        "golden": "golden_hour",
        "cool_moody": "cool_moody",
        "cool": "cool_moody",
        "moody": "cool_moody",
    }
    if key in aliases:
        return aliases[key]
    if key in AI_STYLES:
        return key
    for sid, meta in AI_STYLES.items():
        if key in sid or any(key in t.replace(" ", "_") for t in meta.get("tags", [])):
            return sid
    return DEFAULT_STYLE


def _preserve_alpha(image: Image.Image, result_rgb: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        out = result_rgb.convert("RGBA")
        out.putalpha(image.split()[3])
        return out
    return result_rgb


def _add_grain(arr: np.ndarray, amount: float = 0.04, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amount * 255, arr.shape).astype(np.float32)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_cinematic(rgb: Image.Image) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    shadow = np.clip(1.0 - lum * 1.4, 0, 1)
    highlight = np.clip((lum - 0.45) * 2.2, 0, 1)
    r = np.clip(r + highlight * 0.12 - shadow * 0.04, 0, 1)
    g = np.clip(g + highlight * 0.06 - shadow * 0.10, 0, 1)
    b = np.clip(b + highlight * 0.02 + shadow * 0.08, 0, 1)
    out = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(out)
    img = ImageEnhance.Contrast(img).enhance(1.22)
    img = ImageEnhance.Color(img).enhance(1.08)
    return img


def _apply_vintage(rgb: Image.Image) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.float32)
    arr[..., 0] = arr[..., 0] * 1.06 + 12
    arr[..., 2] = arr[..., 2] * 0.88
    arr = np.clip(arr * 0.92 + 18, 0, 255).astype(np.uint8)
    arr = _add_grain(arr, 0.035)
    img = Image.fromarray(arr)
    return ImageEnhance.Contrast(img).enhance(0.88)


def _apply_vivid(rgb: Image.Image) -> Image.Image:
    img = ImageEnhance.Color(rgb).enhance(1.45)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.35)
    return img


def _apply_portrait(rgb: Image.Image) -> Image.Image:
    arr = np.array(rgb.convert("RGB"))
    smooth = cv2.bilateralFilter(arr, d=9, sigmaColor=40, sigmaSpace=40)
    h, w = arr.shape[:2]
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_d = np.sqrt(cx**2 + cy**2)
    center_w = np.clip(1.0 - (dist / max_d) ** 1.6, 0, 1)
    blend = (center_w[:, :, None] * 0.65 + 0.35)
    blended = (smooth.astype(np.float32) * blend + arr.astype(np.float32) * (1 - blend)).astype(np.uint8)
    img = Image.fromarray(blended)
    return ImageEnhance.Brightness(img).enhance(1.04)


def _apply_bw_dramatic(rgb: Image.Image) -> Image.Image:
    gray = rgb.convert("L")
    arr = np.array(gray, dtype=np.float32)
    arr = np.clip((arr - 128) * 1.35 + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _apply_golden_hour(rgb: Image.Image) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 1.12 + 10, 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] * 1.06 + 6, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * 0.92, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return ImageEnhance.Sharpness(img).enhance(1.15)


def _apply_cool_moody(rgb: Image.Image) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.float32)
    arr[..., 2] = np.clip(arr[..., 2] * 1.14 + 8, 0, 255)
    arr[..., 0] = np.clip(arr[..., 0] * 0.90, 0, 255)
    arr = np.clip(arr * 0.94, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return ImageEnhance.Contrast(img).enhance(1.12)


def _tone_shift(rgb: Image.Image, r: float, g: float, b: float, contrast: float = 1.0) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * r, 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] * g, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * b, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return ImageEnhance.Contrast(img).enhance(contrast)


def _apply_faded(rgb): return ImageEnhance.Contrast(ImageEnhance.Color(rgb).enhance(0.85)).enhance(0.82)
def _apply_noir(rgb): return _tone_shift(_apply_bw_dramatic(rgb), 1.0, 1.0, 1.0, 1.3)
def _apply_sepia(rgb):
    arr = np.array(rgb, dtype=np.float32)
    r = arr[..., 0] * 0.393 + arr[..., 1] * 0.769 + arr[..., 2] * 0.189
    g = arr[..., 0] * 0.349 + arr[..., 1] * 0.686 + arr[..., 2] * 0.168
    b = arr[..., 0] * 0.272 + arr[..., 1] * 0.534 + arr[..., 2] * 0.131
    return Image.fromarray(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
def _apply_pastel(rgb): return ImageEnhance.Color(ImageEnhance.Brightness(rgb).enhance(1.08)).enhance(0.65)
def _apply_arctic(rgb): return _tone_shift(rgb, 0.92, 0.96, 1.12, 1.05)
def _apply_sunset(rgb): return _tone_shift(rgb, 1.18, 1.02, 0.88, 1.08)
def _apply_emerald(rgb): return _tone_shift(rgb, 0.88, 1.12, 0.95, 1.1)
def _apply_rose_gold(rgb): return _tone_shift(rgb, 1.14, 0.96, 0.92, 1.05)
def _apply_matte(rgb): return ImageEnhance.Contrast(ImageEnhance.Brightness(rgb).enhance(1.06)).enhance(0.78)
def _apply_high_key(rgb): return ImageEnhance.Brightness(ImageEnhance.Contrast(rgb).enhance(0.9)).enhance(1.18)
def _apply_low_key(rgb): return ImageEnhance.Brightness(ImageEnhance.Contrast(rgb).enhance(1.25)).enhance(0.82)
def _apply_cross_process(rgb): return _tone_shift(rgb, 1.1, 0.95, 1.15, 1.15)
def _apply_bleach_bypass(rgb): return ImageEnhance.Color(ImageEnhance.Contrast(rgb).enhance(1.35)).enhance(0.55)
def _apply_warm_film(rgb):
    tinted = _tone_shift(rgb, 1.08, 1.02, 0.92)
    arr = _add_grain(np.array(tinted), 0.02)
    return Image.fromarray(arr)
def _apply_cool_film(rgb):
    img = _tone_shift(rgb, 0.94, 1.0, 1.08, 1.02)
    return Image.fromarray(_add_grain(np.array(img), 0.02))
def _apply_neon_pop(rgb): return ImageEnhance.Color(ImageEnhance.Contrast(rgb).enhance(1.2)).enhance(1.55)
def _apply_dusty(rgb): return _tone_shift(ImageEnhance.Color(rgb).enhance(0.8), 1.06, 1.0, 0.9, 0.92)
def _apply_editorial(rgb): return ImageEnhance.Sharpness(ImageEnhance.Contrast(rgb).enhance(1.08)).enhance(1.2)
def _apply_soft_glow(rgb): return ImageEnhance.Brightness(rgb.filter(ImageFilter.GaussianBlur(radius=1.5))).enhance(1.06)
def _apply_punchy(rgb): return ImageEnhance.Sharpness(ImageEnhance.Color(ImageEnhance.Contrast(rgb).enhance(1.25)).enhance(1.3)).enhance(1.4)
def _apply_earthy(rgb): return _tone_shift(rgb, 1.06, 1.02, 0.88, 1.05)
def _apply_aqua(rgb): return _tone_shift(rgb, 0.9, 1.05, 1.14, 1.08)
def _apply_lavender(rgb): return _tone_shift(rgb, 1.02, 0.96, 1.1, 1.02)
def _apply_retro_pop(rgb): return ImageEnhance.Color(_tone_shift(rgb, 1.12, 0.98, 1.18, 1.15)).enhance(1.4)
def _apply_ink(rgb): return _tone_shift(_apply_bw_dramatic(rgb), 0.85, 0.9, 1.2, 1.2)
def _apply_honey(rgb): return _tone_shift(rgb, 1.16, 1.08, 0.82, 1.06)
def _apply_slate(rgb): return ImageEnhance.Color(_tone_shift(rgb, 0.95, 0.97, 1.0, 1.1)).enhance(0.5)
def _apply_velvia(rgb): return ImageEnhance.Sharpness(ImageEnhance.Color(rgb).enhance(1.5)).enhance(1.25)
def _apply_twilight(rgb): return _tone_shift(rgb, 0.92, 0.94, 1.16, 1.12)


_STYLE_APPLIERS = {
    "cinematic": _apply_cinematic,
    "vintage": _apply_vintage,
    "vivid": _apply_vivid,
    "portrait": _apply_portrait,
    "bw_dramatic": _apply_bw_dramatic,
    "golden_hour": _apply_golden_hour,
    "cool_moody": _apply_cool_moody,
    "faded": _apply_faded,
    "noir": _apply_noir,
    "sepia": _apply_sepia,
    "pastel": _apply_pastel,
    "arctic": _apply_arctic,
    "sunset": _apply_sunset,
    "emerald": _apply_emerald,
    "rose_gold": _apply_rose_gold,
    "matte": _apply_matte,
    "high_key": _apply_high_key,
    "low_key": _apply_low_key,
    "cross_process": _apply_cross_process,
    "bleach_bypass": _apply_bleach_bypass,
    "warm_film": _apply_warm_film,
    "cool_film": _apply_cool_film,
    "neon_pop": _apply_neon_pop,
    "dusty": _apply_dusty,
    "editorial": _apply_editorial,
    "soft_glow": _apply_soft_glow,
    "punchy": _apply_punchy,
    "earthy": _apply_earthy,
    "aqua": _apply_aqua,
    "lavender": _apply_lavender,
    "retro_pop": _apply_retro_pop,
    "ink": _apply_ink,
    "honey": _apply_honey,
    "slate": _apply_slate,
    "velvia": _apply_velvia,
    "twilight": _apply_twilight,
}


def apply_style(image: Image.Image, style: str | None) -> Image.Image:
    style_id = normalize_style(style)
    fn = _STYLE_APPLIERS.get(style_id, _apply_cinematic)
    rgb = image.convert("RGB")
    result = fn(rgb)
    return _preserve_alpha(image, result)
