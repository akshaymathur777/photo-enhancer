"""Basic relight presets via curves and radial gradients (CPU-friendly)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

RELIGHT_PRESETS: dict[str, dict] = {
    "flash": {"warmth": 0.05, "brightness": 1.12, "contrast": 1.08, "radial": 0.3},
    "sunsoft": {"warmth": 0.15, "brightness": 1.08, "contrast": 0.95, "radial": 0.5},
    "golden_hour": {"warmth": 0.28, "brightness": 1.1, "contrast": 1.05, "radial": 0.4},
    "blue_hour": {"warmth": -0.18, "brightness": 0.96, "contrast": 1.1, "radial": 0.35},
    "even_light": {"warmth": 0.0, "brightness": 1.06, "contrast": 0.98, "radial": 0.0},
    "deglare": {"warmth": -0.05, "brightness": 0.92, "contrast": 0.9, "radial": -0.2},
    "window": {"warmth": 0.08, "brightness": 1.05, "contrast": 1.0, "radial": 0.6},
    "neon": {"warmth": -0.12, "brightness": 1.04, "contrast": 1.15, "radial": 0.25},
    "backlight": {"warmth": 0.1, "brightness": 1.14, "contrast": 1.12, "radial": 0.7},
}

DEFAULT_PRESET = "even_light"


def list_relight_presets() -> list[dict]:
    labels = {
        "flash": "Flash",
        "sunsoft": "Sunsoft",
        "golden_hour": "Golden Hour",
        "blue_hour": "Blue Hour",
        "even_light": "Even Light",
        "deglare": "DeGlare",
        "window": "Window",
        "neon": "Neon",
        "backlight": "Backlight",
    }
    return [{"id": k, "name": labels.get(k, k.title()), "description": f"{labels.get(k, k)} lighting preset"} for k in RELIGHT_PRESETS]


def normalize_preset(preset: str | None) -> str:
    if not preset:
        return DEFAULT_PRESET
    key = preset.lower().strip().replace("-", "_").replace(" ", "_")
    return key if key in RELIGHT_PRESETS else DEFAULT_PRESET


def apply_relight(image: Image.Image, preset: str | None = None, strength: float = 1.0) -> Image.Image:
    cfg = RELIGHT_PRESETS[normalize_preset(preset)]
    s = max(0.1, min(2.0, strength))
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[3] if has_alpha else None
    arr = np.array(image.convert("RGB"), dtype=np.float32)

    warmth = cfg["warmth"] * s
    arr[..., 0] = np.clip(arr[..., 0] * (1 + warmth) + warmth * 20, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * (1 - warmth * 0.6), 0, 255)

    radial = cfg["radial"] * s
    if abs(radial) > 0.01:
        h, w = arr.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w * 0.45, h * 0.35
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_d = np.sqrt(w**2 + h**2) * 0.5
        glow = np.clip(1.0 - dist / max_d, 0, 1) ** 2
        boost = glow[:, :, None] * radial * 40
        arr = np.clip(arr + boost, 0, 255)

    result = Image.fromarray(arr.astype(np.uint8))
    result = ImageEnhance.Brightness(result).enhance(1.0 + (cfg["brightness"] - 1.0) * s)
    result = ImageEnhance.Contrast(result).enhance(1.0 + (cfg["contrast"] - 1.0) * s)

    if has_alpha and alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    return result
