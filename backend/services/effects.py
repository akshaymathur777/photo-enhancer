"""Creative effects: bokeh, blur, prism, vignette, grain."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from services.image_processing import DEFAULT_BG_MODEL, remove_background


def _preserve_alpha(image: Image.Image, result: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        out = result.convert("RGBA")
        out.putalpha(image.split()[3])
        return out
    return result


def _subject_mask(image: Image.Image, model_key: str = DEFAULT_BG_MODEL) -> np.ndarray:
    cutout = remove_background(image, model_key)
    if cutout.mode != "RGBA":
        return np.ones(cutout.size[::-1], dtype=np.uint8) * 255
    alpha = np.array(cutout.split()[3])
    return alpha


def apply_bokeh(
    image: Image.Image,
    blur_radius: int = 25,
    shape: str = "circle",
    depth_aware: bool = False,
    model_key: str = DEFAULT_BG_MODEL,
) -> Image.Image:
    """Subject-aware background blur with optional depth falloff."""
    rgb = np.array(image.convert("RGB"))
    mask = _subject_mask(image, model_key)
    h, w = mask.shape

    k = max(3, blur_radius if blur_radius % 2 == 1 else blur_radius + 1)
    if shape == "heart":
        kernel = _heart_kernel(k)
        blurred = cv2.filter2D(rgb, -1, kernel / kernel.sum())
    elif shape == "hexagon":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float32)
        blurred = cv2.filter2D(rgb, -1, kernel / kernel.sum())
    else:
        blurred = cv2.GaussianBlur(rgb, (k, k), 0)

    if depth_aware:
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        dist = dist / (dist.max() + 1e-6)
        blend = np.clip(dist * 1.2, 0, 1)[:, :, None]
    else:
        blend = (1.0 - (mask.astype(np.float32) / 255.0))[:, :, None]

    subject = mask.astype(np.float32) / 255.0
    out = (rgb.astype(np.float32) * subject[:, :, None] + blurred.astype(np.float32) * blend).astype(np.uint8)
    return _preserve_alpha(image, Image.fromarray(out))


def _heart_kernel(size: int) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    for y in range(size):
        for x in range(size):
            nx = (x - cx) / (size / 2)
            ny = (y - cy) / (size / 2)
            if (nx * nx + ny * ny - 1) ** 3 - nx * nx * ny * ny * ny <= 0:
                k[y, x] = 1.0
    if k.sum() == 0:
        k[cy, cx] = 1.0
    return k


def apply_blur(
    image: Image.Image,
    radius: int = 15,
    selective: bool = False,
    model_key: str = DEFAULT_BG_MODEL,
) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    k = max(3, radius if radius % 2 == 1 else radius + 1)
    blurred = cv2.GaussianBlur(rgb, (k, k), 0)
    if not selective:
        return _preserve_alpha(image, Image.fromarray(blurred))
    mask = _subject_mask(image, model_key)
    subject = mask.astype(np.float32) / 255.0
    out = (rgb.astype(np.float32) * subject[:, :, None] + blurred.astype(np.float32) * (1 - subject[:, :, None])).astype(np.uint8)
    return _preserve_alpha(image, Image.fromarray(out))


def apply_prism(image: Image.Image, strength: float = 0.6) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    s = max(0.1, min(1.5, strength))
    shift = max(2, int(w * 0.008 * s))
    r = np.roll(arr[:, :, 0], shift, axis=1)
    b = np.roll(arr[:, :, 2], -shift, axis=1)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 15 * s) % 180
    tinted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = np.stack([r, tinted[:, :, 1], b], axis=-1)
    out = np.clip(arr * (1 - s * 0.5) + out * (s * 0.5), 0, 255).astype(np.uint8)
    return _preserve_alpha(image, Image.fromarray(out))


def apply_vignette(image: Image.Image, strength: float = 0.5) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_d = np.sqrt(cx**2 + cy**2)
    vignette = 1.0 - (dist / max_d) ** 1.8 * max(0, min(1, strength))
    out = np.clip(arr * vignette[:, :, None], 0, 255).astype(np.uint8)
    return _preserve_alpha(image, Image.fromarray(out))


def apply_grain(image: Image.Image, amount: float = 0.04) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, amount * 255, arr.shape)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return _preserve_alpha(image, Image.fromarray(out))
