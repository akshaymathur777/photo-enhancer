"""Classical (CPU) image ops for Option B — no CUDA / no host image APIs."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def decode_image_b64(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def auto_enhance(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Mild classical enhance — mirrors desktop auto_enhance spirit."""
    s = max(0.0, min(float(strength), 2.0))
    result = image.convert("RGB")
    result = ImageEnhance.Color(result).enhance(1.0 + 0.08 * s)
    result = ImageEnhance.Brightness(result).enhance(1.0 + 0.03 * s)
    result = ImageEnhance.Contrast(result).enhance(1.0 + 0.06 * s)
    result = ImageEnhance.Sharpness(result).enhance(1.0 + 0.12 * s)
    if s >= 1.2:
        result = result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))
    return result


def inpaint_opencv(
    image: Image.Image,
    mask: Image.Image,
    method: str = "telea",
) -> Image.Image:
    import cv2

    rgb = np.array(image.convert("RGB"))
    m = np.array(mask.convert("L"))
    hard = ((m > 127).astype(np.uint8)) * 255
    ys, xs = np.where(hard > 0)
    if len(xs) == 0:
        return image.convert("RGB")
    hole = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    radius = max(5, int(min(rgb.shape[:2]) * 0.01), int(hole * 0.02))
    radius = min(radius, 18)
    flag = cv2.INPAINT_NS if method.lower() == "ns" else cv2.INPAINT_TELEA
    out = cv2.inpaint(rgb, hard, radius, flag)
    return Image.fromarray(out)


def remove_bg_rembg(image: Image.Image, model_key: str = "lite") -> Image.Image:
    """Optional rembg path — requires rembg + onnxruntime in the local env.

    ``lite`` uses ``u2netp`` (~4MB) so first-run downloads stay small.
    ``general`` / ``portrait`` use heavier BiRefNet checkpoints when installed.
    """
    from rembg import new_session, remove

    model_map = {
        "lite": "u2netp",
        "general": "u2net",
        "portrait": "birefnet-portrait",
    }
    model = model_map.get(model_key, model_map["lite"])
    session = new_session(model)
    out = remove(image.convert("RGBA"), session=session)
    return out if isinstance(out, Image.Image) else Image.open(io.BytesIO(out))


def result_payload(image: Image.Image, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_png_b64": encode_png_b64(image),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        **meta,
    }
