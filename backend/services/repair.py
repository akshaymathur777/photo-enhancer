"""AI Repair and photo restoration lite."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from services.image_processing import auto_enhance, denoise, sharpen
from services.object_removal import inpaint_objects


def _auto_scratch_mask(image: Image.Image) -> Image.Image:
    """Detect thin scratch-like edges for repair."""
    gray = np.array(image.convert("L"))
    edges = cv2.Canny(gray, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    return Image.fromarray(dilated, mode="L")


def ai_repair(image: Image.Image, mask: Image.Image | None = None, auto_detect: bool = False) -> Image.Image:
    if mask is None and auto_detect:
        mask = _auto_scratch_mask(image)
    if mask is None:
        raise ValueError("Provide a mask or enable auto_detect")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    if not mask.getbbox():
        return image
    return inpaint_objects(image, mask, mode="ai", edge_softness=1.0)


def photo_restore_lite(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Denoise + sharpen + auto enhance chain for damaged/low-quality photos."""
    s = max(0.2, min(2.0, strength))
    result = denoise(image, strength=int(8 + 6 * s))
    result = sharpen(result, amount=1.0 + 0.5 * s)
    result = auto_enhance(result, strength=0.8 + 0.4 * s)
    return result
