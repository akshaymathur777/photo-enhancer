"""Portrait retouch: face-aware smooth, magic preset, red-eye, teeth, hair tint, lip/blush."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from services.ai_styles import apply_style
from services.image_processing import auto_enhance

_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        try:
            import mediapipe as mp

            _face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        except ImportError:
            _face_detector = False
    return _face_detector if _face_detector is not False else None


def _face_mask(image: Image.Image) -> np.ndarray | None:
    detector = _get_face_detector()
    h, w = image.size[1], image.size[0]
    if detector is None:
        return None
    rgb = np.array(image.convert("RGB"))
    results = detector.process(rgb)
    if not results.detections:
        return None
    mask = np.zeros((h, w), dtype=np.float32)
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w, int((bb.xmin + bb.width) * w))
        y2 = min(h, int((bb.ymin + bb.height) * h))
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        mask[y1:y2, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return mask


def _center_fallback_mask(image: Image.Image) -> np.ndarray:
    h, w = image.size[1], image.size[0]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_d = min(w, h) * 0.35
    return np.clip(1.0 - (dist / max_d) ** 2, 0, 1).astype(np.float32)


def skin_smooth(image: Image.Image, strength: float = 0.6) -> Image.Image:
    s = max(0.1, min(1.0, strength))
    arr = np.array(image.convert("RGB"))
    smooth = cv2.bilateralFilter(arr, d=9, sigmaColor=50, sigmaSpace=50)
    mask = _face_mask(image)
    if mask is None:
        mask = _center_fallback_mask(image)
    blend = mask[:, :, None] * s
    out = (smooth.astype(np.float32) * blend + arr.astype(np.float32) * (1 - blend)).astype(np.uint8)
    result = Image.fromarray(out)
    if image.mode == "RGBA":
        result = result.convert("RGBA")
        result.putalpha(image.split()[3])
    return result


def portrait_magic(image: Image.Image, strength: float = 1.0) -> Image.Image:
    s = max(0.2, min(2.0, strength))
    result = auto_enhance(image, strength=0.9 * s)
    result = skin_smooth(result, strength=0.45 * s)
    result = apply_style(result, "portrait")
    return result


def fix_red_eye(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 80, 50])
    upper = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([170, 80, 50])
    mask2 = cv2.inRange(hsv, lower2, upper)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if mask.sum() < 100:
        return image
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr[mask > 0] = gray[mask > 0, None]
    result = Image.fromarray(arr)
    if image.mode == "RGBA":
        result = result.convert("RGBA")
        result.putalpha(image.split()[3])
    return result


def whiten_teeth(image: Image.Image, strength: float = 0.5) -> Image.Image:
    s = max(0.1, min(1.0, strength))
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    teeth_mask = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 140) & (hsv[:, :, 0] < 35)
    teeth_mask = teeth_mask.astype(np.float32)
    teeth_mask = cv2.GaussianBlur(teeth_mask, (15, 15), 0)
    boost = teeth_mask[:, :, None] * s * 25
    arr = np.clip(arr + boost, 0, 255)
    result = Image.fromarray(arr.astype(np.uint8))
    if image.mode == "RGBA":
        result = result.convert("RGBA")
        result.putalpha(image.split()[3])
    return result


def hair_tint(image: Image.Image, hue_shift: float = 0.0, saturation: float = 0.3) -> Image.Image:
    """Simple hair tint via portrait-region HSL shift (quality varies)."""
    from services.image_processing import remove_background

    cutout = remove_background(image, "portrait")
    if cutout.mode != "RGBA":
        return image
    alpha = np.array(cutout.split()[3])
    hair_mask = cv2.erode(alpha, np.ones((5, 5), np.uint8), iterations=2)
    hair_mask = hair_mask.astype(np.float32) / 255.0
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)

    arr = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation), 0, 255)
    tinted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    blend = hair_mask[:, :, None] * 0.6
    out = (tinted.astype(np.float32) * blend + arr.astype(np.float32) * (1 - blend)).astype(np.uint8)
    result = Image.fromarray(out)
    if image.mode == "RGBA":
        result = result.convert("RGBA")
        result.putalpha(image.split()[3])
    return result


def lip_blush_overlay(
    image: Image.Image,
    lip_color: str = "#c44",
    blush_color: str = "#f8a",
    intensity: float = 0.35,
) -> Image.Image:
    """Basic lip/cheek overlay using face region heuristics."""
    s = max(0.05, min(0.8, intensity))
    mask = _face_mask(image)
    if mask is None:
        mask = _center_fallback_mask(image)
    h, w = image.size[1], image.size[0]
    arr = np.array(image.convert("RGB"), dtype=np.float32)

    lip_r = int(lip_color[1:3], 16)
    lip_g = int(lip_color[3:5], 16)
    lip_b = int(lip_color[5:7], 16)
    blush_r = int(blush_color[1:3], 16)
    blush_g = int(blush_color[3:5], 16)
    blush_b = int(blush_color[5:7], 16)

    lip_region = np.zeros((h, w), dtype=np.float32)
    lip_region[int(h * 0.55) : int(h * 0.72), int(w * 0.35) : int(w * 0.65)] = 1.0
    lip_region = cv2.GaussianBlur(lip_region * mask, (31, 31), 0)

    cheek_region = np.zeros((h, w), dtype=np.float32)
    cheek_region[int(h * 0.45) : int(h * 0.62), int(w * 0.2) : int(w * 0.4)] = 1.0
    cheek_region[int(h * 0.45) : int(h * 0.62), int(w * 0.6) : int(w * 0.8)] = 1.0
    cheek_region = cv2.GaussianBlur(cheek_region * mask, (41, 41), 0)

    lip_overlay = np.zeros_like(arr)
    lip_overlay[..., 0], lip_overlay[..., 1], lip_overlay[..., 2] = lip_r, lip_g, lip_b
    blush_overlay = np.zeros_like(arr)
    blush_overlay[..., 0], blush_overlay[..., 1], blush_overlay[..., 2] = blush_r, blush_g, blush_b

    arr = arr * (1 - lip_region[:, :, None] * s * 0.5) + lip_overlay * lip_region[:, :, None] * s * 0.5
    arr = arr * (1 - cheek_region[:, :, None] * s * 0.35) + blush_overlay * cheek_region[:, :, None] * s * 0.35
    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if image.mode == "RGBA":
        result = result.convert("RGBA")
        result.putalpha(image.split()[3])
    return result
