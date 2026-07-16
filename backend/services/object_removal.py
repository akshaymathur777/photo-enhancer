from __future__ import annotations

import numpy as np
from PIL import Image

from services.inpaint_backends import (
    InpaintError,
    RemovalMode,
    get_backend,
    inpaint_patch_opencv,
)

_sam_predictor = None
_yolo_seg = None


def _get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None:
        from ultralytics.models.sam import SAM2Predictor

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=1024,
            model="sam2.1_b.pt",
        )
        _sam_predictor = SAM2Predictor(overrides=overrides)
    return _sam_predictor


def _get_yolo_seg():
    global _yolo_seg
    if _yolo_seg is None:
        from ultralytics import YOLO

        _yolo_seg = YOLO("yolov8n-seg.pt")
    return _yolo_seg


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _mask_tensor_to_numpy(mask) -> np.ndarray:
    if hasattr(mask, "detach"):
        return mask.detach().cpu().numpy()
    return np.asarray(mask)


def _best_sam_mask_from_results(
    results,
    h: int,
    w: int,
    click_pt: list[float] | None = None,
) -> np.ndarray | None:
    """Pick the best SAM multimask: must cover the click, then prefer confidence + area."""
    import cv2

    if not results:
        return None

    r = results[0]
    masks = r.masks
    if masks is None or masks.data is None:
        return None

    data = masks.data
    try:
        if hasattr(data, "numel") and int(data.numel()) == 0:
            return None
        n_masks = int(data.shape[0]) if len(data.shape) > 0 else 0
        if n_masks == 0:
            return None
    except (TypeError, RuntimeError):
        return None

    scores = None
    if r.boxes is not None and r.boxes.conf is not None and len(r.boxes.conf) >= n_masks:
        scores = r.boxes.conf.detach().cpu().numpy()[:n_masks]

    best_idx = 0
    best_score = -1.0
    cx = cy = None
    if click_pt is not None:
        cx, cy = int(round(click_pt[0])), int(round(click_pt[1]))

    img_area = float(h * w)
    for i in range(n_masks):
        mask_i = _mask_tensor_to_numpy(data[i])
        if mask_i.shape != (h, w):
            mask_i = cv2.resize(mask_i.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        score = float(scores[i]) if scores is not None else 0.5
        covers_click = (
            cx is not None
            and cy is not None
            and 0 <= cx < w
            and 0 <= cy < h
            and mask_i[cy, cx] > 0.5
        )
        if not covers_click:
            score -= 1.0
        else:
            area_frac = float((mask_i > 0.5).sum()) / img_area
            score += min(0.35, area_frac * 0.6)

        if score > best_score:
            best_score = score
            best_idx = i

    mask = _mask_tensor_to_numpy(data[best_idx])
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return mask


def segment_from_points(
    image: Image.Image,
    points: list[list[float]],
    labels: list[int],
) -> Image.Image:
    """Generate a binary mask from click points using SAM 2.1."""
    arr = _pil_to_rgb_array(image)
    h, w = arr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    predictor = _get_sam_predictor()
    predictor.set_image(arr)

    positive = [(p, l) for p, l in zip(points, labels) if l == 1]
    negative = [(p, l) for p, l in zip(points, labels) if l == 0]

    if not positive:
        return Image.fromarray(combined, mode="L")

    for pt, _ in positive:
        results = predictor(points=[pt], labels=[1], multimask_output=True)
        mask = _best_sam_mask_from_results(results, h, w, click_pt=pt)
        if mask is not None:
            combined = np.maximum(combined, (mask > 0.5).astype(np.uint8) * 255)

    if negative:
        neg_pts = [p for p, _ in negative]
        neg_lbls = [0] * len(neg_pts)
        results = predictor(points=neg_pts, labels=neg_lbls, multimask_output=False)
        mask = _best_sam_mask_from_results(results, h, w)
        if mask is not None:
            combined = np.where(mask > 0.5, 0, combined)

    return Image.fromarray(combined, mode="L")


def auto_detect_people_mask(image: Image.Image, conf: float = 0.35) -> Image.Image:
    """Build a combined mask for all detected people (YOLOv8-seg, COCO class 0)."""
    import cv2

    arr = _pil_to_rgb_array(image)
    h, w = arr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    model = _get_yolo_seg()
    results = model.predict(arr, classes=[0], conf=conf, verbose=False)
    if not results:
        return Image.fromarray(combined, mode="L")

    r = results[0]
    if r.masks is None or r.masks.data is None:
        return Image.fromarray(combined, mode="L")

    for mask_t in r.masks.data:
        mask = _mask_tensor_to_numpy(mask_t)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        combined = np.maximum(combined, (mask > 0.5).astype(np.uint8) * 255)

    return Image.fromarray(combined, mode="L")


def refine_mask_hard(
    mask: Image.Image,
    dilate_px: int = 6,
    edge_softness: float = 1.0,
) -> np.ndarray:
    """Close holes, smooth boundary, and dilate — returns a hard 0/255 mask for inpainting."""
    import cv2

    softness = max(0.5, min(float(edge_softness), 2.0))
    dilate = max(0, int(round(dilate_px * softness)))

    arr = np.array(mask.convert("L"))
    binary = (arr > 127).astype(np.uint8)

    if binary.any():
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=1)
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=1)

    if dilate > 0:
        k = dilate if dilate % 2 == 1 else dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel, iterations=1)

    return binary * 255


def refine_mask(
    mask: Image.Image,
    dilate_px: int = 6,
    feather_px: int = 10,
    edge_softness: float = 1.0,
) -> Image.Image:
    """Refine mask edges for preview/tests — hard core + single distance-based feather."""
    softness = max(0.5, min(float(edge_softness), 2.0))
    feather = max(4, int(round(feather_px * softness)))
    hard = refine_mask_hard(mask, dilate_px=dilate_px, edge_softness=edge_softness)
    soft = _build_blend_mask(hard, feather)
    return Image.fromarray((soft * 255.0).astype(np.uint8), mode="L")


def _build_blend_mask(hard_mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Signed-distance soft alpha: 1 inside mask, smooth falloff outside."""
    import cv2

    binary = (hard_mask > 127).astype(np.uint8)
    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    signed = dist_in - dist_out
    ramp = max(feather_px, 2)
    blend = np.clip((signed + ramp) / (2.0 * ramp), 0.0, 1.0)
    k = ramp if ramp % 2 == 1 else ramp + 1
    blend = cv2.GaussianBlur(blend.astype(np.float32), (k, k), sigmaX=ramp * 0.15)
    return np.clip(blend, 0.0, 1.0)


def _smooth_transition_band(
    patch: np.ndarray,
    original: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    """Edge-aware bilateral smoothing only in the mask transition band."""
    import cv2

    band = (blend > 0.08) & (blend < 0.92)
    if not band.any():
        return patch

    smoothed = cv2.bilateralFilter(patch, d=7, sigmaColor=35, sigmaSpace=35)
    out = patch.copy()
    mix = np.clip(1.0 - np.abs(blend - 0.5) * 2.0, 0.0, 1.0)[:, :, None]
    blended = (smoothed.astype(np.float32) * mix + patch.astype(np.float32) * (1.0 - mix)).astype(np.uint8)
    out = np.where(band[:, :, None], blended, out)
    out = np.where((blend[:, :, None] < 0.02), original, out)
    return out


def _try_seamless_blend(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Poisson seamless clone when mask is large enough; None on failure."""
    import cv2

    hard = (mask > 127).astype(np.uint8) * 255
    ys, xs = np.where(hard > 0)
    if len(xs) < 16:
        return None

    h, w = original.shape[:2]
    cx = int(np.clip(xs.mean(), 8, w - 9))
    cy = int(np.clip(ys.mean(), 8, h - 9))
    if hard[cy, cx] == 0:
        cx, cy = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])

    try:
        return cv2.seamlessClone(inpainted, original, hard, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        try:
            return cv2.seamlessClone(inpainted, original, hard, (cx, cy), cv2.MIXED_CLONE)
        except cv2.error:
            return None


def _crop_padding(bbox_w: int, bbox_h: int, img_h: int, img_w: int) -> int:
    return max(96, int(max(bbox_w, bbox_h) * 0.28), int(max(img_h, img_w) * 0.05))


def _ai_edge_polish(
    patch: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    """Blend LaMa with OpenCV TELEA at edges for Photoshop-like content-aware transitions."""
    try:
        cv_fill = inpaint_patch_opencv(original, mask)
        edge_w = np.clip(1.0 - np.abs(blend - 0.5) * 2.0, 0.0, 1.0)
        edge_w = (edge_w ** 1.5)[:, :, None]
        mixed = (patch.astype(np.float32) * (1.0 - edge_w * 0.35) + cv_fill.astype(np.float32) * (edge_w * 0.35))
        return np.clip(mixed, 0, 255).astype(np.uint8)
    except Exception:
        return patch


def inpaint_objects(
    image: Image.Image,
    mask: Image.Image,
    mode: RemovalMode = "ai",
    edge_softness: float = 1.0,
) -> Image.Image:
    """Remove masked regions using AI Remove (torch big-LaMa) or Quick Remove (ONNX LaMa)."""
    rgb = _pil_to_rgb_array(image)
    h, w = rgb.shape[:2]

    msk = np.array(mask.convert("L"))
    if msk.max() == 0:
        return image.convert("RGB")

    hard_mask = refine_mask_hard(mask, edge_softness=edge_softness)
    ys, xs = np.where(hard_mask > 127)
    if len(xs) == 0:
        return image.convert("RGB")

    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    pad = _crop_padding(bbox_w, bbox_h, h, w)
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(w, xs.max() + pad)
    y1 = min(h, ys.max() + pad)

    crop_rgb = rgb[y0:y1, x0:x1].copy()
    crop_hard = hard_mask[y0:y1, x0:x1].copy()
    feather_px = max(6, int(round(10 * max(0.5, min(edge_softness, 2.0)))))
    use_seamless = mode == "ai"

    backend = get_backend(mode)
    try:
        inpainted_crop = backend.inpaint_crop(crop_rgb, crop_hard)
    except InpaintError as exc:
        if mode == "ai" and "blank white fill" in str(exc).lower():
            try:
                from services.inpaint_backends import QuickRemoveBackend

                inpainted_crop = QuickRemoveBackend().inpaint_crop(crop_rgb, crop_hard)
            except InpaintError:
                raise
        else:
            raise

    seamless = _try_seamless_blend(crop_rgb, inpainted_crop, crop_hard) if use_seamless else None
    base_patch = seamless if seamless is not None else inpainted_crop

    blend = _build_blend_mask(crop_hard, feather_px)
    patch = (
        base_patch.astype(np.float32) * blend[:, :, None]
        + crop_rgb.astype(np.float32) * (1.0 - blend[:, :, None])
    ).astype(np.uint8)
    patch = _smooth_transition_band(patch, crop_rgb, blend)

    if mode == "ai":
        patch = _ai_edge_polish(patch, crop_rgb, crop_hard, blend)

    result = rgb.copy()
    result[y0:y1, x0:x1] = patch
    return Image.fromarray(result)


def mask_overlay_preview(image: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    """Return image with red highlight over the selection mask."""
    rgb = _pil_to_rgb_array(image)
    msk = np.array(mask.convert("L")).astype(np.float32) / 255.0

    overlay = rgb.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    m3 = msk[:, :, None] * alpha
    blended = overlay * (1 - m3) + red * m3
    return Image.fromarray(blended.astype(np.uint8))
