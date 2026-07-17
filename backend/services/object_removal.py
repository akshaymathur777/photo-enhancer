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
    """Pick the best SAM multimask: must cover the click, then prefer confidence + compactness."""
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
            # Prefer person-sized masks; penalize sky/background blobs that swallow the scene.
            if area_frac > 0.40:
                score -= 0.55
            elif area_frac > 0.25:
                score -= 0.25
            else:
                compact = 1.0 - min(1.0, area_frac / 0.22)
                score += 0.30 * compact

        if score > best_score:
            best_score = score
            best_idx = i

    mask = _mask_tensor_to_numpy(data[best_idx])
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return mask


def _yolo_person_mask_containing_click(
    arr: np.ndarray,
    click_pt: list[float] | tuple[float, float],
    conf: float = 0.25,
) -> np.ndarray | None:
    """Return YOLO-seg mask for the person instance that contains the click, if any.

    SAM click masks often cover only the head/face on crowded selfies. Unioning with
    the matching YOLOv8-seg person instance gives a full torso for background people.
    """
    import cv2

    h, w = arr.shape[:2]
    cx, cy = int(round(float(click_pt[0]))), int(round(float(click_pt[1])))
    if not (0 <= cx < w and 0 <= cy < h):
        return None

    model = _get_yolo_seg()
    results = model.predict(arr, classes=[0], conf=conf, verbose=False)
    if not results:
        return None
    r = results[0]
    if r.masks is None or r.masks.data is None or r.boxes is None:
        return None

    best = None
    best_area = None
    for i, mask_t in enumerate(r.masks.data):
        mask = _mask_tensor_to_numpy(mask_t)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = mask > 0.5
        if not binary[cy, cx]:
            # Also accept click inside the person's bounding box (shirt clicks near edge).
            try:
                xyxy = r.boxes.xyxy[i].detach().cpu().numpy().tolist()
                x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
                if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                    continue
            except Exception:
                continue
        area = int(binary.sum())
        if area < 200:
            continue
        # Prefer the smallest person instance that contains the click (not the whole crowd).
        if best is None or area < best_area:
            best = binary.astype(np.uint8)
            best_area = area

    return best


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """True where mask is selected. Handles bool, 0/1, and 0/255 masks."""
    if mask.dtype == bool:
        return mask
    if mask.dtype == np.uint8 or np.issubdtype(mask.dtype, np.integer):
        # 0/1 masks from astype(uint8) must use >0, not >127.
        return mask > 0 if int(mask.max()) <= 1 else mask > 127
    return mask > 0.5


def _other_people_protect_mask(
    arr: np.ndarray,
    click_pts: list[tuple[float, float] | list[float]],
) -> np.ndarray:
    """Dilated mask of people who are NOT the clicked target(s) — must never be filled."""
    import cv2

    h, w = arr.shape[:2]
    all_people = _all_people_binary(arr) > 0
    if not all_people.any():
        return np.zeros((h, w), dtype=bool)

    target = np.zeros((h, w), dtype=np.uint8)
    for pt in click_pts:
        yp = _yolo_person_mask_containing_click(arr, pt)
        if yp is not None:
            target = np.maximum(target, yp.astype(np.uint8))
        cx, cy = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= cx < w and 0 <= cy < h:
            # Tiny seed so we always have a target core even if YOLO misses.
            cv2.circle(target, (cx, cy), 12, 1, -1)

    others = all_people & (~(target > 0))
    if not others.any():
        return np.zeros((h, w), dtype=bool)
    # Dilate for hair, but never claim the clicked subject (avoids shirt-mask holes).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    protect = cv2.dilate(others.astype(np.uint8), k, iterations=1).astype(bool)
    target_keep = cv2.dilate(target, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    return protect & (~target_keep)


def _non_target_people_protect(rgb: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    """Protect people who are not the removal subject (by YOLO IoU vs selection).

    Unlike ``all_people & ~erode(raw)``, this does NOT treat the target's own YOLO
    pixels as protected — only other person instances.
    """
    import cv2

    h, w = rgb.shape[:2]
    raw_b = _as_bool_mask(raw_mask)
    if not raw_b.any():
        return np.zeros((h, w), dtype=bool)

    protect = np.zeros((h, w), dtype=bool)
    try:
        model = _get_yolo_seg()
        results = model.predict(rgb, classes=[0], conf=0.25, verbose=False)
    except Exception:
        return protect
    if not results or results[0].masks is None or results[0].masks.data is None:
        return protect

    raw_area = float(raw_b.sum())
    img_area = float(h * w)
    for mask_t in results[0].masks.data:
        mask = _mask_tensor_to_numpy(mask_t)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = mask > 0.5
        area = float(binary.sum())
        if area < 200:
            continue
        # Ignore absurd full-frame / background-bloated detections.
        if area > 0.35 * img_area:
            continue
        overlap = float((binary & raw_b).sum())
        # Same person if they share a meaningful fraction of either mask.
        if overlap / max(area, 1.0) >= 0.18 or overlap / max(raw_area, 1.0) >= 0.25:
            continue
        protect |= binary

    if not protect.any():
        return protect
    # Hair halo, but never claim the removal subject (dilate expands into the third person).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    protect = cv2.dilate(protect.astype(np.uint8), k, iterations=1).astype(bool)
    # Keep a clear gap so protect cannot carve holes in the selection / shirt.
    raw_keep = cv2.dilate(raw_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    protect = protect & (~raw_keep)
    return protect


def _occlusion_cutout_protect(raw_mask: np.ndarray) -> np.ndarray:
    """Protect foreground cutouts that SAM already carved out of the selection.

    These holes (couple heads in front of the target) are the most reliable
    signal for what must never be painted — tighter than bloated YOLO masks.
    """
    import cv2

    raw_b = _as_bool_mask(raw_mask)
    if not raw_b.any():
        return np.zeros(raw_b.shape, dtype=bool)

    inv = (~raw_b).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    raw_area = float(raw_b.sum())
    protect = np.zeros_like(raw_b, dtype=bool)
    touch = cv2.dilate(raw_b.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(400, int(raw_area * 0.015)):
            continue
        if area > raw_area * 2.5:
            continue
        comp = labels == i
        if not (comp & touch).any():
            continue
        protect |= comp

    if not protect.any():
        return protect
    # Cover hair fringes around the cutout without reaching deep into the subject.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    protect = cv2.dilate(protect.astype(np.uint8), k, iterations=1).astype(bool)
    raw_keep = cv2.dilate(raw_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    return protect & (~raw_keep)


def _apply_people_protect(hard_u8: np.ndarray, protect: np.ndarray, raw: np.ndarray | None = None) -> np.ndarray:
    """Remove protected people from a hard mask; always keep the raw selection intact."""
    hard_b = _as_bool_mask(hard_u8)
    if protect is None or not protect.any():
        if raw is not None:
            return ((hard_b | _as_bool_mask(raw)).astype(np.uint8) * 255)
        return hard_u8
    out = hard_b & (~protect)
    if raw is not None:
        # Critical: never let protect punch holes in the clicked subject (shirt ghosts).
        out = out | _as_bool_mask(raw)
    return out.astype(np.uint8) * 255


def segment_from_points(
    image: Image.Image,
    points: list[list[float]],
    labels: list[int],
) -> Image.Image:
    """Generate a binary mask from click points using SAM 2.1 (+ YOLO person assist)."""
    import cv2

    arr = _pil_to_rgb_array(image)
    h, w = arr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    predictor = _get_sam_predictor()
    predictor.set_image(arr)

    positive = [(p, l) for p, l in zip(points, labels) if l == 1]
    negative = [(p, l) for p, l in zip(points, labels) if l == 0]

    if not positive:
        return Image.fromarray(combined, mode="L")

    # Auto negative points on other people so SAM prefers the clicked subject only.
    protect = _other_people_protect_mask(arr, [p for p, _ in positive])
    auto_neg: list[list[float]] = []
    if protect.any():
        ys, xs = np.where(protect)
        # Sample up to 6 points on protected people (couple).
        step = max(1, len(xs) // 6)
        for i in range(0, len(xs), step):
            auto_neg.append([float(xs[i]), float(ys[i])])
            if len(auto_neg) >= 6:
                break

    for pt, _ in positive:
        pts = [pt] + auto_neg
        lbls = [1] + [0] * len(auto_neg)
        results = predictor(points=pts, labels=lbls, multimask_output=True)
        mask = _best_sam_mask_from_results(results, h, w, click_pt=pt)
        if mask is not None:
            combined = np.maximum(combined, (mask > 0.5).astype(np.uint8) * 255)

        # Visible-person assist: YOLO full body, then cut away protected couple.
        yolo_person = _yolo_person_mask_containing_click(arr, pt)
        if yolo_person is not None:
            yolo_vis = yolo_person.astype(bool) & (~protect)
            # Keep YOLO only where it adds coverage near the SAM/click region.
            near = combined > 127
            if near.any():
                near = cv2.dilate(near.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0
                yolo_vis = yolo_vis & near
            combined = np.maximum(combined, yolo_vis.astype(np.uint8) * 255)

    if negative:
        neg_pts = [p for p, _ in negative]
        neg_lbls = [0] * len(neg_pts)
        results = predictor(points=neg_pts, labels=neg_lbls, multimask_output=False)
        mask = _best_sam_mask_from_results(results, h, w)
        if mask is not None:
            combined = np.where(mask > 0.5, 0, combined)

    # Final hard clip: never keep mask on protected people.
    if protect.any():
        combined = combined.copy()
        combined[protect] = 0

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
    dilate_px: int = 18,
    edge_softness: float = 1.0,
    rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Close holes, smooth boundary, and dilate — returns a hard 0/255 mask for inpainting.

    Larger default dilation covers hair / soft silhouette edges against water/sky.
    When ``rgb`` is provided, grow the mask into a subject-colored halo (hair fringes)
    without expanding into background-like pixels. Occlusion cutouts (people in front)
    are preserved so sides stay intact.
    """
    import cv2

    softness = max(0.5, min(float(edge_softness), 2.0))
    dilate = max(0, int(round(dilate_px * softness)))

    arr = np.array(mask.convert("L"))
    raw = (arr > 127).astype(np.uint8)
    binary = raw.copy()

    if binary.any():
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=2)
        # Intentionally no MORPH_OPEN — it strips thin hair strands that we need covered.

    if rgb is not None and binary.any():
        binary = _expand_mask_subject_halo(rgb, binary)

    if dilate > 0:
        k = dilate if dilate % 2 == 1 else dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel, iterations=1)

    binary = _preserve_occlusion_holes(raw, binary)
    return binary * 255


def _preserve_occlusion_holes(raw: np.ndarray, refined: np.ndarray) -> np.ndarray:
    """Keep SAM cutouts for people/objects in front from being filled by dilation."""
    import cv2

    raw_b = raw.astype(bool)
    ref_b = refined.astype(bool)
    if not raw_b.any() or not ref_b.any():
        return refined

    inv = (~raw_b).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    raw_area = float(raw_b.sum())
    protect = np.zeros_like(raw_b, dtype=bool)
    touch = cv2.dilate(raw_b.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Skip tiny flecks and the huge background (water/sky) component.
        if area < max(400, int(raw_area * 0.015)):
            continue
        if area > raw_area * 2.5:
            continue
        comp = labels == i
        if not (comp & touch).any():
            continue
        protect |= comp

    out = (ref_b & ~protect) | raw_b
    return out.astype(np.uint8)


def _expand_mask_subject_halo(
    rgb: np.ndarray,
    binary: np.ndarray,
    max_radius: int = 26,
    color_thresh: float = 38.0,
) -> np.ndarray:
    """Grow mask into nearby subject-like pixels (hair/ear fringes) vs background.

    Only expands within ``max_radius`` of the current mask, and only where the pixel
    is closer to subject colors than to the local background ring — so preserved
    neighbors (e.g. couple in front) are not swallowed when their colors differ.
    """
    import cv2

    hard = binary.astype(bool)
    if not hard.any():
        return binary

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    core = cv2.erode(hard.astype(np.uint8), erode_k, iterations=1)
    if int(core.sum()) < 64:
        core = hard.astype(np.uint8)

    # Edge band inside the mask captures hair/clothing boundary colors better than mean.
    edge_in = hard & ~(core.astype(bool))
    sample = edge_in if int(edge_in.sum()) >= 64 else hard
    subj = rgb[sample].astype(np.float32)
    subj_mean = subj.mean(axis=0)
    # Dark-hair cue: lower-luminance subject samples (top percentiles of darkness).
    lum = subj.mean(axis=1)
    dark_cut = float(np.percentile(lum, 35))
    dark_mean = subj[lum <= dark_cut].mean(axis=0) if np.any(lum <= dark_cut) else subj_mean

    rad = max(4, int(max_radius))
    k = rad * 2 + 1
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    band = (cv2.dilate(hard.astype(np.uint8), band_k) > 0) & (~hard)
    if not band.any():
        return binary

    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(15, rad * 2 + 1), max(15, rad * 2 + 1)))
    ring = (cv2.dilate(hard.astype(np.uint8), ring_k) > 0) & (~hard) & (~band)
    # Prefer a thin outer ring beyond the expansion band for true background stats.
    if int(ring.sum()) < 64:
        ring = (cv2.dilate(hard.astype(np.uint8), ring_k) > 0) & (~hard)
    bg_mean = rgb[ring].astype(np.float32).mean(axis=0) if int(ring.sum()) >= 64 else None

    pix = rgb.astype(np.float32)
    d_subj = np.abs(pix - subj_mean).mean(axis=2)
    d_dark = np.abs(pix - dark_mean).mean(axis=2)
    d_best = np.minimum(d_subj, d_dark)

    expand = band & (d_best < color_thresh)
    if bg_mean is not None:
        d_bg = np.abs(pix - bg_mean).mean(axis=2)
        # Must look more like subject/hair than like water/sky background.
        expand = expand & (d_best + 5.0 < d_bg)
        # Hair against bright water: dark fringe just outside the mask.
        bg_lum = float(bg_mean.mean())
        lum = pix.mean(axis=2)
        dark_hair = (
            band
            & (lum < bg_lum - 22.0)
            & (lum < float(dark_mean.mean()) + 8.0)
            & (d_best + 2.0 < d_bg)
        )
        expand = expand | dark_hair

    out = hard | expand
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out_u8 = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, close_k, iterations=1)
    return out_u8


def refine_mask(
    mask: Image.Image,
    dilate_px: int = 18,
    feather_px: int = 10,
    edge_softness: float = 1.0,
) -> Image.Image:
    """Refine mask edges for preview/tests — hard core + single distance-based feather."""
    feather = _blend_feather_px(edge_softness)
    hard = refine_mask_hard(mask, dilate_px=dilate_px, edge_softness=edge_softness)
    soft = _build_blend_mask(hard, feather)
    return Image.fromarray((soft * 255.0).astype(np.uint8), mode="L")


def _build_blend_mask(hard_mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Soft alpha that is fully opaque inside the hard mask, with falloff only outside.

    The previous signed-distance ramp put ~half the feather *inside* the selection,
    so original pixels (ghost silhouettes) bled back through the composite. For
    object removal the interior must stay at 1.0; soften only the outer halo.
    """
    import cv2

    binary = (hard_mask > 127).astype(np.uint8)
    ramp = max(int(feather_px), 2)
    dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    blend = np.where(
        binary > 0,
        1.0,
        np.clip(1.0 - dist_out / float(ramp), 0.0, 1.0),
    ).astype(np.float32)
    k = ramp if ramp % 2 == 1 else ramp + 1
    blend = cv2.GaussianBlur(blend, (k, k), sigmaX=max(ramp * 0.20, 1.0))
    # Blur softens the outer edge; never let it pull the hard interior below 1.0.
    blend = np.maximum(blend, binary.astype(np.float32))
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
    """Poisson seamless clone when it does not ghost the original back in.

    NORMAL/MIXED clone often washes LaMa fills toward the pre-removal colors
    (ghost silhouettes). Reject any clone that moves the fill closer to the
    original than raw LaMa already is.
    """
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

    cloned = None
    try:
        cloned = cv2.seamlessClone(inpainted, original, hard, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        try:
            cloned = cv2.seamlessClone(inpainted, original, hard, (cx, cy), cv2.MIXED_CLONE)
        except cv2.error:
            return None

    core = hard > 0
    lama_delta = float(
        np.abs(inpainted[core].astype(np.float32) - original[core].astype(np.float32)).mean()
    )
    clone_delta = float(
        np.abs(cloned[core].astype(np.float32) - original[core].astype(np.float32)).mean()
    )
    # Keep clone only when it preserves most of the fill departure from original.
    if lama_delta < 1.0 or clone_delta < lama_delta * 0.72:
        return None
    return cloned


def _match_fill_to_context(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Nudge washed LaMa fills toward surrounding mean color (coastal water/sky)."""
    import cv2

    hard = mask > 127
    if not hard.any():
        return inpainted

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return inpainted

    ring_mean = original[ring].astype(np.float32).mean(axis=0)
    fill_mean = inpainted[hard].astype(np.float32).mean(axis=0)
    delta = ring_mean - fill_mean
    drift = float(np.abs(delta).mean())
    if drift < 6.0:
        return inpainted

    # Stronger correction when LaMa drifts bright/grey on water.
    strength = float(np.clip(drift / 28.0, 0.45, 0.90))
    out = inpainted.astype(np.float32)
    out[hard] = np.clip(out[hard] + delta * strength, 0, 255)
    return out.astype(np.uint8)


def _inject_context_texture(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Add real water/sky ripple energy into flat LaMa fills."""
    import cv2

    hard = mask > 127
    if not hard.any():
        return inpainted

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return inpainted

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    ring_edge_density = float(cv2.Canny(gray, 40, 120)[ring].mean())
    if ring_edge_density < 8.0:
        return inpainted

    fill = inpainted.astype(np.float32)
    orig_f = original.astype(np.float32)
    sigma = 2.4
    blur_o = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=sigma)
    blur_f = cv2.GaussianBlur(fill, (0, 0), sigmaX=sigma)
    hf_o = orig_f - blur_o

    ring_lum = original[ring].astype(np.float32).mean(axis=1)
    water_floor = float(np.percentile(ring_lum, 20)) - 8.0
    lum = original.astype(np.float32).mean(axis=2)
    waterish = (~hard) & (lum >= water_floor)

    hf_fill = np.zeros_like(fill)
    h, _w = hard.shape
    for y in range(h):
        if not hard[y].any():
            continue
        water_xs = np.where(waterish[y])[0]
        src_y = y
        if len(water_xs) < 8:
            yy = y - 1
            while yy >= 0 and int(waterish[yy].sum()) < 8:
                yy -= 1
            if yy < 0:
                continue
            water_xs = np.where(waterish[yy])[0]
            src_y = yy
        mask_xs = np.where(hard[y])[0]
        for i, x in enumerate(mask_xs):
            src_x = int(water_xs[i % len(water_xs)])
            hf_fill[y, x] = hf_o[src_y, src_x]

    try:
        telea = inpaint_patch_opencv(original, mask).astype(np.float32)
        blur_t = cv2.GaussianBlur(telea, (0, 0), sigmaX=sigma)
        dist = cv2.distanceTransform((hard.astype(np.uint8)) * 255, cv2.DIST_L2, 5)
        edge_w = np.clip(1.0 - dist / 18.0, 0.0, 1.0)[:, :, None]
        hf_combined = hf_fill * (1.0 - edge_w) + (telea - blur_t) * edge_w
        blur_f = blur_f * 0.80 + blur_t * 0.20
    except Exception:
        hf_combined = hf_fill

    textured = blur_f + hf_combined
    out = fill.copy()
    out[hard] = np.clip(textured[hard], 0, 255)
    return out.astype(np.uint8)


def _all_people_binary(arr: np.ndarray, conf: float = 0.25) -> np.ndarray:
    """Binary mask of all YOLO-detected people in an RGB array."""
    import cv2

    h, w = arr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    try:
        model = _get_yolo_seg()
        results = model.predict(arr, classes=[0], conf=conf, verbose=False)
    except Exception:
        return combined
    if not results or results[0].masks is None or results[0].masks.data is None:
        return combined
    for mask_t in results[0].masks.data:
        mask = _mask_tensor_to_numpy(mask_t)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        combined = np.maximum(combined, (mask > 0.5).astype(np.uint8))
    return combined


def _is_water_or_sky_ring(original: np.ndarray, hard: np.ndarray) -> bool:
    """True when the ring around the hole looks like water/sky (not foliage/faces)."""
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 128:
        return False
    # Reuse strict water map (excludes people / white shirts).
    waterish = _water_like_pixels(original)
    return float(waterish[ring].mean()) > 0.28


def _scrub_other_people(rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    """Erase other people from the fill source so inpaint doesn't sample faces into the hole."""
    import cv2

    hard = hard_mask > 127
    people = _all_people_binary(rgb) > 0
    if not people.any():
        return rgb
    # Protect a halo around the target so we don't scrub the subject we're removing.
    protect = cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    others = people & (~protect)
    if int(others.sum()) < 64:
        return rgb
    # Grow slightly so hair/shoulders of bystanders don't leak into TELEA.
    others = cv2.dilate(others.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    others = others & (~hard)
    if not others.any():
        return rgb
    try:
        return inpaint_patch_opencv(rgb, (others.astype(np.uint8) * 255), radius=9)
    except Exception:
        return rgb


def _water_like_pixels(rgb: np.ndarray) -> np.ndarray:
    """Boolean map of water/sky-like pixels (exclude people and low-sat clothing)."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Require blue/cyan hue — low-sat alone matches white shirts and causes ghost copies.
    blue_water = (h >= 85) & (h <= 135) & (s >= 12) & (s <= 110) & (v > 95)
    # Bright haze/sky over water (still exclude pure white clothing via mild sat floor).
    bright_haze = (v > 170) & (s < 45) & (s >= 8) & (h >= 70) & (h <= 140)
    water = blue_water | bright_haze
    people = _all_people_binary(rgb) > 0
    if people.any():
        water = water & (~people)
    return water.astype(bool)


def _water_primary_fill(original: np.ndarray, lama: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill water/sky holes from real water pixels (no LaMa flat plaques / stretch smears).

    Strategy:
    1. Build a guide where non-water context is replaced with local water color
    2. TELEA/NS from that guide (cannot sample faces/grass)
    3. Reinforce with per-column nearest-water-above copy for ripple continuity
    """
    import cv2

    hard = _as_bool_mask(mask)
    if not hard.any():
        return lama

    h, w = original.shape[:2]
    water = _water_like_pixels(original) & (~hard)

    # Water-only guide so OpenCV never pulls hair/faces into the hole.
    guide = original.copy()
    if int(water.sum()) >= 256:
        mean_c = original[water].astype(np.float32).mean(axis=0)
        water_img = original.copy()
        water_img[~water] = mean_c
        water_blur = cv2.GaussianBlur(water_img, (0, 0), sigmaX=6.0)
        guide = original.copy()
        guide[~water & ~hard] = water_blur[~water & ~hard]

    try:
        telea = inpaint_patch_opencv(guide, (hard.astype(np.uint8) * 255), radius=16, method="telea")
        ns = inpaint_patch_opencv(guide, (hard.astype(np.uint8) * 255), radius=11, method="ns")
    except Exception:
        telea, ns = lama, lama

    # Per-column sample from water ABOVE the entire hole (not through the subject).
    # Sampling "nearest above each pixel" previously hit shirt pixels mislabeled as water.
    nearest = original.copy()
    if water.any():
        ys_h, xs_h = np.where(hard)
        y_top = int(ys_h.min())
        for x in np.unique(xs_h):
            x = int(x)
            col_hard = np.where(hard[:, x])[0]
            if len(col_hard) == 0:
                continue
            # Prefer water in this column above the hole top.
            wys = np.where(water[: max(1, y_top), x])[0]
            if len(wys) < 4:
                # Search neighboring columns.
                for dx in (-2, -1, 1, 2, -4, 4, -8, 8):
                    sx = x + dx
                    if 0 <= sx < w:
                        wys = np.where(water[: max(1, y_top), sx])[0]
                        if len(wys) >= 4:
                            break
            if len(wys) < 4:
                continue
            # Tile the water band above into the hard column (preserves ripple frequency).
            band = original[wys[0] : wys[-1] + 1, x].astype(np.float32)
            if band.shape[0] < 2:
                continue
            for i, y in enumerate(col_hard):
                src_i = i % band.shape[0]
                # Mild horizontal jitter from neighbor water columns for less striping.
                nearest[y, x] = band[src_i]
                for dx in (-1, 1):
                    sx = x + dx
                    if 0 <= sx < w and water[: max(1, y_top), sx].any():
                        wy2 = np.where(water[: max(1, y_top), sx])[0]
                        if len(wy2):
                            nearest[y, x] = (
                                nearest[y, x].astype(np.float32) * 0.7
                                + original[wy2[src_i % len(wy2)], sx].astype(np.float32) * 0.3
                            )
                            break
        nearest = np.clip(nearest, 0, 255).astype(np.uint8)

    # Score candidates against water ring.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    ring_water = ring & water if int((ring & water).sum()) >= 64 else ring
    if int(ring_water.sum()) >= 64:
        ring_mean = original[ring_water].astype(np.float32).mean(axis=0)

        def err(img: np.ndarray) -> float:
            return float(np.abs(img[hard].astype(np.float32).mean(axis=0) - ring_mean).mean())

        # Prefer textured water fills over flat LaMa.
        def texture(img: np.ndarray) -> float:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return float(np.abs(g[hard] - cv2.GaussianBlur(g, (0, 0), 2.0)[hard]).mean())

        ranked = sorted(
            [
                (err(telea) - 0.45 * texture(telea), telea),
                (err(ns) - 0.45 * texture(ns), ns),
                (err(nearest) - 0.65 * texture(nearest), nearest),
                # LaMa on CPU coastal holes is almost always a flat plaque — last resort only.
                (err(lama) + 8.0 - 0.05 * texture(lama), lama),
            ],
            key=lambda t: t[0],
        )
        best = ranked[0][1]
        second = ranked[1][1]
        base = (best.astype(np.float32) * 0.7 + second.astype(np.float32) * 0.3).astype(np.uint8)
    else:
        base = telea

    return _match_fill_to_context(original, base, (hard.astype(np.uint8) * 255))


def _hybrid_context_fill(
    original: np.ndarray,
    lama: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Choose water-aware fill or color-matched LaMa. No row-stamping (causes smears)."""
    hard = mask > 127
    if not hard.any():
        return lama

    if _is_water_or_sky_ring(original, hard):
        return _water_primary_fill(original, lama, mask)

    matched = _match_fill_to_context(original, lama, mask)
    scrubbed = _scrub_other_people(original, mask)
    try:
        telea = inpaint_patch_opencv(scrubbed, mask, radius=7)
        k_area = int(hard.sum())
        # On large holes, lean more on TELEA after scrubbing bystanders.
        mix = 0.55 if k_area > 8000 else 0.30
        return (
            matched.astype(np.float32) * (1.0 - mix) + telea.astype(np.float32) * mix
        ).astype(np.uint8)
    except Exception:
        return matched


def _crop_padding(bbox_w: int, bbox_h: int, img_h: int, img_w: int) -> int:
    """Context around the mask — enough for sky/water, but not a full-frame crop.

    Tall person masks previously expanded to nearly the whole image and hurt LaMa
    fill quality on coastal backgrounds. Cap pad so water ripples stay local.
    """
    long_side = max(img_h, img_w)
    short_side = min(bbox_w, bbox_h)
    pad = max(
        80,
        int(short_side * 0.40),
        int(short_side * 0.22 + max(bbox_w, bbox_h) * 0.05),
        int(long_side * 0.03),
    )
    return min(pad, 200)


def _residual_ghost_mask(
    original: np.ndarray,
    result: np.ndarray,
    hard_mask: np.ndarray,
    protect: np.ndarray | None = None,
) -> np.ndarray | None:
    """Detect leftover silhouette edges after a first inpaint pass."""
    import cv2

    hard = hard_mask > 127
    if not hard.any():
        return None

    diff = np.abs(result.astype(np.float32) - original.astype(np.float32)).mean(axis=2)
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    near = (cv2.dilate(hard.astype(np.uint8), band_k) > 0)

    # Pixels near the mask that barely changed — classic soft ghost / hair fringe.
    low_diff = (diff < 16.0) & near

    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    res_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    orig_edges = cv2.Canny(orig_gray, 50, 130)
    res_edges = cv2.Canny(res_gray, 50, 130)
    # Original person edges that still show in the result outline.
    persistent = (orig_edges > 0) & (res_edges > 0) & near

    residual = low_diff | persistent

    # Bright/washed blob vs surrounding water — only force full re-fill when severely off
    # and we already see edge residuals (avoid always doubling LaMa cost).
    ring = near & (~hard)
    if int(ring.sum()) >= 64 and int(residual.sum()) >= 200:
        ring_lum = float(orig_gray[ring].mean())
        fill_lum = float(res_gray[hard].mean())
        if abs(fill_lum - ring_lum) > 12.0:
            residual = residual | hard

    if int(residual.sum()) < 200:
        return None

    # Grow residual into a solid second-pass mask; keep it local to the first mask.
    res_u8 = residual.astype(np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    res_u8 = cv2.morphologyEx(res_u8, cv2.MORPH_CLOSE, close_k, iterations=2)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    res_u8 = cv2.dilate(res_u8, dilate_k, iterations=1)
    # Never expand far beyond the original selection halo.
    limit = cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)))
    res_u8 = res_u8 & limit
    # Union with original hard mask so the second pass re-fills the whole subject.
    res_u8 = np.maximum(res_u8, hard.astype(np.uint8))
    # Keep people-in-front cutouts from the first hard mask.
    res_u8 = _preserve_occlusion_holes(hard.astype(np.uint8), res_u8)
    if protect is not None and protect.any():
        res_u8 = res_u8 & (~protect.astype(np.uint8))
        res_u8 = np.maximum(res_u8, hard.astype(np.uint8) & (~protect.astype(np.uint8)))
    if int(res_u8.sum()) < int(hard.sum()) + 150:
        # Not enough new coverage to justify another LaMa pass.
        return None
    return res_u8 * 255


def _blend_feather_px(edge_softness: float) -> int:
    softness = max(0.5, min(float(edge_softness), 2.0))
    return max(8, int(round(12 * softness)))


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


def _ring_edge_density(original: np.ndarray, hard: np.ndarray) -> float:
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return 0.0
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    return float(cv2.Canny(gray, 40, 120)[ring].mean())


def _inpaint_once(
    rgb: np.ndarray,
    hard_mask: np.ndarray,
    mode: RemovalMode,
    edge_softness: float,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """Single crop → backend → context fill → composite pass."""
    h, w = rgb.shape[:2]
    feather_px = _blend_feather_px(edge_softness)
    # Tighter feather on water so we don't haze couple hair.
    if _is_water_or_sky_ring(rgb, hard_mask > 127):
        feather_px = max(4, int(round(feather_px * 0.55)))
    crop_guide = _build_blend_mask(hard_mask, feather_px)
    if protect is not None and protect.any():
        crop_guide = crop_guide.copy()
        crop_guide[protect] = 0.0
    ys, xs = np.where(crop_guide > 0.04)
    if len(xs) == 0:
        ys, xs = np.where(hard_mask > 127)
    if len(xs) == 0:
        return rgb

    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    pad = _crop_padding(bbox_w, bbox_h, h, w)
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(w, xs.max() + pad)
    y1 = min(h, ys.max() + pad)

    crop_rgb = rgb[y0:y1, x0:x1].copy()
    crop_hard = hard_mask[y0:y1, x0:x1].copy()
    crop_protect = protect[y0:y1, x0:x1] if protect is not None else None
    backend = get_backend(mode)
    # LaMa samples neighboring pixels — scrub bystanders so faces don't bleed into the hole.
    # Generative SD already respects the mask; TELEA-scrubbing the couple leaves muddy context
    # that the diffusion model copies as a flat ghost plaque.
    if getattr(backend, "is_generative", False):
        crop_for_inpaint = crop_rgb
    else:
        crop_for_inpaint = _scrub_other_people(crop_rgb, crop_hard)
    textured_bg = _ring_edge_density(crop_rgb, crop_hard > 127) > 14.0
    water_bg = _is_water_or_sky_ring(crop_rgb, crop_hard > 127)
    use_seamless = mode == "ai" and not textured_bg and not water_bg and not getattr(
        backend, "is_generative", False
    )

    try:
        inpainted_crop = backend.inpaint_crop(crop_for_inpaint, crop_hard)
    except InpaintError as exc:
        if mode == "ai" and "blank white fill" in str(exc).lower():
            try:
                from services.inpaint_backends import QuickRemoveBackend

                inpainted_crop = QuickRemoveBackend().inpaint_crop(crop_for_inpaint, crop_hard)
            except InpaintError:
                raise
        else:
            raise

    # Re-read after inpaint — AiRemoveBackend clears is_generative on LaMa fallback.
    generative = bool(getattr(backend, "is_generative", False))

    if mode == "ai":
        # Generative SD fill already synthesizes water/sky — do not overwrite with
        # TELEA/column-clone (that path exists to rescue classical LaMa plaques).
        if generative:
            inpainted_crop = _match_fill_to_context(crop_rgb, inpainted_crop, crop_hard)
        else:
            # Context fill / water clone must sample the real photo, not the scrubbed copy.
            inpainted_crop = _hybrid_context_fill(crop_rgb, inpainted_crop, crop_hard)

    seamless = _try_seamless_blend(crop_rgb, inpainted_crop, crop_hard) if use_seamless else None
    base_patch = seamless if seamless is not None else inpainted_crop

    # Generative: tighter hole + harder composite so feather doesn't reintroduce the person.
    blend_hard = crop_hard
    if generative:
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blend_hard = (cv2.dilate((crop_hard > 127).astype(np.uint8), k) * 255)
        feather_px = max(2, min(feather_px, 4))

    blend = _build_blend_mask(blend_hard, feather_px)
    if crop_protect is not None and crop_protect.any():
        import cv2

        blend = blend.copy()
        # No soft feather into protected people — hard cut keeps hair sharp.
        near_prot = (
            cv2.dilate(crop_protect.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1)))
            > 0
        )
        hard_b = blend_hard > 127
        blend[near_prot] = hard_b[near_prot].astype(np.float32)
        blend[crop_protect] = 0.0
    # Interior stays fully opaque; composite against the *original* crop (not scrubbed).
    patch = (
        base_patch.astype(np.float32) * blend[:, :, None]
        + crop_rgb.astype(np.float32) * (1.0 - blend[:, :, None])
    ).astype(np.uint8)
    if not generative:
        if crop_protect is None or not crop_protect.any():
            patch = _smooth_transition_band(patch, crop_rgb, blend)
        else:
            # Still smooth edges that are not near protected people.
            import cv2

            near_prot = (
                cv2.dilate(crop_protect.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                > 0
            )
            safe_blend = blend.copy()
            safe_blend[near_prot] = np.where(crop_hard[near_prot] > 127, 1.0, 0.0)
            patch = _smooth_transition_band(patch, crop_rgb, safe_blend)

    # Edge polish can reintroduce soft ghosts on rippled water — skip there.
    if mode == "ai" and not generative and not textured_bg and not water_bg:
        patch = _ai_edge_polish(patch, crop_rgb, crop_hard, blend)

    result = rgb.copy()
    result[y0:y1, x0:x1] = patch
    return result


def inpaint_objects(
    image: Image.Image,
    mask: Image.Image,
    mode: RemovalMode = "ai",
    edge_softness: float = 1.0,
) -> Image.Image:
    """Remove masked regions using AI Remove (SD2 generative) or Quick Remove (ONNX LaMa)."""
    rgb = _pil_to_rgb_array(image)

    msk = np.array(mask.convert("L"))
    if msk.max() == 0:
        return image.convert("RGB")

    raw = (msk > 127).astype(np.uint8)
    protect = _non_target_people_protect(rgb, raw) | _occlusion_cutout_protect(raw)

    hard_mask = refine_mask_hard(mask, edge_softness=edge_softness, rgb=rgb)
    hard_mask = _apply_people_protect(hard_mask, protect, raw=raw)

    # On water/sky, grow the hole a bit more so hair fringes don't leave a cutout plaque.
    if _is_water_or_sky_ring(rgb, hard_mask > 127):
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        grown = cv2.dilate((hard_mask > 127).astype(np.uint8), k, iterations=1)
        grown = _preserve_occlusion_holes(raw, grown)
        hard_mask = _apply_people_protect(grown * 255, protect, raw=raw)

    result = _inpaint_once(rgb, hard_mask, mode=mode, edge_softness=edge_softness, protect=protect)

    # Second pass: expand into leftover silhouette / hair ghost and re-fill.
    # Skip on water/sky — residual growth often re-damages couple hair without helping.
    if mode == "ai" and not _is_water_or_sky_ring(rgb, hard_mask > 127):
        residual = _residual_ghost_mask(rgb, result, hard_mask, protect=protect)
        if residual is not None:
            result = _inpaint_once(rgb, residual, mode=mode, edge_softness=edge_softness, protect=protect)

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
