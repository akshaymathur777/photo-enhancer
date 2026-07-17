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


def segment_from_points(
    image: Image.Image,
    points: list[list[float]],
    labels: list[int],
) -> Image.Image:
    """Generate a binary mask from click points using SAM 2.1 (+ YOLO person assist)."""
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

        # Full-person assist: SAM often returns head-only on selfie crowds.
        yolo_person = _yolo_person_mask_containing_click(arr, pt)
        if yolo_person is not None:
            combined = np.maximum(combined, yolo_person * 255)

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
    dilate_px: int = 24,
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
        dark_hair = band & (lum < bg_lum - 18.0) & (lum < float(dark_mean.mean()) + 12.0)
        expand = expand | dark_hair

    out = hard | expand
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out_u8 = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, close_k, iterations=1)
    return out_u8


def refine_mask(
    mask: Image.Image,
    dilate_px: int = 24,
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
    """Kill flat silhouette plaques by stamping neighboring water/sky texture into the hole."""
    import cv2

    hard = mask > 127
    if not hard.any():
        return inpainted

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return inpainted

    gray_ring = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    ring_edge_density = float(cv2.Canny(gray_ring, 40, 120)[ring].mean())
    if ring_edge_density < 8.0:
        return inpainted

    stamped = _stamp_context_texture(original, hard)
    if stamped is None:
        try:
            telea = inpaint_patch_opencv(original, mask)
        except Exception:
            return inpainted
        stamped = telea

    fill = inpainted.astype(np.float32)
    stamp = stamped.astype(np.float32)
    # Strong graft: keep a hint of LaMa/base structure, mostly stamped ripples.
    mix = 0.94 if ring_edge_density > 14.0 else 0.60
    sigma = 2.2
    blur_fill = cv2.GaussianBlur(fill, (0, 0), sigmaX=sigma)
    blur_stamp = cv2.GaussianBlur(stamp, (0, 0), sigmaX=sigma)
    textured = blur_fill * (1.0 - mix) + blur_stamp * mix + (stamp - blur_stamp)

    out = fill.copy()
    out[hard] = np.clip(textured[hard], 0, 255)
    return out.astype(np.uint8)


def _stamp_context_texture(original: np.ndarray, hard: np.ndarray) -> np.ndarray | None:
    """Fill the hole by mirroring same-row water/sky into each masked span.

    Only samples bright background-like pixels so dark hair from neighbors
    (couple in front) is not mirrored into the hole.
    """
    import cv2

    h, w = hard.shape
    if not hard.any():
        return None

    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(hard.astype(np.uint8), ring_k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return None

    ring_lum = original[ring].astype(np.float32).mean(axis=2)
    # Water/sky is bright; hair/clothes are darker — keep a margin below ring median.
    water_floor = float(np.percentile(ring_lum, 20)) - 8.0
    lum = original.astype(np.float32).mean(axis=2)
    waterish = (~hard) & (lum >= water_floor)

    out = original.copy()
    filled_any = False
    last_good = None

    for y in range(h):
        masked = hard[y]
        if not masked.any():
            if waterish[y].any():
                last_good = original[y].copy()
            continue

        row = masked.astype(np.uint8)
        edges = np.diff(row, prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        row_out = out[y].copy()
        row_filled = False
        water_row = waterish[y]

        for a, b in zip(starts, ends):
            b -= 1  # inclusive end
            left_run = 0
            x = a - 1
            while x >= 0 and water_row[x] and left_run < 400:
                left_run += 1
                x -= 1
            right_run = 0
            x = b + 1
            while x < w and water_row[x] and right_run < 400:
                right_run += 1
                x += 1

            for x in range(a, b + 1):
                dist_l = x - a
                dist_r = b - x
                pick = None
                if left_run > 0:
                    pick = a - 1 - (dist_l % left_run)
                if right_run > 0:
                    cand = b + 1 + (dist_r % right_run)
                    if pick is None or dist_r < dist_l:
                        pick = cand
                if pick is None:
                    free = np.where(water_row)[0]
                    if len(free):
                        pick = int(free[np.argmin(np.abs(free - x))])
                if pick is not None:
                    row_out[x] = original[y, pick]
                    row_filled = True

        if row_filled:
            out[y] = row_out
            last_good = row_out
            filled_any = True
        elif last_good is not None:
            out[y, masked] = last_good[masked]
            filled_any = True

    if not filled_any:
        return None

    ring_mean = original[ring].astype(np.float32).mean(axis=0)
    fill_mean = out[hard].astype(np.float32).mean(axis=0)
    delta = ring_mean - fill_mean
    if float(np.abs(delta).mean()) > 3.0:
        adj = out.astype(np.float32)
        adj[hard] = np.clip(adj[hard] + delta * 0.70, 0, 255)
        out = adj.astype(np.uint8)

    hard_u8 = hard.astype(np.uint8) * 255
    dist_in = cv2.distanceTransform(hard_u8, cv2.DIST_L2, 5)
    band = hard & (dist_in <= 5.0)
    if band.any():
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.8)
        mix = np.clip(dist_in / 5.0, 0.0, 1.0).astype(np.float32)
        blended = (
            blurred.astype(np.float32) * (1.0 - mix[:, :, None])
            + out.astype(np.float32) * mix[:, :, None]
        )
        out = out.copy()
        out[band] = np.clip(blended[band], 0, 255).astype(np.uint8)

    return out


def _hybrid_context_fill(
    original: np.ndarray,
    lama: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Color-match LaMa and lightly mix TELEA when it tracks context better.

    Aggressive same-row texture stamping was removed — it smeared neighboring
    faces/hills into horizontal glitch bands on crowded coastal selfies.
    """
    import cv2

    hard = mask > 127
    if not hard.any():
        return lama

    matched = _match_fill_to_context(original, lama, mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return matched

    ring_mean = original[ring].astype(np.float32).mean(axis=0)
    lama_err = float(np.abs(matched[hard].astype(np.float32).mean(axis=0) - ring_mean).mean())
    try:
        telea = inpaint_patch_opencv(original, mask)
        telea_err = float(np.abs(telea[hard].astype(np.float32).mean(axis=0) - ring_mean).mean())
    except Exception:
        return matched

    # Only mix TELEA when it clearly matches surrounding color better (no stamp).
    if telea_err + 4.0 < lama_err:
        mix = float(np.clip((lama_err - telea_err) / 30.0, 0.25, 0.65))
        return (
            matched.astype(np.float32) * (1.0 - mix) + telea.astype(np.float32) * mix
        ).astype(np.uint8)
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
) -> np.ndarray:
    """Single crop → backend → context fill → composite pass."""
    h, w = rgb.shape[:2]
    feather_px = _blend_feather_px(edge_softness)
    crop_guide = _build_blend_mask(hard_mask, feather_px)
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
    textured_bg = _ring_edge_density(crop_rgb, crop_hard > 127) > 14.0
    use_seamless = mode == "ai" and not textured_bg

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

    if mode == "ai":
        inpainted_crop = _hybrid_context_fill(crop_rgb, inpainted_crop, crop_hard)

    seamless = _try_seamless_blend(crop_rgb, inpainted_crop, crop_hard) if use_seamless else None
    base_patch = seamless if seamless is not None else inpainted_crop

    blend = _build_blend_mask(crop_hard, feather_px)
    # Interior stays fully opaque; no ghost bleed from original.
    patch = (
        base_patch.astype(np.float32) * blend[:, :, None]
        + crop_rgb.astype(np.float32) * (1.0 - blend[:, :, None])
    ).astype(np.uint8)
    patch = _smooth_transition_band(patch, crop_rgb, blend)

    # Edge polish can reintroduce soft ghosts on rippled water — skip there.
    if mode == "ai" and not textured_bg:
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
    """Remove masked regions using AI Remove (torch big-LaMa) or Quick Remove (ONNX LaMa)."""
    rgb = _pil_to_rgb_array(image)

    msk = np.array(mask.convert("L"))
    if msk.max() == 0:
        return image.convert("RGB")

    hard_mask = refine_mask_hard(mask, edge_softness=edge_softness, rgb=rgb)
    result = _inpaint_once(rgb, hard_mask, mode=mode, edge_softness=edge_softness)

    # Second pass: expand into leftover silhouette / hair ghost and re-fill.
    if mode == "ai":
        residual = _residual_ghost_mask(rgb, result, hard_mask)
        if residual is not None:
            result = _inpaint_once(rgb, residual, mode=mode, edge_softness=edge_softness)

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
