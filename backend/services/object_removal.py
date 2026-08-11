from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from services.inpaint_backends import (
    InpaintError,
    RemovalMode,
    generative_engine_ready,
    get_backend,
    get_removal_quality,
    inpaint_patch_opencv,
    set_removal_quality,
)
from services.scene_context import (
    classify_hole_scene,
    fill_for_scene,
    scene_is_open_fill,
)

_sam_predictor = None
_yolo_seg = None
_yolo_seg_name: str | None = None
_last_removal_meta: dict = {
    "engine": "unknown",
    "scene": "generic",
    "mode": "ai",
    "postfill": "none",
}
_GENERATIVE_ENGINES = frozenset({"sdxl", "powerpaint", "sd15"})
_people_mask_cache: dict[tuple, np.ndarray] = {}
_PEOPLE_MASK_CACHE_MAX = 8


def get_last_removal_meta() -> dict:
    """Engine/scene used by the most recent inpaint_objects call (for progress UI)."""
    return dict(_last_removal_meta)


def _is_generative_engine(engine: str | None) -> bool:
    return str(engine or "").lower() in _GENERATIVE_ENGINES


def warm_removal_models() -> dict:
    """Keep SAM + YOLO + inpaint engines resident for interactive remove."""
    from services.inpaint_backends import warm_inpaint_engines

    status: dict = {"sam": False, "yolo": False}
    try:
        _get_sam_predictor()
        status["sam"] = True
    except Exception as exc:
        status["sam_error"] = str(exc)
    try:
        _get_yolo_seg()
        status["yolo"] = True
    except Exception as exc:
        status["yolo_error"] = str(exc)
    status.update(warm_inpaint_engines())
    return status


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
            save=False,
            verbose=False,
        )
        _sam_predictor = SAM2Predictor(overrides=overrides)
    return _sam_predictor


def _resolve_yolo_seg_weight() -> str:
    """Prefer YOLOv8s-seg (better recall); fall back to nano if that's what's installed."""
    from services.model_download import _find_ultralytics_weight

    for name in ("yolov8s-seg.pt", "yolov8n-seg.pt"):
        if _find_ultralytics_weight(name) is not None:
            return name
    return "yolov8s-seg.pt"


def _get_yolo_seg():
    global _yolo_seg, _yolo_seg_name
    want = _resolve_yolo_seg_weight()
    if _yolo_seg is None or _yolo_seg_name != want:
        from ultralytics import YOLO

        _yolo_seg = YOLO(want)
        _yolo_seg_name = want
    return _yolo_seg


def _yolo_seg_predict(
    arr: np.ndarray,
    *,
    classes: list[int] | None = None,
    conf: float = 0.25,
    imgsz: int = 1280,
):
    """Shared YOLO-seg inference with a larger image size for small/distant subjects."""
    model = _get_yolo_seg()
    kwargs: dict = dict(conf=conf, imgsz=imgsz, verbose=False, retina_masks=True, save=False)
    if classes is not None:
        kwargs["classes"] = classes
    return model.predict(arr, **kwargs)


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
    guide_bbox: list[float] | None = None,
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

    guide = None
    if guide_bbox is not None and len(guide_bbox) == 4:
        gx0, gy0, gx1, gy1 = [int(round(v)) for v in guide_bbox]
        guide = np.zeros((h, w), dtype=bool)
        guide[max(0, gy0) : min(h, gy1 + 1), max(0, gx0) : min(w, gx1 + 1)] = True

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

        if guide is not None:
            binary = mask_i > 0.5
            inter = float((binary & guide).sum())
            union = float((binary | guide).sum()) + 1e-6
            iou = inter / union
            # Strongly prefer masks that stay inside the detector box.
            score += 0.55 * iou
            spill = float((binary & (~guide)).sum()) / max(float(binary.sum()), 1.0)
            score -= 0.40 * spill

        if score > best_score:
            best_score = score
            best_idx = i

    mask = _mask_tensor_to_numpy(data[best_idx])
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return mask


def _yolo_instance_at_click(
    arr: np.ndarray,
    click_pt: list[float] | tuple[float, float],
    *,
    conf: float | None = None,
    person_only: bool = False,
) -> dict | None:
    """Best YOLO-seg instance at/near the click: mask, bbox xyxy, class id, area.

    Prefers person (class 0) when several detections overlap the click — typical for
    removal of background people — but also returns non-person objects for smart select.
    """
    import cv2

    h, w = arr.shape[:2]
    cx, cy = int(round(float(click_pt[0]))), int(round(float(click_pt[1])))
    if not (0 <= cx < w and 0 <= cy < h):
        return None

    if conf is None:
        # Size-adaptive: small/distant subjects need lower conf on tall photos.
        long_side = max(h, w)
        conf = 0.10 if long_side >= 1400 else (0.12 if long_side >= 1000 else 0.18)

    classes = [0] if person_only else None
    try:
        results = _yolo_seg_predict(arr, classes=classes, conf=conf, imgsz=1280)
    except Exception:
        return None
    if not results:
        return None
    r = results[0]
    if r.masks is None or r.masks.data is None or r.boxes is None:
        return None

    candidates: list[dict] = []
    for i, mask_t in enumerate(r.masks.data):
        mask = _mask_tensor_to_numpy(mask_t)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = mask > 0.5
        try:
            xyxy = r.boxes.xyxy[i].detach().cpu().numpy().tolist()
            x0, y0, x1, y1 = [float(v) for v in xyxy]
            cls_id = int(r.boxes.cls[i].detach().cpu().item()) if r.boxes.cls is not None else -1
            conf_i = float(r.boxes.conf[i].detach().cpu().item()) if r.boxes.conf is not None else 0.5
        except Exception:
            continue

        in_mask = bool(binary[cy, cx]) if binary.any() else False
        in_box = x0 <= cx <= x1 and y0 <= cy <= y1
        if not (in_mask or in_box):
            continue
        area = int(binary.sum())
        if area < 120:
            continue
        # Prefer mask hits over box-only; prefer people; prefer tighter instances.
        score = conf_i
        score += 0.35 if in_mask else 0.0
        score += 0.25 if cls_id == 0 else 0.0
        score -= min(0.35, area / float(h * w))
        candidates.append(
            {
                "mask": binary.astype(np.uint8),
                "bbox": [x0, y0, x1, y1],
                "cls": cls_id,
                "area": area,
                "score": score,
                "conf": conf_i,
            }
        )

    if not candidates:
        # Retry once at a lower conf for small/background people.
        if conf > 0.08:
            return _yolo_instance_at_click(
                arr, click_pt, conf=max(0.06, conf - 0.06), person_only=person_only
            )
        return None
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


def _yolo_person_mask_containing_click(
    arr: np.ndarray,
    click_pt: list[float] | tuple[float, float],
    conf: float = 0.22,
) -> np.ndarray | None:
    """Return YOLO-seg mask for the person instance that contains the click, if any."""
    hit = _yolo_instance_at_click(arr, click_pt, conf=conf, person_only=True)
    if hit is None:
        return None
    return hit["mask"]


def _expand_bbox(xyxy: list[float], h: int, w: int, pad_frac: float = 0.06) -> list[float]:
    x0, y0, x1, y1 = xyxy
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    return [
        max(0.0, x0 - pad_x),
        max(0.0, y0 - pad_y),
        min(float(w - 1), x1 + pad_x),
        min(float(h - 1), y1 + pad_y),
    ]


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
    # Fat hair halo — coastal couple hair sits against water and gets bitten easily.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    protect = cv2.dilate(others.astype(np.uint8), k, iterations=1).astype(bool)
    target_keep = cv2.dilate(target, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
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
        results = _yolo_seg_predict(rgb, classes=[0], conf=0.22, imgsz=1280)
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
        # Only skip when this YOLO person IS the selection (high IoU). Mild edge
        # overlap with couple hair must still protect them (overlap ~0.1–0.25).
        if overlap / max(area, 1.0) >= 0.42 or overlap / max(raw_area, 1.0) >= 0.45:
            continue
        protect |= binary

    if not protect.any():
        return protect
    # Hair-safe halo: fat enough to stop YOLO expand biting couple hair on water.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    protect = cv2.dilate(protect.astype(np.uint8), k, iterations=1).astype(bool)
    # Only the eroded raw CORE blocks protect — full raw often spills onto couple hair.
    raw_core = cv2.erode(
        raw_b.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    raw_keep = cv2.dilate(raw_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    protect = protect & (~raw_keep)
    return protect


def _occlusion_cutout_protect(raw_mask: np.ndarray) -> np.ndarray:
    """Protect foreground cutouts that SAM already carved out of the selection.

    These holes (couple heads in front of the target) are the most reliable
    signal for what must never be painted — tighter than bloated YOLO masks.

    Critical: ignore small interior holes (eyes / sunglasses / shirt logos) —
    those must be filled, not protected, or the target face remains as a ghost.
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
    # Exterior ring of the selection — couple cutouts touch the outer boundary.
    outer = (
        cv2.dilate(raw_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    ) & (~raw_b)
    # Also require cutout to be relatively large (a head), not eye sockets.
    min_area = max(2500, int(raw_area * 0.04))

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > raw_area * 2.5:
            continue
        comp = labels == i
        if not (comp & touch).any():
            continue
        # Must touch the exterior (people in front), not be a deep interior void.
        if not (comp & outer).any():
            continue
        protect |= comp

    if not protect.any():
        return protect
    # Cover hair fringes around the cutout without reaching deep into the subject.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    protect = cv2.dilate(protect.astype(np.uint8), k, iterations=1).astype(bool)
    raw_core = cv2.erode(
        raw_b.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    raw_keep = cv2.dilate(raw_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    return protect & (~raw_keep)


def _apply_people_protect(hard_u8: np.ndarray, protect: np.ndarray, raw: np.ndarray | None = None) -> np.ndarray:
    """Remove protected people from a hard mask; keep only a tight raw target core.

    Previously unioning full raw back undid protect whenever SAM spilled onto the
    couple (hair/glasses bites). Keep an eroded raw core so the clicked subject
    cannot be hollowed out, but let protect win on couple hair.
    """
    import cv2

    hard_b = _as_bool_mask(hard_u8)
    if protect is None or not protect.any():
        if raw is not None:
            return ((hard_b | _as_bool_mask(raw)).astype(np.uint8) * 255)
        return hard_u8
    out = hard_b & (~protect)
    if raw is not None:
        raw_b = _as_bool_mask(raw)
        # Tight core of the selection — survives protect. Hair spill does not.
        core = cv2.erode(
            raw_b.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        ).astype(bool)
        if not core.any():
            core = cv2.erode(
                raw_b.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ).astype(bool)
        # Raw outside protect is fine; raw under protect only via core.
        out = out | (raw_b & (~protect)) | core
    return out.astype(np.uint8) * 255


def segment_from_points(
    image: Image.Image,
    points: list[list[float]],
    labels: list[int],
) -> Image.Image:
    """Generate a binary mask from click points using SAM 2.1 (+ YOLO box assist)."""
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
        # Prefer person instances for removal clicks (background bystanders).
        hit = _yolo_instance_at_click(arr, pt, person_only=True)
        if hit is None:
            hit = _yolo_instance_at_click(arr, pt, person_only=False)
        guide_bbox = None
        pts = [pt] + auto_neg
        lbls = [1] + [0] * len(auto_neg)

        # Box+point prompts are much more reliable than points alone (full body / object).
        if hit is not None:
            guide_bbox = _expand_bbox(hit["bbox"], h, w, pad_frac=0.08)
            try:
                results = predictor(
                    bboxes=[guide_bbox],
                    points=pts,
                    labels=lbls,
                    multimask_output=True,
                )
            except TypeError:
                results = predictor(points=pts, labels=lbls, multimask_output=True)
            except Exception:
                results = predictor(points=pts, labels=lbls, multimask_output=True)
        else:
            results = predictor(points=pts, labels=lbls, multimask_output=True)

        mask = _best_sam_mask_from_results(results, h, w, click_pt=pt, guide_bbox=guide_bbox)
        if mask is not None:
            combined = np.maximum(combined, (mask > 0.5).astype(np.uint8) * 255)

        # Detector assist: YOLO full instance, then cut away protected people.
        # Critical: for people, trust the FULL YOLO body — SAM often stops at the head
        # on coastal third-person shots, and a 51px near-clip drops the striped torso.
        if hit is not None:
            yolo_vis = hit["mask"].astype(bool) & (~protect)
            if hit.get("cls", -1) == 0:
                # Person: union YOLO body, but don't spill into rocks/grass far below SAM.
                x0, y0, x1, y1 = [int(round(v)) for v in hit["bbox"]]
                pad = int(max(24, 0.12 * max(x1 - x0, y1 - y0)))
                gate = np.zeros((h, w), dtype=bool)
                gate[
                    max(0, y0 - pad) : min(h, y1 + pad + 1),
                    max(0, x0 - pad) : min(w, x1 + pad + 1),
                ] = True
                yolo_vis = yolo_vis & gate
                sam_b = combined > 127
                if sam_b.any():
                    import cv2 as _cv2

                    ys_s, _xs_s = np.where(sam_b)
                    sam_bot = int(ys_s.max())
                    # Allow YOLO only modestly below SAM (shirt/shoulders), not feet on rocks.
                    y_limit = min(h - 1, sam_bot + max(48, int((sam_bot - int(ys_s.min())) * 0.55)))
                    yolo_vis[y_limit + 1 :, :] = False
                    # Prefer keeping YOLO where SAM already is OR background looks like water/sky.
                    hsv = _cv2.cvtColor(arr, _cv2.COLOR_RGB2HSV).astype(np.float32)
                    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                    waterish = ((hh >= 85) & (hh <= 135) & (ss >= 12) & (vv > 95)) | (
                        (ss < 48) & (vv > 125) & (vv < 235)
                    )
                    near_sam = (
                        _cv2.dilate(
                            sam_b.astype(np.uint8),
                            _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (71, 71)),
                        )
                        > 0
                    )
                    yolo_vis = yolo_vis & (near_sam | waterish)
            else:
                near = combined > 127
                if near.any():
                    near = (
                        cv2.dilate(
                            near.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)),
                        )
                        > 0
                    )
                    yolo_vis = yolo_vis & near
            # Re-cut protect with a stronger hair halo so couple isn't eaten by YOLO spill.
            # Also drop YOLO body-assist near protect (IoU / proximity) — hair bite source.
            if protect.any():
                prot_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
                prot_fat = cv2.dilate(protect.astype(np.uint8), prot_k, iterations=1).astype(bool)
                yolo_vis = yolo_vis & (~prot_fat)
                # If YOLO body overlaps protect significantly, shrink assist to SAM-only zone.
                yolo_area = float(yolo_vis.sum()) + 1e-3
                prot_overlap = float((yolo_vis & prot_fat).sum())
                # After cut, if remaining YOLO is mostly near protect edge, prefer SAM.
                near_prot = (
                    cv2.dilate(protect.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)))
                    > 0
                )
                near_frac = float((yolo_vis & near_prot).sum()) / yolo_area
                if near_frac > 0.55:
                    sam_core = (
                        cv2.erode(
                            (combined > 127).astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        )
                        > 0
                    )
                    # Keep YOLO only where SAM already covers (no body expand into hair).
                    yolo_vis = yolo_vis & (combined > 127)
                    _ = prot_overlap  # measured for future tuning
                    _ = sam_core
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

    # Fill small holes inside the selection (better body coverage).
    if combined.any():
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        filled = cv2.morphologyEx((combined > 127).astype(np.uint8), cv2.MORPH_CLOSE, close_k, iterations=2)
        # Flood-fill exterior, then invert — fills interior voids (eyes/sunglasses).
        h2, w2 = filled.shape
        flood = filled.copy()
        ff_mask = np.zeros((h2 + 2, w2 + 2), dtype=np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 1)
        interior = (flood == 0) & (filled == 0)
        # Only fill interior voids that are clearly inside the subject (not couple cutouts).
        if interior.any():
            n_lab, lab, st, _ = cv2.connectedComponentsWithStats(interior.astype(np.uint8), 8)
            area_subj = float(filled.sum())
            for i in range(1, n_lab):
                a = int(st[i, cv2.CC_STAT_AREA])
                # Fill eye/logo voids; leave large exterior-touching cutouts alone.
                if a < max(80, int(area_subj * 0.08)):
                    filled[lab == i] = 1
        if protect.any():
            filled[protect] = 0
        combined = filled * 255

    return Image.fromarray(combined, mode="L")


def auto_detect_people_mask(image: Image.Image, conf: float = 0.18) -> Image.Image:
    """Build a combined mask for all detected people (YOLOv8s/n-seg, COCO class 0)."""
    import cv2

    arr = _pil_to_rgb_array(image)
    h, w = arr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    img_area = float(h * w)

    def _add_masks(results, *, min_area: float, max_area_frac: float, iou_skip: float) -> int:
        nonlocal combined
        added = 0
        if not results or results[0].masks is None or results[0].masks.data is None:
            return 0
        for mask_t in results[0].masks.data:
            mask = _mask_tensor_to_numpy(mask_t)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (mask > 0.5).astype(np.uint8)
            area = float(binary.sum())
            if area < min_area or area > max_area_frac * img_area:
                continue
            if combined.any():
                overlap = float(((combined > 127) & (binary > 0)).sum())
                if overlap / max(area, 1.0) > iou_skip:
                    # Keep the non-overlapping remainder (distant person partially
                    # covered by a bloated couple detection).
                    unique = binary & (~(combined > 127).astype(np.uint8))
                    unique_area = float(unique.sum())
                    if unique_area >= max(min_area, area * 0.18):
                        combined = np.maximum(combined, unique * 255)
                        added += 1
                    continue
            combined = np.maximum(combined, binary * 255)
            added += 1
        return added

    try:
        results = _yolo_seg_predict(arr, classes=[0], conf=conf, imgsz=1280)
    except Exception:
        return Image.fromarray(combined, mode="L")
    _add_masks(results, min_area=150, max_area_frac=0.55, iou_skip=0.55)

    # Low-confidence pass catches distant / partial people (e.g. coastal third person).
    try:
        results_lo = _yolo_seg_predict(arr, classes=[0], conf=max(0.06, conf - 0.10), imgsz=1280)
        _add_masks(results_lo, min_area=250, max_area_frac=0.40, iou_skip=0.35)
    except Exception:
        pass

    # On tall photos, also scan the upper band — background people often sit above the couple.
    # Eval often resizes to ~960h; old h>=1200 gate skipped library far crowd entirely.
    if h >= 480:
        y1 = int(h * 0.78)
        band = arr[:y1, :]
        try:
            results_band = _yolo_seg_predict(band, classes=[0], conf=max(0.06, conf - 0.10), imgsz=1280)
        except Exception:
            results_band = None
        if results_band and results_band[0].masks is not None and results_band[0].masks.data is not None:
            for mask_t in results_band[0].masks.data:
                mask = _mask_tensor_to_numpy(mask_t)
                if mask.shape[:2] != (y1, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, y1), interpolation=cv2.INTER_LINEAR)
                full = np.zeros((h, w), dtype=np.uint8)
                full[:y1, :] = (mask > 0.5).astype(np.uint8)
                area = float(full.sum())
                if area < 120 or area > 0.40 * img_area:
                    continue
                if combined.any():
                    overlap = float(((combined > 127) & (full > 0)).sum())
                    if overlap / max(area, 1.0) > 0.35:
                        continue
                combined = np.maximum(combined, full * 255)

    if combined.any():
        # Close small gaps (arms, torso holes) without exploding into background.
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx((combined > 127).astype(np.uint8), cv2.MORPH_CLOSE, close_k, iterations=2)
        # Mild hair halo.
        halo = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        combined = halo * 255

    return Image.fromarray(combined, mode="L")


def _list_person_instances(
    arr: np.ndarray,
    conf: float | None = None,
) -> list[np.ndarray]:
    """Return binary person masks (bool HxW), largest first.

    Multi-scale + upper-band passes catch mid/far aisle crowds that full-frame
    YOLO often misses at eval resolutions (~720–1280 long side).
    """
    import cv2

    h, w = arr.shape[:2]
    if conf is None:
        long_side = max(h, w)
        conf = 0.10 if long_side >= 1200 else 0.12
    instances: list[tuple[float, np.ndarray]] = []

    def _ingest(results, *, min_area: float, max_area_frac: float, iou_skip: float) -> None:
        if not results or results[0].masks is None or results[0].masks.data is None:
            return
        existing = [m for _a, m in instances]
        for mask_t in results[0].masks.data:
            mask = _mask_tensor_to_numpy(mask_t)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            binary = mask > 0.5
            area = float(binary.sum())
            if area < min_area or area > max_area_frac * h * w:
                continue
            dup = False
            for em in existing:
                inter = float((binary & em).sum())
                if inter / max(area, 1.0) > iou_skip:
                    # Keep non-overlapping remainder (partial far person).
                    unique = binary & (~em)
                    unique_area = float(unique.sum())
                    if unique_area >= max(min_area * 0.7, area * 0.20):
                        instances.append((unique_area, unique))
                        existing.append(unique)
                    dup = True
                    break
            if dup:
                continue
            instances.append((area, binary))
            existing.append(binary)

    try:
        results = _yolo_seg_predict(arr, classes=[0], conf=conf, imgsz=1280)
    except Exception:
        return []
    _ingest(results, min_area=100, max_area_frac=0.55, iou_skip=0.45)

    # Low-conf pass for distant aisle people (library globe crowd).
    try:
        results_lo = _yolo_seg_predict(arr, classes=[0], conf=max(0.04, conf - 0.07), imgsz=1536)
        _ingest(results_lo, min_area=70, max_area_frac=0.45, iou_skip=0.38)
    except Exception:
        pass

    # Upper-third / mid-hall scan — far crowds sit under ceilings/globes.
    # Run whenever height is usable (eval often resizes to ~960).
    if h >= 480:
        y1 = int(h * 0.72)  # upper ~2/3; far crowd under globe is mid-upper
        band = arr[:y1, :]
        try:
            results_band = _yolo_seg_predict(
                band, classes=[0], conf=max(0.04, conf - 0.08), imgsz=1536
            )
        except Exception:
            results_band = None
        if results_band and results_band[0].masks is not None and results_band[0].masks.data is not None:
            existing = [m for _a, m in instances]
            for mask_t in results_band[0].masks.data:
                mask = _mask_tensor_to_numpy(mask_t)
                if mask.shape[:2] != (y1, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, y1), interpolation=cv2.INTER_LINEAR)
                full = np.zeros((h, w), dtype=bool)
                full[:y1, :] = mask > 0.5
                area = float(full.sum())
                if area < 60 or area > 0.40 * h * w:
                    continue
                dup = False
                for em in existing:
                    inter = float((full & em).sum())
                    if inter / max(area, 1.0) > 0.38:
                        # Keep non-overlapping remainder for packed crowds.
                        unique = full & (~em)
                        if float(unique.sum()) >= max(50.0, area * 0.22):
                            instances.append((float(unique.sum()), unique))
                            existing.append(unique)
                        dup = True
                        break
                if dup:
                    continue
                instances.append((area, full))
                existing.append(full)

    # Zoomed center strip for packed far crowds (library under globe).
    if h >= 480 and w >= 400:
        cy0, cy1 = int(h * 0.22), int(h * 0.72)
        cx0, cx1 = int(w * 0.15), int(w * 0.85)
        crop = arr[cy0:cy1, cx0:cx1]
        try:
            results_z = _yolo_seg_predict(
                crop, classes=[0], conf=max(0.03, conf - 0.09), imgsz=1280
            )
        except Exception:
            results_z = None
        if results_z and results_z[0].masks is not None and results_z[0].masks.data is not None:
            existing = [m for _a, m in instances]
            ch, cw = cy1 - cy0, cx1 - cx0
            for mask_t in results_z[0].masks.data:
                mask = _mask_tensor_to_numpy(mask_t)
                if mask.shape[:2] != (ch, cw):
                    mask = cv2.resize(mask.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
                full = np.zeros((h, w), dtype=bool)
                full[cy0:cy1, cx0:cx1] = mask > 0.5
                area = float(full.sum())
                # Far people are small — allow tiny instances.
                if area < 50 or area > 0.30 * h * w:
                    continue
                # Prefer detections whose centroid sits in the upper half (crowd depth).
                ys, xs = np.where(full)
                if len(ys) and float(ys.mean()) > h * 0.82:
                    continue
                dup = False
                for em in existing:
                    inter = float((full & em).sum())
                    if inter / max(area, 1.0) > 0.32:
                        unique = full & (~em)
                        if float(unique.sum()) >= max(40.0, area * 0.20):
                            instances.append((float(unique.sum()), unique))
                            existing.append(unique)
                        dup = True
                        break
                if dup:
                    continue
                instances.append((area, full))
                existing.append(full)

    instances.sort(key=lambda t: t[0], reverse=True)
    return [m for _a, m in instances]


def auto_detect_others_mask(
    image: Image.Image,
    *,
    lock_point: tuple[float, float] | None = None,
    conf: float | None = None,
) -> Image.Image:
    """Mask of people to remove — everyone except the locked / largest subject.

    One-person-at-a-time workflow: never one mega-mask for the whole aisle.
    When ``lock_point`` is set, keep the person instance under that click;
    otherwise keep the largest foreground person.
    """
    import cv2

    arr = _pil_to_rgb_array(image)
    h, w = arr.shape[:2]
    instances = _list_person_instances(arr, conf=conf)
    if len(instances) < 2:
        # Fall back: all people minus nothing useful — return empty if only one.
        if len(instances) == 1:
            return Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")
        return auto_detect_people_mask(image, conf=conf if conf is not None else 0.14)

    keep_idx = 0
    if lock_point is not None:
        lx, ly = int(round(lock_point[0])), int(round(lock_point[1]))
        lx = max(0, min(w - 1, lx))
        ly = max(0, min(h - 1, ly))
        best_i, best_area = 0, -1.0
        for i, m in enumerate(instances):
            if m[ly, lx]:
                area = float(m.sum())
                if area > best_area:
                    best_area = area
                    best_i = i
        if best_area > 0:
            keep_idx = best_i

    keep = instances[keep_idx]
    others = np.zeros((h, w), dtype=bool)
    for i, m in enumerate(instances):
        if i == keep_idx:
            continue
        others |= m
    # Mild halo for hair; never eat into the locked subject.
    if others.any():
        halo = cv2.dilate(others.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
        keep_dil = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
        others = (halo > 0) & (keep_dil == 0)
    return Image.fromarray((others.astype(np.uint8) * 255), mode="L")


def lock_subject_protect_mask(
    image: Image.Image,
    lock_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Binary protect mask for the locked / largest person (hero subject)."""
    arr = _pil_to_rgb_array(image)
    h, w = arr.shape[:2]
    instances = _list_person_instances(arr, conf=0.12)
    if not instances:
        return np.zeros((h, w), dtype=bool)
    keep_idx = 0
    if lock_point is not None:
        lx, ly = int(round(lock_point[0])), int(round(lock_point[1]))
        lx = max(0, min(w - 1, lx))
        ly = max(0, min(h - 1, ly))
        for i, m in enumerate(instances):
            if m[ly, lx]:
                keep_idx = i
                break
    keep = instances[keep_idx]
    import cv2

    return cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1).astype(bool)


def refine_mask_hard(
    mask: Image.Image,
    dilate_px: int = 14,
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
        # Smaller halo — large dark-hair expand eats couple hair against water.
        binary = _expand_mask_subject_halo(rgb, binary, max_radius=8, color_thresh=28.0)

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
    outer = (
        cv2.dilate(raw_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    ) & (~raw_b)
    min_area = max(2500, int(raw_area * 0.04))

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Skip tiny flecks, eye sockets, and the huge background component.
        if area < min_area:
            continue
        if area > raw_area * 2.5:
            continue
        comp = labels == i
        if not (comp & touch).any():
            continue
        if not (comp & outer).any():
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
    """Nudge washed fills toward surrounding mean color (prefer water above on coasts)."""
    import cv2

    hard = mask > 127
    if not hard.any():
        return inpainted

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return inpainted

    # Use the same conservative water map as fill_for_scene (excludes sky glare).
    # Never treat indoor wood/shelf rings as water — cool ceiling light fools the map.
    from services.scene_context import _water_source_map, classify_hole_scene

    hole_scene = classify_hole_scene(original, hard)
    water = _water_source_map(original, hard)
    ys, xs = np.where(hard)
    y_top = int(ys.min())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y_top - max(64, int((ys.max() - y_top + 1) * 0.7)))
    band = np.zeros_like(hard, dtype=bool)
    band[y0:y_top, max(0, x0 - 24) : min(hard.shape[1], x1 + 25)] = True
    above_water = band & water
    allow_water_ref = hole_scene in ("water", "sky") or (
        hole_scene != "structured" and int(above_water.sum()) >= 400
    )

    if allow_water_ref and int(above_water.sum()) >= 64:
        ref = above_water
        # Prefer brighter sparkle water for the target mean.
        gv = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bright = ref & (gv >= float(np.percentile(gv[ref], 40)))
        if int(bright.sum()) >= 64:
            ref = bright
    elif allow_water_ref and int((ring & water).sum()) >= 64:
        ref = ring & water
    else:
        ref = ring
    scene_water = allow_water_ref and (
        int(above_water.sum()) >= 64 or int((ring & water).sum()) >= 64
    )

    g_ref = float(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)[ref].mean())
    g_fill = float(cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)[hard].mean())
    ring_mean = original[ref].astype(np.float32).mean(axis=0)

    out = inpainted.astype(np.float32)
    # Only correct over-bright fills (white plaques). Never brighten toward sky glare.
    if g_fill > g_ref + 4.0:
        scale = g_ref / max(g_fill, 1.0)
        out[hard] = np.clip(out[hard] * scale, 0, 255)
        chroma_delta = ring_mean - out[hard].mean(axis=0)
        out[hard] = np.clip(out[hard] + chroma_delta * 0.30, 0, 255)
        g_out = float(
            (0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2])[hard].mean()
        )
        if g_out > g_ref + 2.0:
            out[hard] *= g_ref / max(g_out, 1.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Brighten dark plaques / match drifted water fills toward context.
    if abs(g_fill - g_ref) > 4.0 or (scene_water and g_fill < g_ref - 3.0):
        delta = ring_mean - out[hard].mean(axis=0)
        # Water: aggressive lift toward bright bay — dark fills read as grey silhouettes.
        strength = 1.0 if scene_water else 0.40
        out[hard] = np.clip(out[hard] + delta * strength, 0, 255)
        g_out = float(
            (0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2])[hard].mean()
        )
        if scene_water and g_out < g_ref - 4.0:
            out[hard] *= float(np.clip(g_ref / max(g_out, 1.0), 1.0, 1.28))
        if g_out > g_ref + 3.0:
            out[hard] *= g_ref / max(g_out, 1.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    return inpainted


def _inject_context_texture(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    scene: str | None = None,
) -> np.ndarray:
    """Scene-authoritative fill: PatchMatch for water/structured, ring HF otherwise."""
    return fill_for_scene(original, inpainted, mask, scene=scene)  # type: ignore[arg-type]


def _all_people_binary(arr: np.ndarray, conf: float = 0.22) -> np.ndarray:
    """Binary mask of all YOLO-detected people in an RGB array."""
    import cv2

    h, w = arr.shape[:2]
    # Keyed on pixel content rather than object identity: a single removal detects
    # on the same photo several times, but numpy reuses buffers across images, so
    # an id()-based key would hand back another photo's people.
    key = (
        hashlib.blake2b(arr.tobytes(), digest_size=16).digest(),
        arr.shape,
        arr.dtype.str,
        float(conf),
    )
    cached = _people_mask_cache.get(key)
    if cached is not None:
        return cached.copy()
    combined = np.zeros((h, w), dtype=np.uint8)
    try:
        results = _yolo_seg_predict(arr, classes=[0], conf=conf, imgsz=1280)
    except Exception:
        return combined
    if results and results[0].masks is not None and results[0].masks.data is not None:
        for mask_t in results[0].masks.data:
            mask = _mask_tensor_to_numpy(mask_t)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            combined = np.maximum(combined, (mask > 0.5).astype(np.uint8))
    if len(_people_mask_cache) >= _PEOPLE_MASK_CACHE_MAX:
        _people_mask_cache.clear()
    _people_mask_cache[key] = combined
    return combined.copy()


def _is_water_or_sky_ring(original: np.ndarray, hard: np.ndarray) -> bool:
    """True when water/sky sits around or above the hole (coastal third-person cases).

    Ring-only checks fail when couple hair/rocks dominate the immediate halo while
    the real background to synthesize is water above the subject.
    """
    import cv2

    if not hard.any():
        return False
    waterish = _water_like_pixels(original)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) >= 128 and float(waterish[ring].mean()) > 0.22:
        return True

    # Band above the hole — typical selfie with person standing in front of bay.
    ys, xs = np.where(hard)
    y_top = int(ys.min())
    x0, x1 = int(xs.min()), int(xs.max())
    h, _w = hard.shape
    y0 = max(0, y_top - max(48, int((ys.max() - y_top + 1) * 0.55)))
    band = np.zeros_like(hard, dtype=bool)
    band[y0:y_top, x0 : x1 + 1] = True
    band &= ~hard
    if int(band.sum()) < 128:
        return False
    return float(waterish[band].mean()) > 0.30


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
    """Boolean map of water/sky-like pixels (exclude people and low-sat clothing).

    Overcast coastal water is often grey/tan (low sat, mixed hue) — not pure blue.
    """
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Classic blue/cyan open water.
    blue_water = (h >= 85) & (h <= 135) & (s >= 12) & (s <= 110) & (v > 95)
    # Bright haze/sky over water.
    bright_haze = (v > 170) & (s < 45) & (s >= 8) & (h >= 70) & (h <= 140)
    # Grey/sunlit coastal water (overcast ripples): low sat, mid-high value.
    # Exclude high-sat rocks/grass (s>=50) and near-white clothing (v>=235 & s<8).
    grey_coastal = (
        (s >= 3)
        & (s < 48)
        & (v > 125)
        & (v < 235)
        & ~((h >= 35) & (h <= 85) & (s >= 40))  # green vegetation
    )
    water = blue_water | bright_haze | grey_coastal
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


def _fill_texture_energy(img: np.ndarray, hard: np.ndarray) -> float:
    """High-frequency energy inside the hole (ripples score high; soft plaques score low)."""
    import cv2

    if not hard.any():
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(g[hard] - cv2.GaussianBlur(g, (0, 0), 2.0)[hard]).mean())


def _is_soft_water_plaque(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> bool:
    """True when generative/classical fill is flatter than surrounding water."""
    import cv2

    hard = mask > 127
    if int(hard.sum()) < 200:
        return False
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    water = _water_like_pixels(original)
    ring_w = ring & water
    if int(ring_w.sum()) < 64:
        # Fall back to upward water band when halo is couple hair/rocks.
        ys, xs = np.where(hard)
        y_top = int(ys.min())
        x0, x1 = int(xs.min()), int(xs.max())
        y0 = max(0, y_top - 80)
        band = np.zeros_like(hard, dtype=bool)
        band[y0:y_top, x0 : x1 + 1] = True
        ring_w = band & water & (~hard)
    if int(ring_w.sum()) < 64:
        return False
    fill_tex = _fill_texture_energy(filled, hard)
    ref_tex = _fill_texture_energy(original, ring_w)
    if ref_tex < 2.0:
        return False
    # Soft plaque: fill much flatter than real water, or washed luminance.
    gray_o = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_f = cv2.cvtColor(filled, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum_delta = abs(float(gray_f[hard].mean()) - float(gray_o[ring_w].mean()))
    # Aggressive: coastal plaques often sit at ~0.3–0.5× ring HF.
    severe_flat = fill_tex < ref_tex * 0.72 or fill_tex < 5.5
    washed = lum_delta > 8.0 and fill_tex < ref_tex * 0.90
    return bool(severe_flat or washed)


def _rescue_soft_generative_water(
    original: np.ndarray,
    generative_fill: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """When SD leaves a soft water plaque, rebuild ripple energy from the photo.

    Prefer exemplar/PatchMatch water texture. Keep almost no generative plaque shape.
    """
    if not _is_water_or_sky_ring(original, mask > 127):
        return _match_fill_to_context(original, generative_fill, mask)

    # Dense PatchMatch from real water — authoritative for coastal plaques.
    rescued = fill_for_scene(original, generative_fill, mask, scene="water")
    rescued = _match_fill_to_context(original, rescued, mask)
    if not _is_soft_water_plaque(original, rescued, mask):
        return rescued

    # TELEA/NS blob → full exemplar replace (no soft plaque blend).
    from services.scene_context import patchmatch_fill, _inject_water_laplacian_hf, _wide_water_source_ring, _water_source_map

    hard = mask > 127
    water_src = _water_source_map(original, hard)
    ring = _wide_water_source_ring(original, hard, water_src)
    water = patchmatch_fill(original, hard, patch=28, searches=8, dense=True)
    water = _inject_water_laplacian_hf(
        original, water, hard, ring if int(ring.sum()) >= 64 else water_src, target_hf=12.0
    )
    water = fill_for_scene(original, water, mask, scene="water")
    water = _match_fill_to_context(original, water, mask)
    # Hard replace — keep almost none of the soft plaque.
    return water


def _soft_plaque_residual_mask(
    original: np.ndarray,
    result: np.ndarray,
    hard_mask: np.ndarray,
    protect: np.ndarray | None = None,
) -> np.ndarray | None:
    """Second-pass mask for leftover soft water plaques after generative fill."""
    import cv2

    hard = hard_mask > 127
    if not hard.any() or not _is_soft_water_plaque(original, result, hard_mask):
        return None

    # Re-fill the whole subject hole (plus a mild hair halo), not just edges.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    res_u8 = cv2.dilate(hard.astype(np.uint8), k, iterations=1)
    res_u8 = _preserve_occlusion_holes(hard.astype(np.uint8), res_u8)
    if protect is not None and protect.any():
        res_u8 = res_u8 & (~protect.astype(np.uint8))
        res_u8 = np.maximum(res_u8, hard.astype(np.uint8) & (~protect.astype(np.uint8)))
    if int(res_u8.sum()) < 200:
        return None
    return res_u8 * 255


def _match_edge_band_luminance(
    original: np.ndarray,
    result: np.ndarray,
    hard_mask: np.ndarray,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """Match luminance of the fill's outer band to the surrounding ring (harder seams)."""
    import cv2

    hard = hard_mask > 127
    if not hard.any():
        return result
    edge = (
        cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    ) & (
        ~(cv2.erode(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0)
    )
    ring = (
        cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    ) & (~hard)
    if protect is not None and protect.any():
        edge = edge & (~protect)
        ring = ring & (~protect)
    if int(edge.sum()) < 64 or int(ring.sum()) < 64:
        return result
    g_o = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g_r = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
    delta = float(g_o[ring].mean() - g_r[edge].mean())
    if abs(delta) < 2.5:
        return result
    out = result.astype(np.float32)
    # Apply mostly on the edge band so interior texture stays.
    strength = float(np.clip(abs(delta) / 18.0, 0.35, 0.85))
    out[edge] = np.clip(out[edge] + delta * strength, 0, 255)
    return out.astype(np.uint8)


def _split_mask_components(
    hard_u8: np.ndarray,
    *,
    min_area: int = 400,
    max_parts: int = 12,
) -> list[np.ndarray]:
    """Split a removal mask into connected components (per-person holes)."""
    import cv2

    hard = hard_u8 > 127
    if not hard.any():
        return []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), connectivity=8)
    parts: list[tuple[int, np.ndarray]] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        parts.append((area, (labels == i).astype(np.uint8) * 255))
    if not parts:
        return [hard_u8]
    parts.sort(key=lambda t: t[0], reverse=True)
    return [m for _a, m in parts[:max_parts]]


def _hybrid_context_fill(
    original: np.ndarray,
    lama: np.ndarray,
    mask: np.ndarray,
    scene: str | None = None,
) -> np.ndarray:
    """Choose scene-aware fill or color-matched LaMa. No row-stamping (causes smears)."""
    hard = mask > 127
    if not hard.any():
        return lama

    if scene is None:
        scene = classify_hole_scene(original, hard)

    if scene == "water" or _is_water_or_sky_ring(original, hard):
        # Skip soft TELEA water_primary — 2D exemplar fill is authoritative.
        return fill_for_scene(original, lama, mask, scene="water")

    if scene == "structured":
        # Structured floors/shelves: PatchMatch-first, barely keep LaMa LF.
        return fill_for_scene(original, lama, mask, scene="structured")

    matched = _match_fill_to_context(original, lama, mask)
    if scene in ("vegetation", "sky", "generic"):
        matched = _inject_context_texture(original, matched, mask, scene=scene)

    scrubbed = _scrub_other_people(original, mask)
    try:
        telea = inpaint_patch_opencv(scrubbed, mask, radius=7)
        k_area = int(hard.sum())
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
    low_diff = (diff < 22.0) & near

    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    res_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    orig_edges = cv2.Canny(orig_gray, 50, 130)
    res_edges = cv2.Canny(res_gray, 50, 130)
    # Original person edges that still show in the result outline.
    persistent = (orig_edges > 0) & (res_edges > 0) & near

    residual = low_diff | persistent

    # Face / sunglasses remnants: skin-like or very dark eye-socket bands that
    # still match the original chrominance inside the hole.
    hsv_o = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_r = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    skin = (
        (hsv_o[:, :, 0] < 35)
        & (hsv_o[:, :, 1] > 20)
        & (hsv_o[:, :, 1] < 160)
        & (hsv_o[:, :, 2] > 60)
        & (hsv_o[:, :, 2] < 230)
    )
    dark_glass = (hsv_o[:, :, 2] < 70) & (hsv_o[:, :, 1] < 80)
    face_like = (skin | dark_glass) & hard
    # Persist when result is still close to original in those pixels.
    face_ghost = face_like & (diff < 40.0)
    residual = residual | face_ghost

    # Bright/washed blob vs surrounding water — force full re-fill when off
    # OR when ghost coverage is material (face strip / silhouette).
    ring = near & (~hard)
    ghost_frac = float((diff[hard] < 18.0).mean()) if hard.any() else 0.0
    if int(ring.sum()) >= 64 and (int(residual.sum()) >= 150 or ghost_frac > 0.12):
        ring_lum = float(orig_gray[ring].mean())
        fill_lum = float(res_gray[hard].mean())
        if abs(fill_lum - ring_lum) > 8.0 or ghost_frac > 0.15 or int(face_ghost.sum()) >= 80:
            residual = residual | hard

    if int(residual.sum()) < 120:
        return None

    # Grow residual into a solid second-pass mask; keep it local to the first mask.
    res_u8 = residual.astype(np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    res_u8 = cv2.morphologyEx(res_u8, cv2.MORPH_CLOSE, close_k, iterations=2)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    res_u8 = cv2.dilate(res_u8, dilate_k, iterations=1)
    # Never expand far beyond the original selection halo.
    limit = cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    res_u8 = res_u8 & limit
    # Union with original hard mask so the second pass re-fills the whole subject.
    res_u8 = np.maximum(res_u8, hard.astype(np.uint8))
    # Keep people-in-front cutouts from the first hard mask.
    res_u8 = _preserve_occlusion_holes(hard.astype(np.uint8), res_u8)
    if protect is not None and protect.any():
        res_u8 = res_u8 & (~protect.astype(np.uint8))
        res_u8 = np.maximum(res_u8, hard.astype(np.uint8) & (~protect.astype(np.uint8)))
    if int(res_u8.sum()) < 150:
        return None
    return res_u8 * 255


def _seal_interior_face_voids(rgb: np.ndarray, hard_u8: np.ndarray, protect: np.ndarray | None) -> np.ndarray:
    """Force-include sunglasses / skin islands enclosed by the subject mask.

    Only seals holes inside a morphological closing of the subject — never the
    AABB, which previously spilled red fill onto the couple's glasses/hair.
    """
    import cv2

    hard = hard_u8 > 127
    if not hard.any():
        return hard_u8
    # Closed subject shape — interior voids only.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(hard.astype(np.uint8), cv2.MORPH_CLOSE, close_k, iterations=2) > 0
    if protect is not None and protect.any():
        closed = closed & (~protect)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    skin = (
        (hsv[:, :, 0] < 35)
        & (hsv[:, :, 1] > 18)
        & (hsv[:, :, 1] < 170)
        & (hsv[:, :, 2] > 55)
        & (hsv[:, :, 2] < 235)
    )
    dark_glass = (hsv[:, :, 2] < 75) & (hsv[:, :, 1] < 90)
    candidates = (skin | dark_glass) & closed & (~hard)
    if protect is not None and protect.any():
        candidates = candidates & (~protect)

    # Flood-fill enclosed holes inside closed subject (catches eye sockets).
    h, w = hard.shape
    filled = closed.astype(np.uint8)
    flood = filled.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    interior = (flood == 0) & (~hard)
    if protect is not None and protect.any():
        interior = interior & (~protect)
    # Only keep small interior components (eyes / logos), not couple-sized voids.
    add = np.zeros_like(hard, dtype=bool)
    for src in (candidates, interior):
        if not src.any():
            continue
        n_lab, lab, st, _ = cv2.connectedComponentsWithStats(src.astype(np.uint8), 8)
        for i in range(1, n_lab):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a < 15 or a > 5000:
                continue
            add |= lab == i
    if not add.any():
        return hard_u8
    out = hard | add
    if protect is not None and protect.any():
        out = out & (~protect)
    return (out.astype(np.uint8) * 255)


def _blobify_water_hole(
    rgb: np.ndarray,
    hard_u8: np.ndarray,
    protect: np.ndarray | None,
    raw: np.ndarray | None = None,
) -> np.ndarray:
    """Round the hole into a water-grown blob so the outline is not person-shaped."""
    import cv2

    hard = hard_u8 > 127
    if not hard.any():
        return hard_u8
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    waterish = ((hh >= 85) & (hh <= 135) & (ss >= 10) & (vv > 100)) | (
        (ss < 50) & (vv > 130) & (vv < 235)
    )
    # Heavy close + dilate into water — breaks head/shoulder silhouette.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    closed = cv2.morphologyEx(hard.astype(np.uint8), cv2.MORPH_CLOSE, k_close, iterations=2)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    grown = cv2.dilate(closed, k_dil, iterations=1)
    blob = ((grown > 0) & (hard | waterish)).astype(bool)
    # Keep largest component overlapping raw/hard.
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), 8)
    if n_lab > 2:
        ref = _as_bool_mask(raw) if raw is not None else hard
        best_i, best_score = 1, -1.0
        for i in range(1, n_lab):
            comp = lab == i
            score = float((comp & ref).sum()) + 0.05 * float(st[i, cv2.CC_STAT_AREA])
            if score > best_score:
                best_score, best_i = score, i
        blob = lab == best_i
    if protect is not None and protect.any():
        blob = blob & (~protect)
        if raw is not None:
            core = cv2.erode(
                _as_bool_mask(raw).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            ).astype(bool)
            blob = blob | core
    return (blob.astype(np.uint8) * 255)


def _erode_hard_near_protect(hard_u8: np.ndarray, protect: np.ndarray | None, raw: np.ndarray | None) -> np.ndarray:
    """Pull the target mask back from protected hair before fill (stops hair bite)."""
    import cv2

    hard = hard_u8 > 127
    if protect is None or not protect.any() or not hard.any():
        return hard_u8
    prot_fat = cv2.dilate(
        protect.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    ).astype(bool)
    # Erode only the portion of hard that touches protect halo.
    near = hard & prot_fat
    if not near.any():
        return hard_u8
    eroded = cv2.erode(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # Keep raw selection core so we don't drop the clicked subject.
    out = hard.copy()
    out[near] = eroded[near] > 0
    if raw is not None:
        raw_b = _as_bool_mask(raw)
        core = cv2.erode(
            raw_b.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        ).astype(bool)
        out = (out & (~protect)) | (raw_b & (~protect)) | core
    else:
        out = out & (~protect)
    return (out.astype(np.uint8) * 255)


def _hard_exemplar_overwrite(
    original: np.ndarray,
    result: np.ndarray,
    mask_u8: np.ndarray,
    protect: np.ndarray | None = None,
    scene: str = "water",
) -> np.ndarray:
    """Force pure exemplar fill into mask — no soft blend, no TELEA of person pixels."""
    hard = mask_u8 > 127
    if protect is not None and protect.any():
        hard = hard & (~protect)
    if not hard.any():
        return result
    m = (hard.astype(np.uint8) * 255)
    filled = fill_for_scene(original, result, m, scene=scene)  # type: ignore[arg-type]
    filled = _match_fill_to_context(original, filled, m)
    out = result.copy()
    out[hard] = filled[hard]
    return out


def _ghost_mad_expand_refill(
    original: np.ndarray,
    result: np.ndarray,
    hard_mask: np.ndarray,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """If fill MAD is too low (ghost), expand mask slightly and pure-exemplar re-fill."""
    import cv2

    hard = hard_mask > 127
    if protect is not None and protect.any():
        hard = hard & (~protect)
    if int(hard.sum()) < 200:
        return result

    diff = np.abs(result.astype(np.float32) - original.astype(np.float32)).mean(axis=2)
    mad = float(diff[hard].mean())
    ghost = float((diff[hard] < 18.0).mean())
    if mad >= 42.0 and ghost <= 0.14:
        # Still catch face/sunglass strips even when overall MAD looks ok.
        ghost_m = _residual_ghost_mask(original, result, hard_mask, protect=protect)
        if ghost_m is None:
            return result
        return _hard_exemplar_overwrite(original, result, ghost_m, protect=protect, scene="water")

    # Expand into water-like surroundings to break person-shaped silhouette edge.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    grown = cv2.dilate(hard.astype(np.uint8), k, iterations=1)
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    waterish = ((hh >= 85) & (hh <= 135) & (ss >= 10) & (vv > 100)) | (
        (ss < 50) & (vv > 130) & (vv < 235)
    )
    expanded = ((grown > 0) & (hard | waterish)).astype(np.uint8)
    if protect is not None and protect.any():
        expanded = expanded & (~protect.astype(np.uint8))
    expanded = np.maximum(expanded, hard.astype(np.uint8))
    return _hard_exemplar_overwrite(
        original, result, expanded * 255, protect=protect, scene="water"
    )


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
    global _last_removal_meta
    h, w = rgb.shape[:2]
    hard_bool = hard_mask > 127
    scene = classify_hole_scene(rgb, hard_bool)
    open_fill = scene_is_open_fill(scene) or _is_water_or_sky_ring(rgb, hard_bool)
    feather_px = _blend_feather_px(edge_softness)
    # Tighter feather on open water/sky so we don't haze couple hair.
    if open_fill:
        feather_px = max(3, int(round(feather_px * 0.40)))
    if scene == "structured":
        feather_px = max(3, int(round(feather_px * 0.55)))
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
    if scene == "structured":
        pad = min(pad + 40, 240)
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(w, xs.max() + pad)
    y1 = min(h, ys.max() + pad)

    crop_rgb = rgb[y0:y1, x0:x1].copy()
    crop_hard = hard_mask[y0:y1, x0:x1].copy()
    crop_protect = protect[y0:y1, x0:x1] if protect is not None else None
    crop_scene = classify_hole_scene(crop_rgb, crop_hard > 127)
    water_bg = crop_scene == "water" or _is_water_or_sky_ring(crop_rgb, crop_hard > 127)
    # Best + CUDA + downloaded weights: PowerPaint (structured) / SDXL (water, sky).
    # Otherwise structured indoor → classical TELEA, open water → classical exemplar.
    generative_tier = (
        mode == "ai"
        and get_removal_quality() == "best"
        and generative_engine_ready(crop_scene)
    )
    force_classical = (
        (crop_scene == "structured" and not generative_tier)
        or (water_bg and not generative_tier)
        or (mode == "quick" and crop_scene in ("structured", "generic"))
    )

    if force_classical:
        from services.inpaint_backends import QuickRemoveBackend

        backend = QuickRemoveBackend()
    else:
        backend = get_backend(mode)
    # LaMa samples neighboring pixels — scrub bystanders so faces don't bleed into the hole.
    # Generative SD already respects the mask; TELEA-scrubbing the couple leaves muddy context
    # that the diffusion model copies as a flat ghost plaque.
    if getattr(backend, "is_generative", False):
        crop_for_inpaint = crop_rgb
    else:
        crop_for_inpaint = _scrub_other_people(crop_rgb, crop_hard)
    textured_bg = _ring_edge_density(crop_rgb, crop_hard > 127) > 14.0
    open_bg = scene_is_open_fill(crop_scene) or water_bg
    use_seamless = mode == "ai" and not textured_bg and not open_bg and not getattr(
        backend, "is_generative", False
    )

    # Water without a generative engine: skip LaMa entirely — it leaves grey
    # person-shaped plaques. Exemplar only. Also the fallback when SDXL fails.
    def _water_exemplar_result() -> np.ndarray:
        global _last_removal_meta
        import cv2

        base_patch = fill_for_scene(crop_rgb, crop_rgb, crop_hard, scene="water")
        base_patch = _match_fill_to_context(crop_rgb, base_patch, crop_hard)
        from services.scene_context import _match_fill_hf_to_ring

        base_patch = _match_fill_hf_to_ring(
            crop_rgb, base_patch, crop_hard > 127, target_ratio=0.95, max_gain=2.4
        )
        _last_removal_meta = {
            "engine": "water_exemplar",
            "scene": crop_scene or "water",
            "mode": mode,
            "postfill": "classical",
        }
        if crop_protect is not None and crop_protect.any():
            base_patch = base_patch.copy()
            base_patch[crop_protect] = crop_rgb[crop_protect]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blend_hard = (cv2.dilate((crop_hard > 127).astype(np.uint8), k) * 255)
        water_feather = max(2, min(feather_px, 3))
        blend = _build_blend_mask(blend_hard, water_feather)
        if crop_protect is not None and crop_protect.any():
            blend = blend.copy()
            near_prot = (
                cv2.dilate(
                    crop_protect.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (water_feather * 2 + 1, water_feather * 2 + 1)
                    ),
                )
                > 0
            )
            hard_b = blend_hard > 127
            blend[near_prot] = hard_b[near_prot].astype(np.float32)
            blend[crop_protect] = 0.0
        # Hard interior composite against original crop.
        patch = (
            base_patch.astype(np.float32) * blend[:, :, None]
            + crop_rgb.astype(np.float32) * (1.0 - blend[:, :, None])
        ).astype(np.uint8)
        # Force pure exemplar inside the hard hole — kill soft feather face ghosts.
        hard_core = crop_hard > 127
        if crop_protect is not None and crop_protect.any():
            hard_core = hard_core & (~crop_protect)
        patch = patch.copy()
        patch[hard_core] = base_patch[hard_core]
        patch = _match_edge_band_luminance(crop_rgb, patch, blend_hard, protect=crop_protect)
        out = rgb.copy()
        out[y0:y1, x0:x1] = patch
        return out

    water_hole = water_bg or crop_scene == "water"
    if water_hole and not generative_tier:
        return _water_exemplar_result()

    try:
        inpainted_crop = backend.inpaint_crop(crop_for_inpaint, crop_hard)
    except Exception as exc:
        if generative_tier and water_hole:
            return _water_exemplar_result()
        if generative_tier and crop_scene == "structured":
            from services.inpaint_backends import QuickRemoveBackend

            backend = QuickRemoveBackend()
            crop_for_inpaint = _scrub_other_people(crop_rgb, crop_hard)
            inpainted_crop = backend.inpaint_crop(crop_for_inpaint, crop_hard)
        elif (
            isinstance(exc, InpaintError)
            and mode == "ai"
            and "blank white fill" in str(exc).lower()
        ):
            from services.inpaint_backends import QuickRemoveBackend

            inpainted_crop = QuickRemoveBackend().inpaint_crop(crop_for_inpaint, crop_hard)
            backend.engine = "lama"  # type: ignore[attr-defined]
        else:
            raise

    # Never let generative dilation / fill overwrite protected people (couple hair).
    if crop_protect is not None and crop_protect.any():
        inpainted_crop = inpainted_crop.copy()
        inpainted_crop[crop_protect] = crop_rgb[crop_protect]

    # Re-read after inpaint — AiRemoveBackend clears is_generative on LaMa fallback.
    generative = bool(getattr(backend, "is_generative", False))
    engine_name = str(getattr(backend, "engine", "classical" if mode == "quick" else "ai"))
    # Best Quality paid for SDXL / PowerPaint — keep that fill. Classical
    # exemplar/PatchMatch used to hard-overwrite every hole pixel afterward,
    # making shipped generative output byte-identical to classical.
    postfill = "seam_only" if generative else "classical"
    _last_removal_meta = {
        "engine": engine_name,
        "scene": crop_scene,
        "mode": mode,
        "postfill": postfill,
    }

    # Scene-aware post-fill. Classical paths still own the hole; generative keeps
    # its interior and only gets seam polish below.
    if not generative:
        # Context fill / water clone must sample the real photo, not the scrubbed copy.
        inpainted_crop = _hybrid_context_fill(
            crop_rgb, inpainted_crop, crop_hard, scene=crop_scene
        )
        if crop_scene == "structured":
            from services.scene_context import _match_fill_hf_to_ring

            inpainted_crop = _match_fill_hf_to_ring(
                crop_rgb, inpainted_crop, crop_hard > 127, target_ratio=0.65, max_gain=1.15
            )

    seamless = _try_seamless_blend(crop_rgb, inpainted_crop, crop_hard) if use_seamless else None
    base_patch = seamless if seamless is not None else inpainted_crop

    # Generative / water: tighter hole + harder composite so feather doesn't reintroduce the person.
    blend_hard = crop_hard
    if generative or open_bg:
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13) if open_bg else (11, 11))
        blend_hard = (cv2.dilate((crop_hard > 127).astype(np.uint8), k) * 255)
        feather_px = max(2, min(feather_px, 3 if open_bg else 4))

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

    # Generative owns the hard hole — never leave feathered original person pixels.
    if generative:
        hard_core = crop_hard > 127
        if crop_protect is not None and crop_protect.any():
            hard_core = hard_core & (~crop_protect)
        patch = patch.copy()
        patch[hard_core] = base_patch[hard_core]

    if not generative and not open_bg:
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
    if mode == "ai" and not generative and not textured_bg and not open_bg:
        patch = _ai_edge_polish(patch, crop_rgb, crop_hard, blend)

    # Seam polish: match the feather band. Generative hard_core stays authoritative.
    if generative:
        import cv2

        hard_core = crop_hard > 127
        if crop_protect is not None and crop_protect.any():
            hard_core = hard_core & (~crop_protect)
        eroded = cv2.erode(
            hard_core.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ).astype(bool)
        seam_patch = _match_edge_band_luminance(
            crop_rgb, patch, blend_hard, protect=crop_protect
        )
        patch = patch.copy()
        patch[~eroded] = seam_patch[~eroded]
        patch[hard_core & eroded] = base_patch[hard_core & eroded]
        if crop_protect is not None and crop_protect.any():
            patch[crop_protect] = crop_rgb[crop_protect]
    else:
        patch = _match_edge_band_luminance(crop_rgb, patch, blend_hard, protect=crop_protect)

    result = rgb.copy()
    result[y0:y1, x0:x1] = patch
    return result


def inpaint_objects(
    image: Image.Image,
    mask: Image.Image,
    mode: RemovalMode = "ai",
    edge_softness: float = 1.0,
    quality: str = "balanced",
    lock_point: tuple[float, float] | None = None,
) -> Image.Image:
    """Remove masked regions using AI Remove (erase/SDXL) or Quick Remove (ONNX LaMa)."""
    global _last_removal_meta
    set_removal_quality(quality)
    rgb = _pil_to_rgb_array(image)

    msk = np.array(mask.convert("L"))
    if msk.max() == 0:
        return image.convert("RGB")

    raw = (msk > 127).astype(np.uint8)
    protect = _non_target_people_protect(rgb, raw) | _occlusion_cutout_protect(raw)
    if lock_point is not None:
        protect = protect | lock_subject_protect_mask(image, lock_point)

    hard_mask = refine_mask_hard(mask, edge_softness=edge_softness, rgb=rgb)
    hard_mask = _apply_people_protect(hard_mask, protect, raw=raw)
    # Extra protect pass: dilation/halo often creeps into couple hair on coastal shots.
    if protect is not None and protect.any():
        import cv2

        prot_fat = cv2.dilate(
            protect.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
            iterations=1,
        ).astype(bool)
        hard_b = (hard_mask > 127) & (~prot_fat)
        # Keep only a tight raw core under protect — not full raw (couple spill).
        raw_core = cv2.erode(
            _as_bool_mask(raw).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ).astype(bool)
        hard_b = hard_b | (_as_bool_mask(raw) & (~prot_fat)) | raw_core
        hard_mask = hard_b.astype(np.uint8) * 255

    # Seal sunglasses / skin voids SAM left inside the subject (face remnant strip).
    hard_mask = _seal_interior_face_voids(rgb, hard_mask, protect)
    # Pull mask back from couple hair before fill.
    hard_mask = _erode_hard_near_protect(hard_mask, protect, raw=raw)

    scene = classify_hole_scene(rgb, hard_mask > 127)
    _last_removal_meta = {
        "engine": "pending",
        "scene": scene,
        "mode": mode,
        "postfill": "none",
    }
    # On water/sky, grow the hole a bit more so hair fringes don't leave a cutout plaque.
    # Never grow into protect (hair-safe); use a thinner grow than before.
    if scene_is_open_fill(scene) or _is_water_or_sky_ring(rgb, hard_mask > 127):
        import cv2

        # Grow into water to break person-shaped silhouette edge — but NOT into protect.
        # Stronger grow: silhouette edge is the #1 visual failure.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        grown = cv2.dilate((hard_mask > 127).astype(np.uint8), k, iterations=2)
        # Only keep growth where background looks like water/sky.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        waterish = ((hh >= 85) & (hh <= 135) & (ss >= 10) & (vv > 100)) | (
            (ss < 50) & (vv > 130) & (vv < 235)
        )
        core = hard_mask > 127
        grown = np.where(grown > 0, np.where(core | waterish, 1, 0), 0).astype(np.uint8)
        # Keep only the largest component near the raw selection (drop stray arms).
        n_lab, lab, st, _ = cv2.connectedComponentsWithStats(grown, 8)
        if n_lab > 2 and raw is not None and _as_bool_mask(raw).any():
            raw_b = _as_bool_mask(raw)
            best_i, best_score = 1, -1.0
            for i in range(1, n_lab):
                comp = lab == i
                score = float((comp & raw_b).sum()) + 0.15 * float(st[i, cv2.CC_STAT_AREA])
                if score > best_score:
                    best_score, best_i = score, i
            grown = (lab == best_i).astype(np.uint8)
        grown = _preserve_occlusion_holes(raw, grown)
        hard_mask = _apply_people_protect(grown * 255, protect, raw=raw)
        if protect is not None and protect.any():
            prot_fat = cv2.dilate(
                protect.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)),
                1,
            ).astype(bool)
            hard_b = (hard_mask > 127) & (~prot_fat)
            raw_core = cv2.erode(
                _as_bool_mask(raw).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            ).astype(bool)
            hard_b = hard_b | (_as_bool_mask(raw) & (~prot_fat)) | raw_core
            hard_mask = hard_b.astype(np.uint8) * 255
        hard_mask = _erode_hard_near_protect(hard_mask, protect, raw=raw)
        # Round into a water blob — person silhouette outline is the #1 visual fail.
        hard_mask = _blobify_water_hole(rgb, hard_mask, protect, raw=raw)
        hard_mask = _erode_hard_near_protect(hard_mask, protect, raw=raw)

    # Multi-person / structured: inpaint per connected component so wood floors
    # aren't melted by one LaMa/SDXL mega-hole across the aisle.
    # Always sequential for structured (library remove-others) — one instance at a time.
    parts = _split_mask_components(hard_mask, min_area=220, max_parts=18)
    multi = len(parts) >= 2 and (
        scene == "structured"
        or scene == "generic"
        or int((hard_mask > 127).sum()) > 18000
    )
    if scene == "structured" and len(parts) >= 2:
        multi = True
    if multi:
        result = rgb
        # Smallest-first for far crowd (tiny under globe), then larger midground.
        parts_sorted = sorted(parts, key=lambda p: int((p > 127).sum()))
        for part in parts_sorted:
            # Skip flecks that are only protect leftovers.
            if protect is not None and protect.any():
                usable = (part > 127) & (~protect)
                if int(usable.sum()) < 180:
                    continue
            result = _inpaint_once(
                result, part, mode=mode, edge_softness=edge_softness, protect=protect
            )
    else:
        result = _inpaint_once(rgb, hard_mask, mode=mode, edge_softness=edge_softness, protect=protect)

    # Second pass: classical residual cleanup — skip hard hole overwrite when a
    # generative engine already owns the fill. Soft-plaque rescue still runs when
    # the plaque detector fires (it replaces washed generative water only then).
    open_bg = scene_is_open_fill(scene) or _is_water_or_sky_ring(rgb, hard_mask > 127)
    gen_kept = _is_generative_engine(_last_removal_meta.get("engine"))
    if open_bg:
        residual = _soft_plaque_residual_mask(rgb, result, hard_mask, protect=protect)
        if residual is not None:
            # Soft-plaque detector fired — classical water synthesis rescues washed fills.
            rescued = _rescue_soft_generative_water(rgb, result, residual)
            rescued = fill_for_scene(rgb, rescued, residual, scene="water")
            rescued = _match_fill_to_context(rgb, rescued, residual)
            hard_r = residual > 127
            if protect is not None and protect.any():
                hard_r = hard_r & (~protect)
            out = result.copy()
            out[hard_r] = rescued[hard_r]
            result = out
            result = _match_edge_band_luminance(rgb, result, residual, protect=protect)
            _last_removal_meta = {
                **_last_removal_meta,
                "postfill": "plaque_rescue",
            }
        if not gen_kept:
            # Classical path: HF-match + ghost exemplar overwrite of the hole.
            from services.scene_context import _match_fill_hf_to_ring

            polished = _match_fill_hf_to_ring(rgb, result, hard_mask > 127, target_ratio=0.95)
            hard_b = hard_mask > 127
            if protect is not None and protect.any():
                hard_b = hard_b & (~protect)
            out = result.copy()
            out[hard_b] = polished[hard_b]
            result = out
            ghost = _residual_ghost_mask(rgb, result, hard_mask, protect=protect)
            if ghost is not None:
                result = _hard_exemplar_overwrite(rgb, result, ghost, protect=protect, scene="water")
            result = _ghost_mad_expand_refill(rgb, result, hard_mask, protect=protect)
            ghost2 = _residual_ghost_mask(rgb, result, hard_mask, protect=protect)
            if ghost2 is not None:
                result = _hard_exemplar_overwrite(rgb, result, ghost2, protect=protect, scene="water")
        elif protect is not None and protect.any():
            # Generative kept: still never leave paint on locked people.
            out = result.copy()
            out[protect] = rgb[protect]
            result = out
    elif mode == "ai" and not gen_kept:
        residual = _residual_ghost_mask(rgb, result, hard_mask, protect=protect)
        if residual is not None:
            # Structured: PatchMatch residual, not another SDXL mega-pass.
            if scene == "structured":
                rescued = fill_for_scene(rgb, result, residual, scene="structured")
                rescued = _match_fill_to_context(rgb, rescued, residual)
                hard_r = residual > 127
                if protect is not None and protect.any():
                    hard_r = hard_r & (~protect)
                out = result.copy()
                out[hard_r] = rescued[hard_r]
                result = out
            else:
                result = _inpaint_once(
                    rgb, residual, mode=mode, edge_softness=edge_softness, protect=protect
                )
        elif str(quality).lower() in ("best", "best_quality") and scene != "structured":
            # Best polish on open/generic only. Structured classical fills are
            # already authoritative — polish washed wood into milky blue plaques.
            polished = _inject_context_texture(rgb, result, hard_mask, scene=scene)
            polished = _match_fill_to_context(rgb, polished, hard_mask)
            hard_b = hard_mask > 127
            if protect is not None and protect.any():
                hard_b = hard_b & (~protect)
            out = result.copy()
            out[hard_b] = polished[hard_b]
            result = out
    elif mode == "ai" and gen_kept and protect is not None and protect.any():
        out = result.copy()
        out[protect] = rgb[protect]
        result = out

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
