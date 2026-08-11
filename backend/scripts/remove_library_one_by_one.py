"""Remove library background people one at a time with tight masks (brush-style)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from paths import configure_runtime_paths  # noqa: E402

configure_runtime_paths()

from services.object_removal import (  # noqa: E402
    _best_sam_mask_from_results,
    _expand_bbox,
    _get_sam_predictor,
    _mask_tensor_to_numpy,
    _pil_to_rgb_array,
    _yolo_seg_predict,
    inpaint_objects,
    mask_overlay_preview,
    segment_from_points,
)

SRC = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.23.51 PM.jpeg"
OUT = Path(__file__).resolve().parent / "_test_assets" / "library_one_by_one"
OUT.mkdir(parents=True, exist_ok=True)


def find_fg(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    res = _yolo_seg_predict(rgb, classes=[0], conf=0.4, imgsz=1280)[0]
    best, best_s = None, -1.0
    for mt in res.masks.data:
        m = _mask_tensor_to_numpy(mt)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h))
        b = m > 0.5
        ys, xs = np.where(b)
        s = (float(ys.max()) / h) * 3.5 + (b.sum() / float(h * w)) * 6.0 - abs(float(xs.mean()) / w - 0.5)
        if s > best_s:
            best_s, best = s, b
    assert best is not None
    return best


def list_bg_people(rgb: np.ndarray, fg: np.ndarray, conf: float = 0.35) -> list[dict]:
    h, w = rgb.shape[:2]
    fg_d = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    people: list[dict] = []
    res = _yolo_seg_predict(rgb, classes=[0], conf=conf, imgsz=1280)[0]
    if res.masks is None:
        return people
    for i, mt in enumerate(res.masks.data):
        m = _mask_tensor_to_numpy(mt)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h))
        b = m > 0.5
        area = int(b.sum())
        if area < 800:
            continue
        if float((b & (fg_d > 0)).sum()) / max(area, 1) > 0.22:
            continue
        ys, xs = np.where(b)
        cy, cx = float(ys.mean()), float(xs.mean())
        # Skip bottom scraps / likely busts on far sides
        if cy > h * 0.92:
            continue
        patch = rgb[b].mean(axis=0)
        if (cx / w < 0.15 or cx / w > 0.85) and float(patch.mean()) > 170 and area < 12000:
            continue
        conf_i = float(res.boxes.conf[i])
        xyxy = [float(v) for v in res.boxes.xyxy[i].detach().cpu().numpy().tolist()]
        people.append(
            {
                "mask": b,
                "area": area,
                "conf": conf_i,
                "bbox": xyxy,
                "cy": cy,
                "cx": cx,
                "click": [cx, cy],
            }
        )
    # Nearest/largest first (easier fills), then farther
    people.sort(key=lambda p: (-p["area"], -p["conf"]))
    return people


def tight_mask(rgb: np.ndarray, person: dict, fg_d: np.ndarray) -> np.ndarray:
    """SAM box+click, then very slight dilate — like a careful brush."""
    h, w = rgb.shape[:2]
    predictor = _get_sam_predictor()
    predictor.set_image(rgb)
    bbox = _expand_bbox(person["bbox"], h, w, 0.03)
    pt = person["click"]
    neg: list[list[float]] = []
    fys, fxs = np.where(fg_d > 0)
    if len(fxs):
        step = max(1, len(fxs) // 4)
        for j in range(0, min(len(fxs), step * 4), step):
            neg.append([float(fxs[j]), float(fys[j])])
    try:
        results = predictor(
            bboxes=[bbox],
            points=[pt] + neg,
            labels=[1] + [0] * len(neg),
            multimask_output=True,
        )
        sam = _best_sam_mask_from_results(results, h, w, click_pt=pt, guide_bbox=bbox)
        if sam is not None:
            b = (sam > 0.5) & (~(fg_d > 0))
            if int(b.sum()) >= 400:
                m = b.astype(np.uint8)
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                m[fg_d > 0] = 0
                return m
    except Exception:
        pass
    # Fallback: click segment
    mimg = segment_from_points(Image.fromarray(rgb), [pt], [1])
    m = (np.array(mimg) > 127).astype(np.uint8)
    m[fg_d > 0] = 0
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    m[fg_d > 0] = 0
    return m


def main() -> None:
    img = Image.open(SRC).convert("RGB")
    rgb = _pil_to_rgb_array(img)
    h, w = rgb.shape[:2]
    print(f"size {w}x{h}", flush=True)

    fg = find_fg(rgb)
    fg_d = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    print(f"FG px={int(fg.sum())}", flush=True)

    # Pass A: high-confidence clear people only
    people = list_bg_people(rgb, fg, conf=0.45)
    print(f"pass A clear targets={len(people)}", flush=True)
    for i, p in enumerate(people):
        print(
            f"  A{i} conf={p['conf']:.2f} area={p['area']} cy={p['cy']:.0f} cx={p['cx']:.0f}",
            flush=True,
        )

    current = img
    step = 0

    def remove_one(person: dict, label: str) -> bool:
        nonlocal current, step, rgb
        rgb = _pil_to_rgb_array(current)
        m = tight_mask(rgb, person, fg_d)
        px = int(m.sum())
        if px < 300:
            print(f"  skip {label} tiny px={px}", flush=True)
            return False
        # Cap absurd masks that swallowed shelves
        if px > 0.12 * h * w:
            print(f"  skip {label} too large px={px}", flush=True)
            return False
        mask_img = Image.fromarray(m * 255, mode="L")
        mask_overlay_preview(current, mask_img).save(OUT / f"step_{step:02d}_{label}_mask.jpg", quality=92)
        print(f"remove {label} px={px} quick...", flush=True)
        t0 = time.time()
        # Quick + low softness = brush-like local fill
        current = inpaint_objects(
            current, mask_img, mode="quick", edge_softness=0.6, quality="balanced"
        )
        print(f"  {time.time() - t0:.1f}s", flush=True)
        current.save(OUT / f"step_{step:02d}_{label}_after.jpg", quality=94)
        step += 1
        return True

    for i, p in enumerate(people):
        remove_one(p, f"A{i}")

    # Pass B: redetect remaining on updated image (medium conf)
    rgb = _pil_to_rgb_array(current)
    people_b = list_bg_people(rgb, fg, conf=0.20)
    # Drop anyone overlapping FG heavily (already filtered) and tiny
    print(f"pass B remaining={len(people_b)}", flush=True)
    for i, p in enumerate(people_b):
        print(
            f"  B{i} conf={p['conf']:.2f} area={p['area']} cy={p['cy']:.0f} cx={p['cx']:.0f}",
            flush=True,
        )
        # Only remove if reasonably sized — skip dust/bust scraps
        if p["area"] < 1200 and p["conf"] < 0.25:
            print(f"  skip B{i} weak", flush=True)
            continue
        remove_one(p, f"B{i}")

    outp = OUT / "final.jpg"
    current.save(outp, quality=95)
    current.save(ROOT / "assets" / "library_background_removed.jpg", quality=95)
    desk = Path.home() / "OneDrive" / "Desktop" / "library_background_removed.jpg"
    try:
        current.save(desk, quality=95)
    except Exception:
        pass
    print("WROTE", outp.resolve(), flush=True)


if __name__ == "__main__":
    main()
