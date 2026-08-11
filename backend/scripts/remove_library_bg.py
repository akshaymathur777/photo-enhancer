"""Remove background people from the library photo — tight masks, sequential fill."""
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
)

SRC = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.23.51 PM.jpeg"
OUT = Path(__file__).resolve().parent / "_test_assets"
OUT.mkdir(parents=True, exist_ok=True)


def find_fg(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    res = _yolo_seg_predict(rgb, classes=[0], conf=0.4, imgsz=1280)[0]
    best, best_s = None, -1.0
    for i, mt in enumerate(res.masks.data):
        m = _mask_tensor_to_numpy(mt)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h))
        b = m > 0.5
        area = int(b.sum())
        ys, xs = np.where(b)
        s = (float(ys.max()) / h) * 3.5 + (area / float(h * w)) * 6.0 - abs(float(xs.mean()) / w - 0.5) * 1.2
        if s > best_s:
            best_s, best = s, b
    assert best is not None
    return best


def is_bust_like(rgb: np.ndarray, binary: np.ndarray, conf: float, w: int) -> bool:
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return True
    cx = float(xs.mean()) / w
    area = int(binary.sum())
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    aspect = bh / float(bw)
    mean = rgb[binary].mean(axis=0)
    bright = float(mean.mean())
    chroma = float(np.std(mean))
    on_side = cx < 0.20 or cx > 0.80
    if on_side and bright > 155 and aspect > 1.35 and area < 9000 and conf < 0.40:
        return True
    if on_side and bright > 170 and chroma < 20 and area < 14000:
        return True
    return False


def collect_targets(rgb: np.ndarray, fg: np.ndarray) -> list[dict]:
    h, w = rgb.shape[:2]
    fg_d = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)))
    people: list[dict] = []
    seen: list[np.ndarray] = []
    for conf in (0.50, 0.30, 0.15, 0.08):
        res = _yolo_seg_predict(rgb, classes=[0], conf=conf, imgsz=1280)[0]
        if res.masks is None:
            continue
        for i, mt in enumerate(res.masks.data):
            m = _mask_tensor_to_numpy(mt)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h))
            b = m > 0.5
            area = int(b.sum())
            if area < 400:
                continue
            ov = float((b & (fg_d > 0)).sum()) / max(area, 1)
            if ov > 0.22:
                continue
            conf_i = float(res.boxes.conf[i])
            ys, xs = np.where(b)
            cy, cx = float(ys.mean()), float(xs.mean())
            # Skip bottom-corner false positives / edge scraps
            if cy > h * 0.90 and area < 6000:
                continue
            if is_bust_like(rgb, b, conf_i, w):
                continue
            # Keep real people: high conf OR mid-aisle with decent size
            aisle = 0.12 <= (cx / w) <= 0.88
            if conf_i < 0.20 and not (aisle and area > 1200):
                continue
            dup = False
            for s in seen:
                inter = float((b & s).sum())
                union = float((b | s).sum()) + 1e-6
                if inter / union > 0.4:
                    dup = True
                    break
            if dup:
                continue
            xyxy = [float(v) for v in res.boxes.xyxy[i].detach().cpu().numpy().tolist()]
            people.append(
                {"mask": b, "area": area, "conf": conf_i, "bbox": xyxy, "cy": cy, "cx": cx}
            )
            seen.append(b)
    # Prefer clear near people first (largest / highest conf), then farther
    people.sort(key=lambda p: (-p["conf"], -p["area"]))
    return people


def refine_sam(rgb: np.ndarray, person: dict, fg_d: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    predictor = _get_sam_predictor()
    predictor.set_image(rgb)
    bbox = _expand_bbox(person["bbox"], h, w, 0.05)
    pt = [
        (person["bbox"][0] + person["bbox"][2]) / 2,
        (person["bbox"][1] + person["bbox"][3]) / 2,
    ]
    neg: list[list[float]] = []
    fys, fxs = np.where(fg_d > 0)
    if len(fxs):
        step = max(1, len(fxs) // 5)
        for j in range(0, min(len(fxs), step * 5), step):
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
            if int(b.sum()) >= 200:
                return b
    except Exception:
        pass
    return person["mask"] & (~(fg_d > 0))


def main() -> None:
    img0 = Image.open(SRC).convert("RGB")
    rgb0 = _pil_to_rgb_array(img0)
    h, w = rgb0.shape[:2]
    print(f"size {w}x{h}", flush=True)

    fg = find_fg(rgb0)
    fg_d = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)))
    print(f"FG px={int(fg.sum())}", flush=True)

    people = collect_targets(rgb0, fg)
    print(f"targets={len(people)}", flush=True)
    for i, p in enumerate(people):
        print(
            f"  {i} conf={p['conf']:.2f} area={p['area']} cy={p['cy']:.0f} cx={p['cx']:.0f}",
            flush=True,
        )

    preview = np.zeros((h, w), dtype=np.uint8)
    current = img0
    rgb = rgb0.copy()

    for i, p in enumerate(people):
        t0 = time.time()
        m = refine_sam(rgb, p, fg_d)
        m_u8 = m.astype(np.uint8)
        # Tight hair halo only — large dilate caused shelf smears before
        m_u8 = cv2.dilate(m_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        m_u8 = cv2.morphologyEx(
            m_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
        )
        m_u8[fg_d > 0] = 0
        px = int(m_u8.sum())
        if px < 200:
            print(f"skip {i} tiny", flush=True)
            continue
        preview = np.maximum(preview, m_u8 * 255)
        mask_img = Image.fromarray(m_u8 * 255, mode="L")

        # Library shelves: classical LaMa beats SDXL plaques for this scene
        mode = "quick"
        print(f"remove {i}/{len(people)-1} mode={mode} px={px}...", flush=True)
        current = inpaint_objects(current, mask_img, mode=mode, edge_softness=0.75, quality="balanced")
        rgb = _pil_to_rgb_array(current)
        print(f"  {time.time()-t0:.1f}s", flush=True)
        current.save(OUT / f"library_clean_{i:02d}.jpg", quality=92)

    mask_overlay_preview(img0, Image.fromarray(preview, mode="L")).save(
        OUT / "library_clean_mask.jpg", quality=92
    )
    outp = OUT / "library_clean_final.jpg"
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
