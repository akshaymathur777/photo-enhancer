"""In-process AI object removal on assets photos (no HTTP timeout)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from services.object_removal import (  # noqa: E402
    inpaint_objects,
    mask_overlay_preview,
    segment_from_points,
)

OUT = Path(__file__).resolve().parent / "_test_assets"
OUT.mkdir(parents=True, exist_ok=True)

COASTAL = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.31.13 PM.jpeg"
LIBRARY = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.23.51 PM.jpeg"


def run(name: str, path: Path, click: tuple[int, int], max_side: int | None = 1600) -> None:
    im = Image.open(path).convert("RGB")
    ow, oh = im.size
    scale = 1.0
    if max_side and max(ow, oh) > max_side:
        scale = max_side / float(max(ow, oh))
        im = im.resize((int(ow * scale), int(oh * scale)), Image.Resampling.LANCZOS)
    w, h = im.size
    cx = int(round(click[0] * scale))
    cy = int(round(click[1] * scale))
    print(f"\n=== {name} {ow}x{oh} -> {w}x{h} click=({cx},{cy}) ===", flush=True)

    im.save(OUT / f"{name}_before.jpg", quality=92)

    t0 = time.time()
    mask = segment_from_points(im, [(cx, cy)], [1])
    print(f"SAM {time.time() - t0:.1f}s coverage={(np.array(mask) > 127).mean()*100:.2f}%", flush=True)
    mask_overlay_preview(im, mask).save(OUT / f"{name}_mask.jpg", quality=92)

    t1 = time.time()
    result = inpaint_objects(im, mask, mode="ai", edge_softness=1.0)
    print(f"inpaint {time.time() - t1:.1f}s total {time.time() - t0:.1f}s", flush=True)
    result.save(OUT / f"{name}_after.jpg", quality=92)

    b = np.array(im)
    a = np.array(result)
    x0, x1 = max(0, cx - 180), min(w, cx + 180)
    y0, y1 = max(0, cy - 220), min(h, cy + 220)
    Image.fromarray(b[y0:y1, x0:x1]).save(OUT / f"{name}_crop_before.jpg", quality=92)
    Image.fromarray(a[y0:y1, x0:x1]).save(OUT / f"{name}_crop_after.jpg", quality=92)

    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    hard = np.array(mask) > 127
    print(f"mask-region mean abs diff: {diff[hard].mean() if hard.any() else 0:.2f}", flush=True)
    print(f"center-window mean abs diff: {diff[y0:y1, x0:x1].mean():.2f}", flush=True)
    print(f"global mean abs diff: {diff.mean():.2f}", flush=True)
    if name == "coastal":
        left = diff[int(h * 0.4) : int(h * 0.65), int(w * 0.2) : int(w * 0.4)].mean()
        right = diff[int(h * 0.4) : int(h * 0.65), int(w * 0.55) : int(w * 0.8)].mean()
        print(f"couple left/right mean abs diff: {left:.2f} / {right:.2f}", flush=True)
    # Ghost heuristic: inside hard mask, pixels still close to original
    if hard.any():
        ghost = ((diff < 18) & hard).sum() / hard.sum() * 100
        print(f"ghost-like residual inside mask: {ghost:.1f}%", flush=True)


if __name__ == "__main__":
    # Full-res coastal click was (1088,1200) on 2316x3088
    run("coastal", COASTAL, (1088, 1200), max_side=1600)
    run("library", LIBRARY, (420, 700), max_side=1280)
    print("\nDONE", flush=True)
