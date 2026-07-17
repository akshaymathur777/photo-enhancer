"""Synthetic edge-quality smoke test for object removal pipeline."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.object_removal import inpaint_objects, refine_mask, segment_from_points


def _make_coastal_couple_scene(w: int = 800, h: int = 500) -> Image.Image:
    """Textured sky + water + three person silhouettes (middle one to remove).

    Flat solid bands starve LaMa of context; add mild gradients/noise so the fill
    has sky/water structure to copy — closer to a real coastal selfie.
    """
    rng = np.random.default_rng(42)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    sky = np.stack(
        [
            110 + 40 * (1.0 - yy[: h // 2]),
            165 + 30 * (1.0 - yy[: h // 2]),
            210 + 25 * yy[: h // 2],
        ],
        axis=-1,
    )
    water = np.stack(
        [
            30 + 25 * yy[h // 2 :],
            95 + 35 * yy[h // 2 :],
            145 + 20 * (1.0 - yy[h // 2 :]),
        ],
        axis=-1,
    )
    # Soft horizon band + ripple noise so inpaint has texture to borrow.
    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[: h // 2] = sky
    arr[h // 2 :] = water
    noise = rng.normal(0, 4.5, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255)
    # Gentle horizontal water ripples
    for i, y in enumerate(range(h // 2, h)):
        arr[y] += 3.0 * np.sin(np.linspace(0, 18, w) + i * 0.15)[:, None]
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    for cx, color in [(w // 4, [200, 80, 80]), (w // 2, [90, 90, 90]), (3 * w // 4, [200, 80, 80])]:
        arr[h // 3 : h - 40, cx - 35 : cx + 35] = color
        arr[h // 3 - 30 : h // 3 + 10, cx - 18 : cx + 18] = color
    return Image.fromarray(arr)


def run_synthetic() -> None:
    img = _make_coastal_couple_scene()
    w, h = img.size
    center = [w // 2, h // 2 - 40]
    mask = segment_from_points(img, [center], [1])
    refined = refine_mask(mask)
    result = inpaint_objects(img, mask)

    m = np.array(mask)
    assert m.max() > 0, "SAM mask empty"
    r = np.array(refined)
    assert r.max() > 0 and r.std() > 5, "refined mask should be soft at edges"

    out = np.array(result)
    # Middle person region should differ from original
    patch = out[h // 3 : h - 40, w // 2 - 35 : w // 2 + 35]
    orig = np.array(img)[h // 3 : h - 40, w // 2 - 35 : w // 2 + 35]
    diff = np.abs(patch.astype(np.int16) - orig.astype(np.int16)).mean()
    assert diff > 8, f"inpaint should change center person region (diff={diff:.1f})"

    # Hard-mask interior must not keep the original grey silhouette (ghosting).
    from services.object_removal import refine_mask_hard

    hard_arr = refine_mask_hard(mask) > 127
    inside = out[hard_arr]
    grey_like = np.abs(inside.astype(np.int16) - np.array([90, 90, 90])).mean(axis=1) < 30
    ghost_pct = float(grey_like.mean()) * 100.0
    assert ghost_pct < 15.0, f"too much grey ghosting inside mask ({ghost_pct:.1f}%)"

    # Side people should remain mostly unchanged
    left = out[h // 3 : h - 40, w // 4 - 35 : w // 4 + 35]
    left_orig = np.array(img)[h // 3 : h - 40, w // 4 - 35 : w // 4 + 35]
    left_diff = np.abs(left.astype(np.int16) - left_orig.astype(np.int16)).mean()
    assert left_diff < 25, f"left person should be mostly preserved (diff={left_diff:.1f})"

    out_path = Path(__file__).parent / "_test_removal_out.jpg"
    result.save(out_path, quality=92)
    print(f"OK synthetic coastal couple scene -> {out_path} (ghost={ghost_pct:.1f}%)")


if __name__ == "__main__":
    run_synthetic()
