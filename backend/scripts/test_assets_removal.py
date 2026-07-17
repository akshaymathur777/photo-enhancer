"""Test AI object removal on assets folder photos."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_test_assets"
OUT.mkdir(parents=True, exist_ok=True)

COASTAL = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.31.13 PM.jpeg"
LIBRARY = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.23.51 PM.jpeg"

# Coastal: third person behind couple (center background striped shirt)
COASTAL_CLICK = (1088, 1200)
# Library: mid-left background tourist near busts/aisle
LIBRARY_CLICK = (420, 700)


def run(name: str, path: Path, click: tuple[int, int]) -> None:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    print(f"\n=== {name} {w}x{h} click={click} ===")
    before = OUT / f"{name}_before.jpg"
    im.save(before, quality=92)

    with path.open("rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        data = {
            "points": json.dumps([list(click)]),
            "labels": json.dumps([1]),
            "inpaint_mode": "ai",
            "mode": "best_quality",
            "edge_softness": "1.0",
        }
        r = httpx.post(
            "http://127.0.0.1:8000/api/remove-objects",
            files=files,
            data=data,
            timeout=600.0,
        )

    print("status", r.status_code, "bytes", len(r.content))
    if r.status_code != 200:
        print(r.text[:800])
        return

    after = OUT / f"{name}_after.jpg"
    after.write_bytes(r.content)
    a = np.array(Image.open(after).convert("RGB"))
    b = np.array(im)

    x, y = click
    x0, x1 = max(0, x - 200), min(w, x + 200)
    y0, y1 = max(0, y - 250), min(h, y + 250)
    Image.fromarray(b[y0:y1, x0:x1]).save(OUT / f"{name}_crop_before.jpg", quality=92)
    Image.fromarray(a[y0:y1, x0:x1]).save(OUT / f"{name}_crop_after.jpg", quality=92)

    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    print(f"center-window mean abs diff: {diff[y0:y1, x0:x1].mean():.2f}")
    print(f"global mean abs diff: {diff.mean():.2f}")

    if name == "coastal":
        # Approx couple regions (woman left, man right)
        left = diff[1200:1800, 500:900].mean()
        right = diff[1200:1800, 1400:1900].mean()
        print(f"couple left/right mean abs diff: {left:.2f} / {right:.2f}")

    print(f"saved {after}")


if __name__ == "__main__":
    run("coastal", COASTAL, COASTAL_CLICK)
    run("library", LIBRARY, LIBRARY_CLICK)
    print("\nDONE")
