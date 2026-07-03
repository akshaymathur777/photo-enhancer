"""Generate a simple Photo Enhancer app icon (assets/icon.ico)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "icon.ico"


def _draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size // 8
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=size // 6,
        fill=(15, 23, 42, 255),
    )
    inner = size // 4
    draw.rounded_rectangle(
        (inner, inner, size - inner, size - inner),
        radius=size // 10,
        fill=(56, 189, 248, 255),
    )
    dot = size // 3
    draw.ellipse(
        (size - dot - margin, margin, size - margin, margin + dot),
        fill=(251, 191, 36, 255),
    )
    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    images = [_draw_icon(s) for s, _ in sizes]
    images[0].save(OUT, format="ICO", sizes=sizes, append_images=images[1:])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
