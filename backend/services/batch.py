"""Batch folder processing."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

from PIL import Image

from services.image_io import open_image_bytes


def process_folder(
    input_dir: str,
    output_dir: str,
    processor: Callable[[Image.Image], Image.Image],
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"),
) -> dict:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    if not in_path.is_dir():
        raise ValueError(f"Input folder not found: {input_dir}")
    out_path.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors: list[str] = []
    for file in sorted(in_path.iterdir()):
        if file.suffix.lower() not in extensions:
            continue
        try:
            data = file.read_bytes()
            image = open_image_bytes(data, file.name)
            result = processor(image)
            out_file = out_path / f"{file.stem}_enhanced{file.suffix}"
            fmt = "PNG" if result.mode == "RGBA" else "JPEG"
            if fmt == "JPEG" and result.mode == "RGBA":
                result = result.convert("RGB")
            buf = io.BytesIO()
            result.save(buf, format=fmt)
            out_file.write_bytes(buf.getvalue())
            processed += 1
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

    return {"processed": processed, "errors": errors, "output_dir": str(out_path)}
