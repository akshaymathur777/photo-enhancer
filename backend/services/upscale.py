from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Process large photos in tiles to avoid GPU OOM (e.g. 12MP+ iPhone shots).
TILE_SIZE = 512
TILE_PAD = 16
TILE_THRESHOLD_MP = 2.0  # use tiling when source exceeds ~2MP


def _get_model():
    global _model
    if _model is None:
        from spandrel import ModelLoader

        from services.model_download import ensure_realesrgan_model

        path = ensure_realesrgan_model()
        descriptor = ModelLoader().load_from_file(str(path))
        _model = descriptor.model.to(_device).eval()
    return _model


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(_device)


def _tensor_to_pil(tensor: torch.Tensor, alpha: Image.Image | None = None) -> Image.Image:
    arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    rgb = Image.fromarray((arr * 255).astype(np.uint8))
    if alpha is not None:
        if alpha.size != rgb.size:
            alpha = alpha.resize(rgb.size, Image.Resampling.LANCZOS)
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)
    return rgb


def _run_model(model, tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(tensor)


def _upscale_whole(image: Image.Image, scale: int, alpha: Image.Image | None) -> Image.Image:
    model = _get_model()
    out = _run_model(model, _pil_to_tensor(image))
    result = _tensor_to_pil(out, alpha)
    if scale == 2:
        w, h = image.size
        result = result.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    return result


def _upscale_tiled(image: Image.Image, scale: int, alpha: Image.Image | None) -> Image.Image:
    """Tile-based Real-ESRGAN for large images (model runs at 4x internally)."""
    model = _get_model()
    w, h = image.size
    out_w, out_h = w * 4, h * 4
    output = Image.new("RGB", (out_w, out_h))

    rgb = image.convert("RGB")
    arr = np.array(rgb)

    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            x1 = max(x - TILE_PAD, 0)
            y1 = max(y - TILE_PAD, 0)
            x2 = min(x + TILE_SIZE + TILE_PAD, w)
            y2 = min(y + TILE_SIZE + TILE_PAD, h)

            tile = arr[y1:y2, x1:x2]
            tile_tensor = torch.from_numpy(tile.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            tile_tensor = tile_tensor.to(_device)

            out_tile = _run_model(model, tile_tensor)
            out_arr = out_tile.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            out_tile_img = Image.fromarray((out_arr * 255).astype(np.uint8))

            # Map input crop back to output coordinates (model is 4x).
            crop_left = (x - x1) * 4
            crop_top = (y - y1) * 4
            tile_w = min(TILE_SIZE, w - x)
            tile_h = min(TILE_SIZE, h - y)
            crop_right = crop_left + tile_w * 4
            crop_bottom = crop_top + tile_h * 4

            cropped = out_tile_img.crop((crop_left, crop_top, crop_right, crop_bottom))
            output.paste(cropped, (x * 4, y * 4))

    result: Image.Image = output
    if alpha is not None:
        scaled_alpha = alpha.resize((out_w, out_h), Image.Resampling.LANCZOS)
        result = result.convert("RGBA")
        result.putalpha(scaled_alpha)
    if scale == 2:
        result = result.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    return result


def upscale_realesrgan(image: Image.Image, scale: int = 2) -> Image.Image:
    """Upscale with Real-ESRGAN x4plus (runs at 4x, resized for 2x)."""
    if scale not in (2, 4):
        raise ValueError("Scale must be 2 or 4")

    has_alpha = image.mode == "RGBA"
    alpha = image.split()[3] if has_alpha else None

    src_mp = (image.size[0] * image.size[1]) / 1_000_000
    if src_mp > TILE_THRESHOLD_MP:
        return _upscale_tiled(image, scale, alpha)
    return _upscale_whole(image, scale, alpha)
