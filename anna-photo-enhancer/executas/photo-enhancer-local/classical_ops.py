"""Classical (CPU) image ops for Option B — no CUDA / no host image APIs."""

from __future__ import annotations

import base64
import io
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def bundled_u2netp_candidates() -> list[Path]:
    """Locations where the lite rembg weights may be baked for offline first-run."""
    paths: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        root = Path(meipass)
        paths.append(root / "models" / "u2netp.onnx")
        paths.append(root / "u2netp.onnx")
    here = Path(__file__).resolve().parent
    paths.append(here / "models" / "u2netp.onnx")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        paths.append(exe_dir / "models" / "u2netp.onnx")
        paths.append(exe_dir / "u2netp.onnx")
    return paths


def seed_bundled_u2netp() -> str | None:
    """Copy baked ``u2netp.onnx`` into rembg's model dir when present.

    Avoids a flaky GitHub download on Cloud Agent / first Remove BG invoke.
    Returns the path rembg should use, or None if nothing was bundled/seeded.
    """
    src = next((p for p in bundled_u2netp_candidates() if p.is_file()), None)
    if src is None:
        return None
    try:
        from rembg.sessions.u2netp import U2netpSession

        existing = U2netpSession.resolve_existing("u2netp.onnx")
        if existing and Path(existing).is_file():
            return existing
        dest_dir = Path(U2netpSession.model_dir())
        dest = dest_dir / "u2netp.onnx"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        return str(dest)
    except Exception:
        return None


def decode_image_b64(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def auto_enhance(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Mild classical enhance — mirrors desktop auto_enhance spirit."""
    s = max(0.0, min(float(strength), 2.0))
    result = image.convert("RGB")
    result = ImageEnhance.Color(result).enhance(1.0 + 0.08 * s)
    result = ImageEnhance.Brightness(result).enhance(1.0 + 0.03 * s)
    result = ImageEnhance.Contrast(result).enhance(1.0 + 0.06 * s)
    result = ImageEnhance.Sharpness(result).enhance(1.0 + 0.12 * s)
    if s >= 1.2:
        result = result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))
    return result


def crop_rotate(
    image: Image.Image,
    left: int = 0,
    top: int = 0,
    width: int | None = None,
    height: int | None = None,
    angle: float = 0.0,
    flip_h: bool = False,
    flip_v: bool = False,
) -> Image.Image:
    w, h = image.size
    left = max(0, min(int(left), w - 1))
    top = max(0, min(int(top), h - 1))
    cw = max(1, min(int(width if width is not None else w - left), w - left))
    ch = max(1, min(int(height if height is not None else h - top), h - top))
    cropped = image.crop((left, top, left + cw, top + ch))
    if flip_h:
        cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        cropped = cropped.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if abs(float(angle)) > 0.01:
        cropped = cropped.rotate(float(angle), expand=True, resample=Image.Resampling.BICUBIC)
    return cropped


def adjust_image(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    exposure: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    rgb = image.convert("RGB")
    result = ImageEnhance.Brightness(rgb).enhance(float(brightness))
    if abs(float(exposure) - 1.0) > 0.01:
        arr = np.array(result, dtype=np.float32) / 255.0
        gamma = max(0.25, min(4.0, 1.0 / max(0.25, float(exposure))))
        adjusted = np.clip(np.power(arr, gamma), 0.0, 1.0)
        result = Image.fromarray((adjusted * 255).astype(np.uint8))
    result = ImageEnhance.Contrast(result).enhance(float(contrast))
    result = ImageEnhance.Color(result).enhance(float(saturation))
    result = ImageEnhance.Sharpness(result).enhance(float(sharpness))
    return result


def sharpen(image: Image.Image, amount: float = 1.5) -> Image.Image:
    rgb = image.convert("RGB")
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(rgb, dtype=np.float32)
    blur_arr = np.array(blurred, dtype=np.float32)
    a = max(0.0, min(4.0, float(amount)))
    sharpened = np.clip(arr + a * (arr - blur_arr), 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)


def denoise(image: Image.Image, strength: int = 10) -> Image.Image:
    import cv2

    arr = np.array(image.convert("RGB"))
    s = max(1, min(21, int(strength)))
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, s, s, 7, 21)
    return Image.fromarray(denoised)


def restore(image: Image.Image) -> Image.Image:
    return sharpen(denoise(image, strength=10), amount=1.4)


def apply_blur(image: Image.Image, radius: int = 8) -> Image.Image:
    r = max(1, min(40, int(radius)))
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=r))


def apply_vignette(image: Image.Image, strength: float = 0.5) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_d = np.sqrt(cx**2 + cy**2) + 1e-6
    s = max(0.0, min(1.0, float(strength)))
    vignette = 1.0 - (dist / max_d) ** 1.8 * s
    out = np.clip(arr * vignette[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def apply_grain(image: Image.Image, amount: float = 0.04) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(42)
    a = max(0.0, min(0.2, float(amount)))
    noise = rng.normal(0, a * 255, arr.shape)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def apply_prism(image: Image.Image, strength: float = 0.6) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    s = max(0.1, min(1.5, float(strength)))
    shift = max(2, int(w * 0.008 * s))
    r = np.roll(arr[:, :, 0], shift, axis=1)
    b = np.roll(arr[:, :, 2], -shift, axis=1)
    out = np.stack([r, arr[:, :, 1], b], axis=-1)
    out = np.clip(arr * (1 - s * 0.5) + out * (s * 0.5), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


_LOOKS = {
    "cinematic",
    "vintage",
    "vivid",
    "bw_dramatic",
    "golden_hour",
    "cool_moody",
    "sepia",
    "faded",
    "punchy",
    "soft_glow",
    "matte",
    "warm_film",
}


def list_looks() -> list[str]:
    return sorted(_LOOKS)


def apply_look(image: Image.Image, style: str = "cinematic") -> Image.Image:
    """PIL/numpy Looks subset (mirrors browser canvas_ops)."""
    key = (style or "cinematic").lower().strip().replace("-", "_").replace(" ", "_")
    if key in ("bw", "black_and_white", "monochrome"):
        key = "bw_dramatic"
    rgb = image.convert("RGB")
    if key == "vintage":
        arr = np.array(rgb, dtype=np.float32)
        arr[..., 0] = arr[..., 0] * 1.06 + 12
        arr[..., 2] = arr[..., 2] * 0.88
        arr = np.clip(arr * 0.92 + 18, 0, 255).astype(np.uint8)
        return apply_grain(Image.fromarray(arr), 0.035)
    if key == "vivid":
        return adjust_image(rgb, saturation=1.45, contrast=1.15, sharpness=1.2)
    if key == "bw_dramatic":
        gray = rgb.convert("L")
        arr = np.array(gray, dtype=np.float32)
        arr = np.clip((arr - 128) * 1.35 + 128, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    if key == "golden_hour":
        arr = np.array(rgb, dtype=np.float32)
        arr[..., 0] = np.clip(arr[..., 0] * 1.12 + 10, 0, 255)
        arr[..., 1] = np.clip(arr[..., 1] * 1.06 + 6, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.92, 0, 255)
        return adjust_image(Image.fromarray(arr.astype(np.uint8)), brightness=1.08, sharpness=1.1)
    if key == "cool_moody":
        arr = np.array(rgb, dtype=np.float32)
        arr[..., 2] = np.clip(arr[..., 2] * 1.14 + 8, 0, 255)
        arr[..., 0] = np.clip(arr[..., 0] * 0.90, 0, 255)
        arr = np.clip(arr * 0.94, 0, 255).astype(np.uint8)
        return adjust_image(Image.fromarray(arr), contrast=1.12)
    if key == "sepia":
        arr = np.array(rgb, dtype=np.float32)
        r = arr[..., 0] * 0.393 + arr[..., 1] * 0.769 + arr[..., 2] * 0.189
        g = arr[..., 0] * 0.349 + arr[..., 1] * 0.686 + arr[..., 2] * 0.168
        b = arr[..., 0] * 0.272 + arr[..., 1] * 0.534 + arr[..., 2] * 0.131
        return Image.fromarray(np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8))
    if key == "faded":
        return adjust_image(rgb, saturation=0.85, contrast=0.82)
    if key == "punchy":
        return adjust_image(rgb, contrast=1.25, saturation=1.3, sharpness=1.3)
    if key == "soft_glow":
        return adjust_image(apply_blur(rgb, 2), brightness=1.06)
    if key == "matte":
        return adjust_image(rgb, brightness=1.06, contrast=0.78, saturation=0.92)
    if key == "warm_film":
        arr = np.array(rgb, dtype=np.float32)
        arr[..., 0] = np.clip(arr[..., 0] * 1.08, 0, 255)
        arr[..., 1] = np.clip(arr[..., 1] * 1.02, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.92, 0, 255)
        return apply_grain(Image.fromarray(arr.astype(np.uint8)), 0.02)
    # cinematic default
    arr = np.array(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    shadow = np.clip(1.0 - lum * 1.4, 0, 1)
    highlight = np.clip((lum - 0.45) * 2.2, 0, 1)
    r = np.clip(r + highlight * 0.12 - shadow * 0.04, 0, 1)
    g = np.clip(g + highlight * 0.06 - shadow * 0.10, 0, 1)
    b = np.clip(b + highlight * 0.02 + shadow * 0.08, 0, 1)
    out = Image.fromarray((np.stack([r, g, b], axis=-1) * 255).astype(np.uint8))
    return adjust_image(out, contrast=1.22, saturation=1.08)


def inpaint_opencv(
    image: Image.Image,
    mask: Image.Image,
    method: str = "telea",
) -> Image.Image:
    import cv2

    rgb = np.array(image.convert("RGB"))
    m = np.array(mask.convert("L"))
    hard = ((m > 127).astype(np.uint8)) * 255
    ys, xs = np.where(hard > 0)
    if len(xs) == 0:
        return image.convert("RGB")
    hole = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    radius = max(5, int(min(rgb.shape[:2]) * 0.01), int(hole * 0.02))
    radius = min(radius, 18)
    flag = cv2.INPAINT_NS if method.lower() == "ns" else cv2.INPAINT_TELEA
    out = cv2.inpaint(rgb, hard, radius, flag)
    return Image.fromarray(out)


def remove_bg_rembg(image: Image.Image, model_key: str = "lite") -> Image.Image:
    """Optional rembg path — requires rembg + onnxruntime in the local env.

    ``lite`` uses ``u2netp`` (~4MB). Prefer a baked copy (see ``seed_bundled_u2netp``)
    so Cloud Agent / first-run does not depend on GitHub downloads.
    ``general`` / ``portrait`` use heavier checkpoints when installed.
    """
    from rembg import new_session, remove

    model_map = {
        "lite": "u2netp",
        "general": "u2net",
        "portrait": "birefnet-portrait",
    }
    key = (model_key or "lite").lower().strip()
    model = model_map.get(key, model_map["lite"])
    if model == "u2netp":
        seed_bundled_u2netp()
    session = new_session(model)
    out = remove(image.convert("RGBA"), session=session)
    return out if isinstance(out, Image.Image) else Image.open(io.BytesIO(out))


def result_payload(image: Image.Image, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_png_b64": encode_png_b64(image),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        **meta,
    }
