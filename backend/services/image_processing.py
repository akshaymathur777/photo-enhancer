from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BG_MODELS = {
    "massive": "birefnet-massive",
    "general": "birefnet-general",
    "portrait": "birefnet-portrait",
    "matting": "birefnet-matting",
    "anime": "birefnet-dis",
    "bria": "bria-rmbg",
    "lite": "birefnet-general-lite",
}

DEFAULT_BG_MODEL = "massive"

_sessions: dict[str, object] = {}


def _get_session(model: str):
    from rembg import new_session

    if model not in _sessions:
        _sessions[model] = new_session(model)
    return _sessions[model]


def remove_background(image: Image.Image, model_key: str = DEFAULT_BG_MODEL) -> Image.Image:
    from rembg import remove

    model = BG_MODELS.get(model_key, BG_MODELS[DEFAULT_BG_MODEL])
    session = _get_session(model)
    return remove(image, session=session)


def replace_background(
    image: Image.Image,
    bg_type: str = "transparent",
    color: str = "#ffffff",
    blur_radius: int = 20,
    model_key: str = DEFAULT_BG_MODEL,
    bg_image: Image.Image | None = None,
) -> Image.Image:
    cutout = remove_background(image, model_key)
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")

    if bg_type == "transparent":
        return cutout

    w, h = cutout.size
    if bg_type == "color":
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        background = Image.new("RGBA", (w, h), (r, g, b, 255))
    elif bg_type == "blur":
        rgb = image.convert("RGB")
        arr = np.array(rgb)
        k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        blurred = cv2.GaussianBlur(arr, (k, k), 0)
        background = Image.fromarray(blurred).convert("RGBA")
    elif bg_type == "image" and bg_image is not None:
        bg = bg_image.convert("RGB")
        bg = bg.resize((w, h), Image.Resampling.LANCZOS)
        background = bg.convert("RGBA")
    else:
        return cutout

    background.paste(cutout, mask=cutout.split()[3])
    return background


def crop_rotate(
    image: Image.Image,
    left: int,
    top: int,
    width: int,
    height: int,
    angle: float = 0.0,
    flip_h: bool = False,
    flip_v: bool = False,
) -> Image.Image:
    """Crop a region, optionally flip, and rotate (expand canvas to fit)."""
    w, h = image.size
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    width = max(1, min(width, w - left))
    height = max(1, min(height, h - top))
    cropped = image.crop((left, top, left + width, top + height))
    if flip_h:
        cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        cropped = cropped.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if abs(angle) > 0.01:
        cropped = cropped.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    return cropped


def auto_enhance(image: Image.Image, strength: float = 1.0) -> Image.Image:
    rgb = image.convert("RGB")
    s = max(0.1, min(2.0, strength))

    contrast = ImageEnhance.Contrast(rgb).enhance(1.0 + 0.15 * s)
    color = ImageEnhance.Color(contrast).enhance(1.0 + 0.1 * s)
    sharp = ImageEnhance.Sharpness(color).enhance(1.0 + 0.2 * s)
    bright = ImageEnhance.Brightness(sharp).enhance(1.0 + 0.05 * s)

    if image.mode == "RGBA":
        bright = bright.convert("RGBA")
        bright.putalpha(image.split()[3])
    return bright


def adjust_image(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    exposure: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[3] if has_alpha else None
    rgb = image.convert("RGB")

    result = ImageEnhance.Brightness(rgb).enhance(brightness)
    if abs(exposure - 1.0) > 0.01:
        arr = np.array(result, dtype=np.float32) / 255.0
        gamma = max(0.25, min(4.0, 1.0 / exposure))
        adjusted = np.clip(np.power(arr, gamma), 0.0, 1.0)
        result = Image.fromarray((adjusted * 255).astype(np.uint8))
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(saturation)
    result = ImageEnhance.Sharpness(result).enhance(sharpness)

    if has_alpha and alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    return result


def denoise(image: Image.Image, strength: int = 10) -> Image.Image:
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[3] if has_alpha else None
    arr = np.array(image.convert("RGB"))
    k = max(3, min(21, strength if strength % 2 == 1 else strength + 1))
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, strength, strength, 7, 21)
    result = Image.fromarray(denoised)

    if has_alpha and alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    return result


def upscale(image: Image.Image, scale: int = 2) -> Image.Image:
    from services.upscale import upscale_realesrgan

    return upscale_realesrgan(image, scale)


def sharpen(image: Image.Image, amount: float = 1.5) -> Image.Image:
    has_alpha = image.mode == "RGBA"
    alpha = image.split()[3] if has_alpha else None
    rgb = image.convert("RGB")

    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(rgb, dtype=np.float32)
    blur_arr = np.array(blurred, dtype=np.float32)
    sharpened = np.clip(arr + amount * (arr - blur_arr), 0, 255).astype(np.uint8)
    result = Image.fromarray(sharpened)

    if has_alpha and alpha is not None:
        result = result.convert("RGBA")
        result.putalpha(alpha)
    return result


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    if fmt.upper() == "JPEG" and image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buf, format=fmt)
    return buf.getvalue()
