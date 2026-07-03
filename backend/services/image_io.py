from io import BytesIO

from PIL import Image

_heif_registered = False

HEIC_EXTENSIONS = {".heic", ".heif", ".hif"}


def register_heif() -> None:
    global _heif_registered
    if not _heif_registered:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _heif_registered = True


def is_heic_filename(name: str) -> bool:
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return ext in HEIC_EXTENSIONS


def open_image_bytes(data: bytes, filename: str = "") -> Image.Image:
    register_heif()
    return Image.open(BytesIO(data))
