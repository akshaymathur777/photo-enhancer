import sys
import urllib.request
from pathlib import Path

from paths import get_models_dir

MODELS_DIR = get_models_dir()
LAMA_ONNX = MODELS_DIR / "lama_fp32.onnx"
LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

LAMA_BIG_TORCH = MODELS_DIR / "big-lama.pt"
LAMA_BIG_TORCH_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)

REALESRGAN_PTH = MODELS_DIR / "RealESRGAN_x4plus.pth"
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} to {dest}...", flush=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(100, block_num * block_size * 100 // total_size)
        sys.stderr.write(f"\r  {label}: {pct}%")
        sys.stderr.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    sys.stderr.write("\n")
    print(f"{label} downloaded.", flush=True)


def ensure_lama_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not LAMA_ONNX.exists():
        _download_with_progress(LAMA_URL, LAMA_ONNX, "LaMa ONNX (Quick Remove)")
    return LAMA_ONNX


def ensure_lama_big_torch() -> Path:
    """PyTorch JIT big-LaMa (~200 MB) for AI Remove mode."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not LAMA_BIG_TORCH.exists():
        _download_with_progress(LAMA_BIG_TORCH_URL, LAMA_BIG_TORCH, "LaMa big (AI Remove)")
    return LAMA_BIG_TORCH


def ensure_realesrgan_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not REALESRGAN_PTH.exists():
        print(f"Downloading Real-ESRGAN model to {REALESRGAN_PTH}...")
        urllib.request.urlretrieve(REALESRGAN_URL, REALESRGAN_PTH)
        print("Real-ESRGAN model downloaded.")
    return REALESRGAN_PTH
