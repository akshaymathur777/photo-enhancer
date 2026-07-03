"""Model download registry, progress tracking, and on-demand ensure helpers."""

from __future__ import annotations

import sys
import threading
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

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

# rembg session ids used by image_processing.BG_MODELS (+ bria)
REMBG_MODEL_IDS = [
    "birefnet-massive",
    "birefnet-general",
    "birefnet-portrait",
    "birefnet-matting",
    "birefnet-dis",
    "bria-rmbg",
    "birefnet-general-lite",
]

ULTRALYTICS_DIR = MODELS_DIR / "ultralytics"
U2NET_DIR = MODELS_DIR / "u2net"

_progress_lock = threading.Lock()
_download_thread: threading.Thread | None = None


@dataclass
class ProgressState:
    status: str = "idle"  # idle | downloading | complete | error
    current_model_id: str | None = None
    current_model_name: str | None = None
    bytes_downloaded: int = 0
    bytes_total: int = 0
    percent: float = 0.0
    model_statuses: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    message: str | None = None


_state = ProgressState()


def _u2net_home() -> Path:
    import os

    return Path(os.environ.get("U2NET_HOME", str(U2NET_DIR)))


def _ultralytics_home() -> Path:
    import os

    return Path(os.environ.get("ULTRALYTICS_HOME", str(ULTRALYTICS_DIR)))


def _find_ultralytics_weight(name: str) -> Path | None:
    home = _ultralytics_home()
    for candidate in (home / name, home / "weights" / name, MODELS_DIR / name):
        if candidate.is_file():
            return candidate
    return None


def _rembg_present(model_id: str) -> bool:
    return (_u2net_home() / f"{model_id}.onnx").is_file()


def _download_rembg(model_id: str) -> None:
    from rembg.sessions import sessions

    session_cls = sessions.get(model_id)
    if session_cls is None:
        raise ValueError(f"Unknown rembg model: {model_id}")
    session_cls.download_models()


def _download_ultralytics(name: str) -> None:
    if name.startswith("sam"):
        from ultralytics.models.sam import SAM2Predictor

        SAM2Predictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=name))
    else:
        from ultralytics import YOLO

        YOLO(name)


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    size_estimate_mb: int
    feature: str
    present_fn: Callable[[], bool]
    download_fn: Callable[[], None]

    def status_key(self) -> str:
        return self.id


def _build_registry() -> list[ModelEntry]:
    entries: list[ModelEntry] = [
        ModelEntry(
            id="lama_onnx",
            name="LaMa ONNX (Quick Remove)",
            size_estimate_mb=196,
            feature="Object removal (Quick)",
            present_fn=lambda: LAMA_ONNX.is_file(),
            download_fn=lambda: _download_direct(LAMA_URL, LAMA_ONNX, "LaMa ONNX (Quick Remove)", "lama_onnx"),
        ),
        ModelEntry(
            id="lama_big",
            name="LaMa big (AI Remove)",
            size_estimate_mb=196,
            feature="Object removal (AI)",
            present_fn=lambda: LAMA_BIG_TORCH.is_file(),
            download_fn=lambda: _download_direct(
                LAMA_BIG_TORCH_URL, LAMA_BIG_TORCH, "LaMa big (AI Remove)", "lama_big"
            ),
        ),
        ModelEntry(
            id="realesrgan",
            name="Real-ESRGAN x4plus",
            size_estimate_mb=64,
            feature="Upscale 2× / 4×",
            present_fn=lambda: REALESRGAN_PTH.is_file(),
            download_fn=lambda: _download_direct(
                REALESRGAN_URL, REALESRGAN_PTH, "Real-ESRGAN x4plus", "realesrgan"
            ),
        ),
        ModelEntry(
            id="sam21",
            name="SAM 2.1 Base",
            size_estimate_mb=160,
            feature="Smart object selection",
            present_fn=lambda: _find_ultralytics_weight("sam2.1_b.pt") is not None,
            download_fn=lambda: _download_ultralytics_tracked("sam2.1_b.pt", "sam21", "SAM 2.1 Base"),
        ),
        ModelEntry(
            id="yolov8_seg",
            name="YOLOv8 Seg (people)",
            size_estimate_mb=7,
            feature="Auto-detect people",
            present_fn=lambda: _find_ultralytics_weight("yolov8n-seg.pt") is not None,
            download_fn=lambda: _download_ultralytics_tracked("yolov8n-seg.pt", "yolov8_seg", "YOLOv8 Seg"),
        ),
    ]

    rembg_labels = {
        "birefnet-massive": ("BiRefNet Massive", "Background removal (default)", 973),
        "birefnet-general": ("BiRefNet General", "Background removal", 973),
        "birefnet-portrait": ("BiRefNet Portrait", "Background removal (portrait)", 973),
        "birefnet-matting": ("BiRefNet Matting", "Background removal (matting)", 973),
        "birefnet-dis": ("BiRefNet DIS", "Background removal (anime/art)", 973),
        "bria-rmbg": ("BRIA RMBG 2.0", "Background removal (BRIA)", 220),
        "birefnet-general-lite": ("BiRefNet Lite", "Background removal (fast)", 44),
    }

    for model_id, (label, feature, size_mb) in rembg_labels.items():
        safe_id = model_id.replace("-", "_")
        entries.append(
            ModelEntry(
                id=safe_id,
                name=label,
                size_estimate_mb=size_mb,
                feature=feature,
                present_fn=lambda mid=model_id: _rembg_present(mid),
                download_fn=lambda mid=model_id, lbl=label, sid=safe_id: _download_rembg_tracked(mid, sid, lbl),
            )
        )

    return entries


REGISTRY: list[ModelEntry] = _build_registry()
REGISTRY_BY_ID: dict[str, ModelEntry] = {m.id: m for m in REGISTRY}


def _update_model_status(model_id: str, status: str) -> None:
    with _progress_lock:
        _state.model_statuses[model_id] = status


def _set_progress(**kwargs) -> None:
    with _progress_lock:
        for key, value in kwargs.items():
            setattr(_state, key, value)


def get_download_progress() -> dict:
    with _progress_lock:
        return asdict(_state)


def _refresh_all_statuses() -> None:
    statuses: dict[str, str] = {}
    for entry in REGISTRY:
        if entry.present_fn():
            statuses[entry.id] = "present"
        else:
            existing = _state.model_statuses.get(entry.id)
            statuses[entry.id] = existing if existing in ("downloading", "error") else "missing"
    _set_progress(model_statuses=statuses)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    if path.is_file():
        return path.stat().st_size
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def get_models_status() -> dict:
    _refresh_all_statuses()
    models = []
    downloaded = 0
    total_estimate_mb = 0
    for entry in REGISTRY:
        present = entry.present_fn()
        if present:
            downloaded += 1
        total_estimate_mb += entry.size_estimate_mb
        with _progress_lock:
            status = _state.model_statuses.get(entry.id, "present" if present else "missing")
        models.append(
            {
                "id": entry.id,
                "name": entry.name,
                "size_estimate_mb": entry.size_estimate_mb,
                "feature": entry.feature,
                "status": "present" if present else status,
            }
        )

    disk_used_bytes = _dir_size(MODELS_DIR)
    progress = get_download_progress()

    return {
        "models": models,
        "downloaded_count": downloaded,
        "total_count": len(REGISTRY),
        "all_present": downloaded == len(REGISTRY),
        "disk_used_bytes": disk_used_bytes,
        "disk_used_mb": round(disk_used_bytes / (1024 * 1024), 1),
        "total_estimate_mb": total_estimate_mb,
        "models_dir": str(MODELS_DIR),
        "download_in_progress": progress["status"] == "downloading",
    }


def _download_direct(url: str, dest: Path, label: str, model_id: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _update_model_status(model_id, "downloading")
    _set_progress(
        current_model_id=model_id,
        current_model_name=label,
        bytes_downloaded=0,
        bytes_total=0,
        percent=0.0,
        message=f"Downloading {label}…",
    )
    print(f"Downloading {label} to {dest}...", flush=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            _set_progress(bytes_downloaded=downloaded, bytes_total=total_size, percent=pct)
        else:
            _set_progress(bytes_downloaded=downloaded)

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    sys.stderr.write("\n")
    print(f"{label} downloaded.", flush=True)
    _update_model_status(model_id, "present")


def _download_ultralytics_tracked(weight: str, model_id: str, label: str) -> None:
    _update_model_status(model_id, "downloading")
    _set_progress(
        current_model_id=model_id,
        current_model_name=label,
        bytes_downloaded=0,
        bytes_total=0,
        percent=0.0,
        message=f"Downloading {label}…",
    )
    _download_ultralytics(weight)
    _set_progress(percent=100.0)
    _update_model_status(model_id, "present")


def _download_rembg_tracked(model_id: str, registry_id: str, label: str) -> None:
    _update_model_status(registry_id, "downloading")
    _set_progress(
        current_model_id=registry_id,
        current_model_name=label,
        bytes_downloaded=0,
        bytes_total=0,
        percent=0.0,
        message=f"Downloading {label}…",
    )
    _download_rembg(model_id)
    _set_progress(percent=100.0)
    _update_model_status(registry_id, "present")


def _download_entry(entry: ModelEntry) -> None:
    if entry.present_fn():
        _update_model_status(entry.id, "present")
        return
    entry.download_fn()


def _run_downloads(entries: list[ModelEntry]) -> None:
    global _download_thread
    try:
        _set_progress(status="downloading", error=None, message="Starting downloads…")
        _refresh_all_statuses()
        pending = [e for e in entries if not e.present_fn()]
        total = len(pending)
        for idx, entry in enumerate(pending):
            _set_progress(
                message=f"Downloading {entry.name} ({idx + 1}/{total})…",
                percent=(idx / total * 100.0) if total else 100.0,
            )
            _download_entry(entry)
        _set_progress(status="complete", percent=100.0, message="All models downloaded.", current_model_id=None)
        _refresh_all_statuses()
    except Exception as exc:
        if _state.current_model_id:
            _update_model_status(_state.current_model_id, "error")
        _set_progress(status="error", error=str(exc), message=f"Download failed: {exc}")
    finally:
        with _progress_lock:
            global _download_thread
            _download_thread = None


def _start_background(entries: list[ModelEntry]) -> bool:
    global _download_thread
    with _progress_lock:
        if _download_thread is not None and _download_thread.is_alive():
            return False
        _download_thread = threading.Thread(
            target=_run_downloads,
            args=(entries,),
            daemon=True,
            name="model-download",
        )
        _download_thread.start()
        return True


def download_model(model_id: str) -> dict:
    entry = REGISTRY_BY_ID.get(model_id)
    if entry is None:
        raise ValueError(f"Unknown model: {model_id}")
    if entry.present_fn():
        _update_model_status(model_id, "present")
        return {"started": False, "message": "Model already present"}
    started = _start_background([entry])
    return {"started": started, "message": "Download started" if started else "Download already in progress"}


def download_all_models() -> dict:
    missing = [e for e in REGISTRY if not e.present_fn()]
    if not missing:
        return {"started": False, "message": "All models already downloaded"}
    started = _start_background(missing)
    return {
        "started": started,
        "message": "Download started" if started else "Download already in progress",
        "missing_count": len(missing),
    }


# --- On-demand ensure helpers (used by feature code) ---


def _track_on_demand(model_id: str, label: str, fn: Callable[[], None]) -> None:
    progress = get_download_progress()
    if progress["status"] != "downloading":
        _set_progress(status="downloading", error=None, message=f"Downloading {label}…")
    _update_model_status(model_id, "downloading")
    _set_progress(current_model_id=model_id, current_model_name=label)
    try:
        fn()
        _update_model_status(model_id, "present")
        current = get_download_progress()
        bg_active = _download_thread is not None and _download_thread.is_alive()
        if current["status"] != "downloading" or not bg_active:
            _set_progress(status="idle", message=None, current_model_id=None, percent=0.0)
    except Exception:
        _update_model_status(model_id, "error")
        _set_progress(status="error")
        raise


def ensure_lama_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not LAMA_ONNX.exists():
        _track_on_demand(
            "lama_onnx",
            "LaMa ONNX (Quick Remove)",
            lambda: _download_direct(LAMA_URL, LAMA_ONNX, "LaMa ONNX (Quick Remove)", "lama_onnx"),
        )
    return LAMA_ONNX


def ensure_lama_big_torch() -> Path:
    """PyTorch JIT big-LaMa (~200 MB) for AI Remove mode."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not LAMA_BIG_TORCH.exists():
        _track_on_demand(
            "lama_big",
            "LaMa big (AI Remove)",
            lambda: _download_direct(
                LAMA_BIG_TORCH_URL, LAMA_BIG_TORCH, "LaMa big (AI Remove)", "lama_big"
            ),
        )
    return LAMA_BIG_TORCH


def ensure_realesrgan_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not REALESRGAN_PTH.exists():
        _track_on_demand(
            "realesrgan",
            "Real-ESRGAN x4plus",
            lambda: _download_direct(
                REALESRGAN_URL, REALESRGAN_PTH, "Real-ESRGAN x4plus", "realesrgan"
            ),
        )
    return REALESRGAN_PTH
