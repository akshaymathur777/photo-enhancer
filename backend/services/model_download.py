"""Model download registry, progress tracking, and on-demand ensure helpers."""

from __future__ import annotations

import logging
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from paths import get_logs_dir, get_models_dir

DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SEC = 2.0

_progress_lock = threading.Lock()
_download_thread: threading.Thread | None = None
_logger_instance: logging.Logger | None = None


def _get_download_logger() -> logging.Logger:
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "download.log"

    logger = logging.getLogger("photo_enhancer.download")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    _logger_instance = logger
    return logger


def _log(level: int, message: str, *, exc: BaseException | None = None) -> None:
    logger = _get_download_logger()
    if exc is not None:
        logger.log(level, "%s\n%s", message, traceback.format_exc())
    else:
        logger.log(level, message)


def _models_dir() -> Path:
    return get_models_dir()


def _lama_onnx() -> Path:
    return _models_dir() / "lama_fp32.onnx"


def _lama_big_torch() -> Path:
    return _models_dir() / "big-lama.pt"


def _sd_inpaint_dir() -> Path:
    return _models_dir() / "sd15-inpainting"


def _sdxl_inpaint_dir() -> Path:
    return _models_dir() / "sdxl-inpainting"


def _sdxl_juggernaut_dir() -> Path:
    return _models_dir() / "sdxl-inpainting-juggernaut"


def _realesrgan_pth() -> Path:
    return _models_dir() / "RealESRGAN_x4plus.pth"


LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
LAMA_BIG_TORCH_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
# Public ungated SD1.5 inpaint (Stability SD2 requires HF auth / license accept).
SD_INPAINT_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
# Official Diffusers SDXL inpaint (ungated). Photoreal Juggernaut-X preferred when present.
SDXL_INPAINT_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SDXL_JUGGERNAUT_REPO = "GraydientPlatformAPI/juggernaut-x-inpainting-sdxl"


def _sd_inpaint_present() -> bool:
    return (_sd_inpaint_dir() / "model_index.json").is_file()


def _sdxl_inpaint_present() -> bool:
    return (_sdxl_inpaint_dir() / "model_index.json").is_file() or (
        _sdxl_juggernaut_dir() / "model_index.json"
    ).is_file()


def resolve_sdxl_inpaint_dir() -> Path:
    """Prefer photoreal Juggernaut-X inpaint when downloaded; else official SDXL."""
    jug = _sdxl_juggernaut_dir()
    if (jug / "model_index.json").is_file():
        return jug
    return _sdxl_inpaint_dir()


def _repair_sdxl_inpaint_snapshot(model_dir: Path) -> None:
    """Fix mislabeled community SDXL-inpaint snapshots (base class / 4-ch UNet config)."""
    import json

    index_path = model_dir / "model_index.json"
    unet_cfg_path = model_dir / "unet" / "config.json"
    try:
        if index_path.is_file():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            if data.get("_class_name") == "StableDiffusionXLPipeline":
                data["_class_name"] = "StableDiffusionXLInpaintPipeline"
                index_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        if unet_cfg_path.is_file():
            cfg = json.loads(unet_cfg_path.read_text(encoding="utf-8"))
            if int(cfg.get("in_channels", 4)) != 4:
                return
            weight = model_dir / "unet" / "diffusion_pytorch_model.safetensors"
            weight_fp16 = model_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors"
            wpath = weight if weight.is_file() else weight_fp16
            if not wpath.is_file():
                return
            try:
                from safetensors import safe_open

                with safe_open(str(wpath), framework="pt", device="cpu") as f:
                    shape = f.get_tensor("conv_in.weight").shape
                if len(shape) >= 2 and int(shape[1]) == 9:
                    cfg["in_channels"] = 9
                    unet_cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass

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
    error_kind: str | None = None  # network | permission | disk | ssl | unknown
    message: str | None = None


_state = ProgressState()


def _u2net_home() -> Path:
    import os

    return Path(os.environ["U2NET_HOME"])


def _ultralytics_home() -> Path:
    import os

    return Path(os.environ["ULTRALYTICS_HOME"])


def _ultralytics_weight_candidates(name: str) -> list[Path]:
    home = _ultralytics_home()
    models = _models_dir()
    candidates = [
        home / name,
        home / "weights" / name,
        models / name,
        Path.cwd() / name,
        Path.cwd() / "weights" / name,
    ]
    try:
        from ultralytics.utils import SETTINGS

        weights_dir = Path(str(SETTINGS.get("weights_dir", "")))
        if weights_dir.name:
            candidates.extend([weights_dir / name, weights_dir.resolve() / name])
    except Exception:
        pass
    # Preserve order while deduplicating
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _find_ultralytics_weight(name: str) -> Path | None:
    for candidate in _ultralytics_weight_candidates(name):
        if candidate.is_file():
            return candidate
    return None


def _rembg_present(model_id: str) -> bool:
    return (_u2net_home() / f"{model_id}.onnx").is_file()


def _classify_error(exc: BaseException) -> tuple[str, str]:
    """Return (user_message, error_kind)."""
    raw = str(exc).strip() or exc.__class__.__name__
    lowered = raw.lower()

    if isinstance(exc, PermissionError) or "permission" in lowered or "access is denied" in lowered:
        return (
            "Cannot write to the models folder. Install Photo Enhancer in a user-writable location "
            f"or check permissions for {_models_dir()}.",
            "permission",
        )
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return (
            "Not enough disk space to download models. Free at least 10 GB and try again.",
            "disk",
        )
    if "no space left" in lowered or "disk full" in lowered:
        return (
            "Not enough disk space to download models. Free at least 10 GB and try again.",
            "disk",
        )
    if isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, "reason", None), ssl.SSLError):
        return (
            "Secure connection failed (SSL). Check antivirus/firewall settings or try again on a trusted network.",
            "ssl",
        )
    if "certificate" in lowered or "ssl" in lowered or "certifi" in lowered:
        return (
            "Secure connection failed (SSL). Check antivirus/firewall settings or try again on a trusted network.",
            "ssl",
        )
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)) or "timed out" in lowered:
        return (
            "Network error while downloading models. Check your internet connection, firewall, or proxy settings.",
            "network",
        )
    if "404" in lowered or "not found" in lowered:
        return (
            f"Download URL is unavailable ({raw}). Check for an app update or try again later.",
            "network",
        )
    return (f"Download failed: {raw}", "unknown")


# Late import for ssl used in _classify_error
import ssl  # noqa: E402


def _download_rembg(model_id: str) -> None:
    from rembg.sessions import sessions

    session_cls = sessions.get(model_id)
    if session_cls is None:
        raise ValueError(f"Unknown rembg model: {model_id}")
    last_exc: BaseException | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            session_cls.download_models()
            if not _rembg_present(model_id):
                raise RuntimeError(f"rembg finished but {model_id}.onnx was not saved to {_u2net_home()}")
            return
        except Exception as exc:
            last_exc = exc
            _log(logging.WARNING, f"rembg {model_id} attempt {attempt}/{DOWNLOAD_RETRIES} failed: {exc}", exc=exc)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_BACKOFF_SEC * attempt)
    assert last_exc is not None
    raise last_exc


def _relocate_ultralytics_weight(name: str) -> Path:
    """Ensure weight lives under our ultralytics models directory."""
    dest = _ultralytics_home() / name
    if dest.is_file():
        return dest

    found = _find_ultralytics_weight(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} was not found after download. Expected location: {dest}"
        )

    if found.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(found), str(dest))
        _log(logging.INFO, f"Moved {name} from {found} to {dest}")

    return dest


def _download_ultralytics(name: str) -> None:
    weights_dir = _ultralytics_home()
    weights_dir.mkdir(parents=True, exist_ok=True)

    dest = weights_dir / name
    if dest.is_file():
        return

    last_exc: BaseException | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            if name.startswith("sam"):
                from ultralytics.models.sam import SAM2Predictor

                SAM2Predictor(
                    overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=name)
                )
            else:
                from ultralytics import YOLO

                YOLO(name)
            return _relocate_ultralytics_weight(name)
        except Exception as exc:
            last_exc = exc
            _log(
                logging.WARNING,
                f"ultralytics {name} attempt {attempt}/{DOWNLOAD_RETRIES} failed: {exc}",
                exc=exc,
            )
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_BACKOFF_SEC * attempt)
    assert last_exc is not None
    raise last_exc


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
            present_fn=lambda: _lama_onnx().is_file(),
            download_fn=lambda: _download_direct(LAMA_URL, _lama_onnx(), "LaMa ONNX (Quick Remove)", "lama_onnx"),
        ),
        ModelEntry(
            id="lama_big",
            name="LaMa big (AI Remove fallback)",
            size_estimate_mb=196,
            feature="Object removal (fallback)",
            present_fn=lambda: _lama_big_torch().is_file(),
            download_fn=lambda: _download_direct(
                LAMA_BIG_TORCH_URL, _lama_big_torch(), "LaMa big (AI Remove fallback)", "lama_big"
            ),
        ),
        ModelEntry(
            id="sdxl_inpaint",
            name="SDXL Inpaint (AI Remove / Best Quality)",
            size_estimate_mb=7000,
            feature="Object removal (generative AI, CUDA)",
            present_fn=_sdxl_inpaint_present,
            download_fn=lambda: _download_sdxl_inpaint(),
        ),
        ModelEntry(
            id="sd_inpaint",
            name="SD1.5 Inpaint (AI Remove fallback)",
            size_estimate_mb=4200,
            feature="Object removal (generative fallback)",
            present_fn=_sd_inpaint_present,
            download_fn=lambda: _download_sd_inpaint(),
        ),
        ModelEntry(
            id="realesrgan",
            name="Real-ESRGAN x4plus",
            size_estimate_mb=64,
            feature="Upscale 2× / 4×",
            present_fn=lambda: _realesrgan_pth().is_file(),
            download_fn=lambda: _download_direct(
                REALESRGAN_URL, _realesrgan_pth(), "Real-ESRGAN x4plus", "realesrgan"
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

    disk_used_bytes = _dir_size(_models_dir())
    progress = get_download_progress()

    return {
        "models": models,
        "downloaded_count": downloaded,
        "total_count": len(REGISTRY),
        "all_present": downloaded == len(REGISTRY),
        "disk_used_bytes": disk_used_bytes,
        "disk_used_mb": round(disk_used_bytes / (1024 * 1024), 1),
        "total_estimate_mb": total_estimate_mb,
        "models_dir": str(_models_dir()),
        "logs_dir": str(get_logs_dir()),
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
        error=None,
        error_kind=None,
        message=f"Downloading {label}…",
    )
    _log(logging.INFO, f"Starting direct download: {label} -> {dest} from {url}")
    print(f"Downloading {label} to {dest}...", flush=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            _set_progress(bytes_downloaded=downloaded, bytes_total=total_size, percent=pct)
        else:
            _set_progress(bytes_downloaded=downloaded)

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc: BaseException | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            if tmp.is_file():
                tmp.unlink()
            urllib.request.urlretrieve(url, tmp, reporthook=_report)
            if not tmp.is_file() or tmp.stat().st_size == 0:
                raise RuntimeError(f"Download produced an empty file for {label}")
            tmp.replace(dest)
            sys.stderr.write("\n")
            print(f"{label} downloaded.", flush=True)
            _log(logging.INFO, f"Completed direct download: {label} ({dest.stat().st_size} bytes)")
            _update_model_status(model_id, "present")
            return
        except Exception as exc:
            last_exc = exc
            _log(
                logging.WARNING,
                f"Direct download {label} attempt {attempt}/{DOWNLOAD_RETRIES} failed: {exc}",
                exc=exc,
            )
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_BACKOFF_SEC * attempt)

    assert last_exc is not None
    raise last_exc


def _download_ultralytics_tracked(weight: str, model_id: str, label: str) -> None:
    _update_model_status(model_id, "downloading")
    _set_progress(
        current_model_id=model_id,
        current_model_name=label,
        bytes_downloaded=0,
        bytes_total=0,
        percent=0.0,
        error=None,
        error_kind=None,
        message=f"Downloading {label}…",
    )
    _log(logging.INFO, f"Starting ultralytics download: {label} ({weight})")
    path = _download_ultralytics(weight)
    _set_progress(percent=100.0, bytes_downloaded=path.stat().st_size, bytes_total=path.stat().st_size)
    _update_model_status(model_id, "present")
    _log(logging.INFO, f"Completed ultralytics download: {label} at {path}")


def _download_rembg_tracked(model_id: str, registry_id: str, label: str) -> None:
    _update_model_status(registry_id, "downloading")
    _set_progress(
        current_model_id=registry_id,
        current_model_name=label,
        bytes_downloaded=0,
        bytes_total=0,
        percent=0.0,
        error=None,
        error_kind=None,
        message=f"Downloading {label}…",
    )
    _log(logging.INFO, f"Starting rembg download: {label} ({model_id})")
    _download_rembg(model_id)
    _set_progress(percent=100.0)
    _update_model_status(registry_id, "present")
    _log(logging.INFO, f"Completed rembg download: {label}")


def _download_entry(entry: ModelEntry) -> None:
    if entry.present_fn():
        _update_model_status(entry.id, "present")
        return
    entry.download_fn()
    if not entry.present_fn():
        raise RuntimeError(f"{entry.name} download finished but the model file is still missing")


def _run_downloads(entries: list[ModelEntry]) -> None:
    global _download_thread
    try:
        _models_dir().mkdir(parents=True, exist_ok=True)
        _set_progress(status="downloading", error=None, error_kind=None, message="Starting downloads…")
        _refresh_all_statuses()
        pending = [e for e in entries if not e.present_fn()]
        total = len(pending)
        for idx, entry in enumerate(pending):
            _set_progress(
                message=f"Downloading {entry.name} ({idx + 1}/{total})…",
                percent=(idx / total * 100.0) if total else 100.0,
            )
            _download_entry(entry)
        still_missing = [e.name for e in REGISTRY if not e.present_fn()]
        if still_missing:
            raise RuntimeError(f"Some models are still missing after download: {', '.join(still_missing[:3])}")
        _set_progress(
            status="complete",
            percent=100.0,
            message="All models downloaded.",
            current_model_id=None,
            error=None,
            error_kind=None,
        )
        _refresh_all_statuses()
        _log(logging.INFO, "Background download batch completed successfully")
    except Exception as exc:
        user_msg, kind = _classify_error(exc)
        if _state.current_model_id:
            _update_model_status(_state.current_model_id, "error")
        _set_progress(status="error", error=user_msg, error_kind=kind, message=f"Download failed: {user_msg}")
        _log(logging.ERROR, f"Background download failed: {user_msg}", exc=exc)
    finally:
        with _progress_lock:
            global _download_thread
            _download_thread = None


def _start_background(entries: list[ModelEntry]) -> bool:
    global _download_thread
    with _progress_lock:
        if _download_thread is not None and _download_thread.is_alive():
            return False
        if _state.status == "error":
            _set_progress(status="idle", error=None, error_kind=None, message=None)
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
        _set_progress(status="downloading", error=None, error_kind=None, message=f"Downloading {label}…")
    _update_model_status(model_id, "downloading")
    _set_progress(current_model_id=model_id, current_model_name=label)
    try:
        fn()
        _update_model_status(model_id, "present")
        current = get_download_progress()
        bg_active = _download_thread is not None and _download_thread.is_alive()
        if current["status"] != "downloading" or not bg_active:
            _set_progress(status="idle", message=None, current_model_id=None, percent=0.0)
    except Exception as exc:
        user_msg, kind = _classify_error(exc)
        _update_model_status(model_id, "error")
        _set_progress(status="error", error=user_msg, error_kind=kind, message=user_msg)
        _log(logging.ERROR, f"On-demand download failed for {label}: {user_msg}", exc=exc)
        raise RuntimeError(user_msg) from exc


def ensure_lama_model() -> Path:
    _models_dir().mkdir(parents=True, exist_ok=True)
    path = _lama_onnx()
    if not path.exists():
        _track_on_demand(
            "lama_onnx",
            "LaMa ONNX (Quick Remove)",
            lambda: _download_direct(LAMA_URL, path, "LaMa ONNX (Quick Remove)", "lama_onnx"),
        )
    return path


def ensure_lama_big_torch() -> Path:
    """PyTorch JIT big-LaMa (~200 MB) — Fast/fallback when generative inpaint fails."""
    _models_dir().mkdir(parents=True, exist_ok=True)
    path = _lama_big_torch()
    if not path.exists():
        _track_on_demand(
            "lama_big",
            "LaMa big (AI Remove fallback)",
            lambda: _download_direct(
                LAMA_BIG_TORCH_URL, path, "LaMa big (AI Remove fallback)", "lama_big"
            ),
        )
    return path


def _download_sd_inpaint() -> None:
    """Download Stable Diffusion 1.5 Inpainting weights (~4 GB) into the models folder."""
    from huggingface_hub import snapshot_download

    dest = _sd_inpaint_dir()
    dest.mkdir(parents=True, exist_ok=True)
    _update_model_status("sd_inpaint", "downloading")
    _set_progress(
        current_model_id="sd_inpaint",
        current_model_name="SD1.5 Inpaint (AI Remove fallback)",
        bytes_downloaded=0,
        bytes_total=0,
        percent=0.0,
        message=(
            "Downloading SD1.5 Inpaint (~4 GB). Used when SDXL is unavailable; CPU works but is slow."
        ),
    )
    _log(logging.INFO, f"Starting HuggingFace snapshot: {SD_INPAINT_REPO} -> {dest}")

    last_exc: BaseException | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            # token=False avoids broken/expired local HF oauth tokens causing 401s.
            snapshot_download(
                repo_id=SD_INPAINT_REPO,
                local_dir=str(dest),
                token=False,
            )
            if not _sd_inpaint_present():
                raise RuntimeError(
                    f"SD1.5 Inpaint download finished but model_index.json is missing under {dest}"
                )
            _update_model_status("sd_inpaint", "present")
            _set_progress(percent=100.0, message="SD1.5 Inpaint downloaded.")
            _log(logging.INFO, f"Completed SD1.5 Inpaint download at {dest}")
            return
        except Exception as exc:
            last_exc = exc
            _log(
                logging.WARNING,
                f"SD1.5 Inpaint download attempt {attempt}/{DOWNLOAD_RETRIES} failed: {exc}",
                exc=exc,
            )
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_BACKOFF_SEC * attempt)
    assert last_exc is not None
    raise last_exc


def ensure_sd_inpaint() -> Path:
    """Stable Diffusion 1.5 Inpainting (~4 GB) — generative fallback when SDXL is unavailable."""
    _models_dir().mkdir(parents=True, exist_ok=True)
    dest = _sd_inpaint_dir()
    if not _sd_inpaint_present():
        _track_on_demand(
            "sd_inpaint",
            "SD1.5 Inpaint (AI Remove fallback)",
            _download_sd_inpaint,
        )
    return dest


def _download_sdxl_inpaint() -> None:
    """Download SDXL Inpainting weights. Prefer photoreal Juggernaut-X; else official Diffusers."""
    from huggingface_hub import snapshot_download

    # Prefer Juggernaut when downloading fresh — stronger photoreal coastal fills.
    dest = _sdxl_juggernaut_dir()
    repo = SDXL_JUGGERNAUT_REPO
    label = "SDXL Juggernaut Inpaint (AI Remove / Best Quality)"
    dest.mkdir(parents=True, exist_ok=True)
    _update_model_status("sdxl_inpaint", "downloading")
    _set_progress(
        current_model_id="sdxl_inpaint",
        current_model_name=label,
        bytes_downloaded=0,
        bytes_total=0,
        percent=0.0,
        message=(
            "Downloading SDXL Inpaint (~7 GB, photoreal). NVIDIA CUDA required for Best Quality; "
            "falls back to SD1.5 / LaMa if unavailable."
        ),
    )
    _log(logging.INFO, f"Starting HuggingFace snapshot: {repo} -> {dest}")

    last_exc: BaseException | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(dest),
                token=False,
                ignore_patterns=["*.bin", "*.onnx", "*.msgpack", "*.h5", "*.ot"],
            )
            if (dest / "model_index.json").is_file():
                _repair_sdxl_inpaint_snapshot(dest)
                _update_model_status("sdxl_inpaint", "present")
                _set_progress(percent=100.0, message="SDXL Inpaint downloaded.")
                _log(logging.INFO, f"Completed SDXL Inpaint download at {dest}")
                return
            raise RuntimeError(f"SDXL download finished but model_index.json is missing under {dest}")
        except Exception as exc:
            last_exc = exc
            _log(
                logging.WARNING,
                f"SDXL Juggernaut download attempt {attempt}/{DOWNLOAD_RETRIES} failed: {exc}",
                exc=exc,
            )
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_BACKOFF_SEC * attempt)

    # Fallback: official Diffusers SDXL inpaint.
    dest = _sdxl_inpaint_dir()
    dest.mkdir(parents=True, exist_ok=True)
    _set_progress(
        message=(
            "Juggernaut unavailable — downloading official SDXL Inpaint (~7 GB)…"
        ),
    )
    _log(logging.INFO, f"Falling back to HuggingFace snapshot: {SDXL_INPAINT_REPO} -> {dest}")
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=SDXL_INPAINT_REPO,
                local_dir=str(dest),
                token=False,
                ignore_patterns=["*.bin", "*.onnx", "*.msgpack", "*.h5", "*.ot"],
            )
            if not (dest / "model_index.json").is_file():
                raise RuntimeError(
                    f"SDXL Inpaint download finished but model_index.json is missing under {dest}"
                )
            _update_model_status("sdxl_inpaint", "present")
            _set_progress(percent=100.0, message="SDXL Inpaint downloaded.")
            _log(logging.INFO, f"Completed official SDXL Inpaint download at {dest}")
            return
        except Exception as exc:
            last_exc = exc
            _log(
                logging.WARNING,
                f"Official SDXL Inpaint download attempt {attempt}/{DOWNLOAD_RETRIES} failed: {exc}",
                exc=exc,
            )
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_BACKOFF_SEC * attempt)
    assert last_exc is not None
    raise last_exc


def ensure_sdxl_inpaint() -> Path:
    """SDXL Inpainting (~7 GB) for generative AI Remove / Best Quality on CUDA."""
    _models_dir().mkdir(parents=True, exist_ok=True)
    if not _sdxl_inpaint_present():
        _track_on_demand(
            "sdxl_inpaint",
            "SDXL Inpaint (AI Remove / Best Quality)",
            _download_sdxl_inpaint,
        )
    return resolve_sdxl_inpaint_dir()


def ensure_realesrgan_model() -> Path:
    _models_dir().mkdir(parents=True, exist_ok=True)
    path = _realesrgan_pth()
    if not path.exists():
        _track_on_demand(
            "realesrgan",
            "Real-ESRGAN x4plus",
            lambda: _download_direct(REALESRGAN_URL, path, "Real-ESRGAN x4plus", "realesrgan"),
        )
    return path
