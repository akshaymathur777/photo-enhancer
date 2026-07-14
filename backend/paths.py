"""Resolve application paths for dev, launcher, and PyInstaller frozen builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_project_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_bundle_root() -> Path:
    if is_frozen():
        return Path(sys._MEIPASS)
    return get_project_root()


def get_backend_dir() -> Path:
    bundled = get_bundle_root() / "backend"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parent


def get_frontend_dist() -> Path:
    return get_bundle_root() / "frontend" / "dist"


def get_app_data_dir() -> Path:
    """User-writable Photo Enhancer data root (never the read-only install dir)."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "PhotoEnhancer"
    return get_project_root()


def get_models_dir() -> Path:
    """User-writable model cache (outside read-only PyInstaller bundle)."""
    if is_frozen():
        return get_app_data_dir() / "models"
    return get_backend_dir() / "models"


def get_logs_dir() -> Path:
    """User-writable log directory for diagnostics."""
    return get_app_data_dir() / "logs"


def get_env_file() -> Path:
    for candidate in (get_project_root() / ".env", get_bundle_root() / ".env"):
        if candidate.is_file():
            return candidate
    return get_project_root() / ".env"


def get_settings_path() -> Path:
    """User-writable app settings (upscale limits, etc.)."""
    return get_app_data_dir() / "settings.json"


def get_app_icon() -> Path | None:
    for candidate in (
        get_project_root() / "assets" / "icon.ico",
        get_bundle_root() / "assets" / "icon.ico",
    ):
        if candidate.is_file():
            return candidate
    return None


def _force_env(key: str, value: str) -> None:
    """Always point third-party libraries at our user-writable directories."""
    os.environ[key] = value


def _configure_ssl_certificates() -> None:
    """PyInstaller builds often lack CA bundles; urllib/pooch/rembg need certifi."""
    try:
        import certifi

        bundle = certifi.where()
        _force_env("SSL_CERT_FILE", bundle)
        _force_env("REQUESTS_CA_BUNDLE", bundle)
    except ImportError:
        pass


def _configure_ultralytics(weights_dir: Path, config_root: Path) -> None:
    """Keep Ultralytics weights and settings under PhotoEnhancer AppData."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    # Ultralytics reads YOLO_CONFIG_DIR/Ultralytics/settings.json
    _force_env("YOLO_CONFIG_DIR", str(config_root))
    # Alias used in docs / some tooling
    _force_env("ULTRALYTICS_CONFIG_DIR", str(config_root / "Ultralytics"))

    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update(weights_dir=str(weights_dir))
    except Exception:
        # ultralytics not imported yet on first boot — settings.json is written on first use
        settings_file = config_root / "Ultralytics" / "settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        if not settings_file.is_file():
            import json

            settings_file.write_text(
                json.dumps(
                    {
                        "settings_version": "0.0.6",
                        "datasets_dir": str(config_root / "datasets"),
                        "weights_dir": str(weights_dir),
                        "runs_dir": str(config_root / "runs"),
                        "sync": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


def configure_runtime_paths() -> Path:
    """Set third-party cache directories before ML libraries import models."""
    models = get_models_dir()
    models.mkdir(parents=True, exist_ok=True)
    get_logs_dir().mkdir(parents=True, exist_ok=True)

    u2net_home = models / "u2net"
    ultralytics_home = models / "ultralytics"
    torch_home = models / "torch"
    for path in (u2net_home, ultralytics_home, torch_home):
        path.mkdir(parents=True, exist_ok=True)

    _force_env("PHOTO_ENHANCER_MODELS", str(models))
    _force_env("U2NET_HOME", str(u2net_home))
    _force_env("ULTRALYTICS_HOME", str(ultralytics_home))
    _force_env("TORCH_HOME", str(torch_home))
    _force_env("XDG_CACHE_HOME", str(get_app_data_dir() / "cache"))

    _configure_ultralytics(ultralytics_home, get_app_data_dir())
    _configure_ssl_certificates()

    return models
