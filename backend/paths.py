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


def get_models_dir() -> Path:
    """User-writable model cache (outside read-only PyInstaller bundle)."""
    if is_frozen():
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "PhotoEnhancer" / "models"
        return get_project_root() / "models"
    return get_backend_dir() / "models"


def get_env_file() -> Path:
    for candidate in (get_project_root() / ".env", get_bundle_root() / ".env"):
        if candidate.is_file():
            return candidate
    return get_project_root() / ".env"


def get_settings_path() -> Path:
    """User-writable app settings (upscale limits, etc.)."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "PhotoEnhancer" / "settings.json"
    return get_project_root() / "settings.json"


def get_app_icon() -> Path | None:
    for candidate in (
        get_project_root() / "assets" / "icon.ico",
        get_bundle_root() / "assets" / "icon.ico",
    ):
        if candidate.is_file():
            return candidate
    return None


def configure_runtime_paths() -> Path:
    """Set third-party cache directories before ML libraries import models."""
    models = get_models_dir()
    models.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PHOTO_ENHANCER_MODELS", str(models))
    os.environ.setdefault("U2NET_HOME", str(models / "u2net"))

    ultralytics_home = models / "ultralytics"
    ultralytics_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_HOME", str(ultralytics_home))

    torch_home = models / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    return models
