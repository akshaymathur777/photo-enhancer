"""Persistent app settings (upscale limits, etc.)."""

from __future__ import annotations

import json
import os

from paths import get_settings_path
from services.upscale_limits import (
    DEFAULT_MAX_OUTPUT_MP,
    MAX_MAX_OUTPUT_MP,
    MIN_MAX_OUTPUT_MP,
    validate_max_output_mp,
)

UPSCALE_PRESETS = (
    {"id": "conservative", "label": "Conservative", "megapixels": 100},
    {"id": "standard", "label": "Standard", "megapixels": 150},
    {"id": "high", "label": "High", "megapixels": 200},
)


def _read_env_max_output_mp() -> int | None:
    raw = os.environ.get("MAX_OUTPUT_MP")
    if not raw or not raw.strip():
        return None
    try:
        megapixels = int(raw.strip())
    except ValueError:
        return None
    return validate_max_output_mp(megapixels * 1_000_000)


def _load_settings_file() -> dict:
    path = get_settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_settings_file(data: dict) -> None:
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def get_max_output_mp() -> int:
    """Effective upscale limit in pixels. Priority: settings file > env > default."""
    data = _load_settings_file()
    if "max_output_mp" in data:
        try:
            return validate_max_output_mp(int(data["max_output_mp"]))
        except (TypeError, ValueError):
            pass
    env_val = _read_env_max_output_mp()
    if env_val is not None:
        return env_val
    return DEFAULT_MAX_OUTPUT_MP


def set_max_output_mp(pixels: int) -> int:
    validated = validate_max_output_mp(pixels)
    data = _load_settings_file()
    data["max_output_mp"] = validated
    _save_settings_file(data)
    return validated


def set_max_output_mp_from_megapixels(megapixels: int) -> int:
    return set_max_output_mp(megapixels * 1_000_000)


def get_settings_payload() -> dict:
    mp_pixels = get_max_output_mp()
    return {
        "max_output_megapixels": mp_pixels // 1_000_000,
        "min_max_output_megapixels": MIN_MAX_OUTPUT_MP // 1_000_000,
        "max_max_output_megapixels": MAX_MAX_OUTPUT_MP // 1_000_000,
        "default_max_output_megapixels": DEFAULT_MAX_OUTPUT_MP // 1_000_000,
        "presets": list(UPSCALE_PRESETS),
        "settings_path": str(get_settings_path()),
    }
