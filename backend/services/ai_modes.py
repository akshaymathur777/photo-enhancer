"""AI processing mode presets — quality/speed trade-offs."""

from __future__ import annotations

from typing import Any

AI_MODES: dict[str, dict[str, Any]] = {
    "best_quality": {
        "id": "best_quality",
        "name": "Best Quality",
        "description": "Heaviest models — best results, slower processing.",
        "bg_model": "massive",
        "removal": "ai",
        "upscale_max_scale": 4,
        "upscale_prefer": 4,
        "denoise_strength": 12,
        "sharpen_amount": 1.8,
    },
    "balanced": {
        "id": "balanced",
        "name": "Balanced",
        "description": "Default middle ground between speed and quality.",
        "bg_model": "general",
        "removal": "ai",
        "upscale_max_scale": 4,
        "upscale_prefer": 2,
        "denoise_strength": 10,
        "sharpen_amount": 1.5,
    },
    "fast": {
        "id": "fast",
        "name": "Fast",
        "description": "Lite models and quicker inpainting.",
        "bg_model": "lite",
        "removal": "quick",
        "upscale_max_scale": 2,
        "upscale_prefer": 2,
        "denoise_strength": 8,
        "sharpen_amount": 1.2,
    },
}

DEFAULT_MODE = "balanced"


def normalize_mode(mode: str | None) -> str:
    if mode and mode in AI_MODES:
        return mode
    return DEFAULT_MODE


def get_mode_config(mode: str | None) -> dict[str, Any]:
    return AI_MODES[normalize_mode(mode)]


def list_modes() -> list[dict[str, Any]]:
    return [
        {
            "id": m["id"],
            "name": m["name"],
            "description": m["description"],
        }
        for m in AI_MODES.values()
    ]


def resolve_bg_model(mode: str | None, explicit_model: str | None = None) -> str:
    if explicit_model:
        return explicit_model
    return get_mode_config(mode)["bg_model"]


def resolve_removal_mode(mode: str | None) -> str:
    return get_mode_config(mode)["removal"]


def resolve_inpaint_mode(global_mode: str | None, explicit: str | None = None) -> str:
    """Map UI/global presets to inpaint backend: ai | quick."""
    if explicit is not None:
        explicit = str(explicit).strip().lower()
    if explicit in ("ai", "quick"):
        return explicit
    preset = resolve_removal_mode(global_mode)
    if preset == "quick":
        return "quick"
    return "ai"


def clamp_scale_for_mode(
    width: int,
    height: int,
    requested: int,
    mode: str | None,
    max_output_mp: int | None = None,
) -> tuple[int, str | None]:
    """Apply mode-specific upscale caps on top of global MP limits."""
    from services.upscale_limits import clamp_scale

    effective, warning = clamp_scale(width, height, requested, max_output_mp)
    max_for_mode = get_mode_config(mode)["upscale_max_scale"]
    if effective > max_for_mode:
        out_mp = (width * height * max_for_mode * max_for_mode) / 1_000_000
        mode_warning = (
            f"Fast/Balanced mode caps upscale at {max_for_mode}× "
            f"(~{out_mp:.0f}MP output)."
        )
        combined = f"{warning} {mode_warning}" if warning else mode_warning
        return max_for_mode, combined
    return effective, warning
