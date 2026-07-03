"""Shared upscale output limits and scale selection."""

from __future__ import annotations

# Default 150 MP: supports ~12 MP sources at 4× while staying within typical 16 GB RAM budgets.
DEFAULT_MAX_OUTPUT_MP = 150_000_000
MIN_MAX_OUTPUT_MP = 50_000_000
MAX_MAX_OUTPUT_MP = 400_000_000
ALLOWED_SCALES = (2, 4)


def validate_max_output_mp(value: int) -> int:
    if value < MIN_MAX_OUTPUT_MP:
        raise ValueError(
            f"max_output_mp must be at least {MIN_MAX_OUTPUT_MP // 1_000_000} MP "
            f"(got {value // 1_000_000} MP)."
        )
    if value > MAX_MAX_OUTPUT_MP:
        raise ValueError(
            f"max_output_mp must be at most {MAX_MAX_OUTPUT_MP // 1_000_000} MP "
            f"(got {value // 1_000_000} MP)."
        )
    return value


def resolve_max_output_mp(override: int | None = None) -> int:
    if override is not None:
        return validate_max_output_mp(override)
    from services.settings import get_max_output_mp

    return get_max_output_mp()


def output_pixels(width: int, height: int, scale: int) -> int:
    return width * height * scale * scale


def max_allowed_scale(
    width: int, height: int, max_output_mp: int | None = None
) -> int | None:
    """Highest scale (4 or 2) within the output limit, or None if even 2x is too large."""
    limit = resolve_max_output_mp(max_output_mp)
    for scale in sorted(ALLOWED_SCALES, reverse=True):
        if output_pixels(width, height, scale) <= limit:
            return scale
    return None


def clamp_scale(
    width: int, height: int, requested: int, max_output_mp: int | None = None
) -> tuple[int, str | None]:
    """
    Return the scale to use and an optional user-facing warning.
    Auto-caps downward (4 -> 2) when the requested scale would exceed the limit.
    """
    limit = resolve_max_output_mp(max_output_mp)
    limit_mp = limit // 1_000_000

    if requested not in ALLOWED_SCALES:
        requested = 2

    allowed = max_allowed_scale(width, height, limit)
    if allowed is None:
        src_mp = (width * height) / 1_000_000
        raise ValueError(
            f"Image is already {src_mp:.0f}MP — even 2× upscale would exceed the "
            f"{limit_mp}MP output limit. Try cropping, lowering the limit preset, or a smaller photo."
        )

    if requested <= allowed:
        warning = None
        if output_pixels(width, height, requested) > limit * 0.8:
            out_mp = output_pixels(width, height, requested) / 1_000_000
            warning = (
                f"Output will be ~{out_mp:.0f}MP — processing may use significant memory."
            )
        return requested, warning

    out_mp = output_pixels(width, height, allowed) / 1_000_000
    return allowed, (
        f"Requested {requested}× would exceed the {limit_mp}MP limit; "
        f"used {allowed}× instead (~{out_mp:.0f}MP output)."
    )
