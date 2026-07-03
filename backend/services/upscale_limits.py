"""Shared upscale output limits and scale selection."""

MAX_OUTPUT_MP = 100_000_000  # 100 megapixels
ALLOWED_SCALES = (2, 4)


def output_pixels(width: int, height: int, scale: int) -> int:
    return width * height * scale * scale


def max_allowed_scale(width: int, height: int) -> int | None:
    """Highest scale (4 or 2) that stays within MAX_OUTPUT_MP, or None if even 2x is too large."""
    for scale in sorted(ALLOWED_SCALES, reverse=True):
        if output_pixels(width, height, scale) <= MAX_OUTPUT_MP:
            return scale
    return None


def clamp_scale(width: int, height: int, requested: int) -> tuple[int, str | None]:
    """
    Return the scale to use and an optional user-facing warning.
    Auto-caps downward (4 -> 2) when the requested scale would exceed the limit.
    """
    if requested not in ALLOWED_SCALES:
        requested = 2

    allowed = max_allowed_scale(width, height)
    if allowed is None:
        src_mp = (width * height) / 1_000_000
        raise ValueError(
            f"Image is already {src_mp:.0f}MP — even 2× upscale would exceed the "
            f"{MAX_OUTPUT_MP // 1_000_000}MP output limit. Try cropping or a smaller photo."
        )

    if requested <= allowed:
        warning = None
        if output_pixels(width, height, requested) > MAX_OUTPUT_MP * 0.8:
            out_mp = output_pixels(width, height, requested) / 1_000_000
            warning = (
                f"Output will be ~{out_mp:.0f}MP — processing may use significant memory."
            )
        return requested, warning

    out_mp = output_pixels(width, height, allowed) / 1_000_000
    return allowed, (
        f"Requested {requested}× would exceed the {MAX_OUTPUT_MP // 1_000_000}MP limit; "
        f"used {allowed}× instead (~{out_mp:.0f}MP output)."
    )
