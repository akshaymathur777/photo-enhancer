"""Perceptual metrics for the removal eval bench.

There is no ground-truth inpaint for the eval photos, so the perceptual score is a
self-consistency ratio measured on the output image alone:

    perceptual_context_score = median LPIPS(fill patch, context patch)
                             / median LPIPS(context patch, context patch)

A value near 1.0 means the filled region sits as far from its surroundings as real
surrounding texture sits from itself; >1 means the fill reads as alien. Blur cannot
game this (losing texture moves a patch away from textured context), and
``perceptual_blur_ratio`` is a second guard: how much a further Gaussian blur
perturbs the fill relative to the ring. An already-smooth fill barely moves, so
values below 1 mean the fill is smoother than the scene around it.

The backend is an optional import. When no perceptual package is installed every
value is None and ``reason`` explains why, so the harness degrades instead of
crashing. Runs on CPU on purpose: the inpaint backends want the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import cv2
import numpy as np

PATCH_SIZES = (96, 64, 48, 32)
MAX_PATCHES = 6
MAX_CONTEXT_PAIRS = 15
BLUR_SIGMA = 1.2
CONTEXT_FLOOR = 0.01

_BACKENDS: dict[str, "Backend"] = {}


@dataclass
class Backend:
    name: str | None
    reason: str | None
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None

    @property
    def available(self) -> bool:
        return self.distance is not None


@dataclass
class PerceptualResult:
    backend: str | None = None
    reason: str | None = None
    patch_px: int | None = None
    fill_vs_context: float | None = None
    context_vs_context: float | None = None
    context_score: float | None = None
    blur_ratio: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self, prefix: str = "perceptual_") -> dict[str, object]:
        return {
            f"{prefix}backend": self.backend,
            f"{prefix}reason": self.reason,
            f"{prefix}patch_px": self.patch_px,
            f"{prefix}fill_vs_context": _round(self.fill_vs_context, 4),
            f"{prefix}context_vs_context": _round(self.context_vs_context, 4),
            f"{prefix}context_score": _round(self.context_score, 4),
            f"{prefix}blur_ratio": _round(self.blur_ratio, 4),
        }


def _round(v: float | None, nd: int) -> float | None:
    return None if v is None else round(float(v), nd)


def _lpips_backend(net: str) -> Backend:
    try:
        import torch
        import lpips
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return Backend(None, f"lpips unavailable ({type(exc).__name__}: {exc}); "
                             f"install with: pip install --no-deps lpips")
    try:
        model = lpips.LPIPS(net=net, verbose=False).eval()
    except Exception as exc:  # noqa: BLE001 - backbone weights may be missing offline
        return Backend(None, f"lpips backbone '{net}' could not load "
                             f"({type(exc).__name__}: {exc}); torchvision weights are "
                             f"fetched once to the torch hub cache")

    def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            ta = torch.from_numpy(a).permute(0, 3, 1, 2) * 2.0 - 1.0
            tb = torch.from_numpy(b).permute(0, 3, 1, 2) * 2.0 - 1.0
            out = model(ta, tb)
        return out.detach().cpu().numpy().reshape(-1)

    return Backend(f"lpips:{net}", None, distance)


def _dists_backend() -> Backend:
    try:
        import torch
        from DISTS_pytorch import DISTS
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return Backend(None, f"DISTS unavailable ({type(exc).__name__}: {exc})")
    try:
        model = DISTS().eval()
    except Exception as exc:  # noqa: BLE001
        return Backend(None, f"DISTS could not load ({type(exc).__name__}: {exc})")

    def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            ta = torch.from_numpy(a).permute(0, 3, 1, 2)
            tb = torch.from_numpy(b).permute(0, 3, 1, 2)
            out = model(ta, tb, require_grad=False, batch_average=False)
        return out.detach().cpu().numpy().reshape(-1)

    return Backend("dists", None, distance)


def get_backend(net: str = "squeeze") -> Backend:
    """Resolve a perceptual distance backend, preferring LPIPS then DISTS."""
    if net in _BACKENDS:
        return _BACKENDS[net]
    backend = _lpips_backend(net)
    if not backend.available:
        fallback = _dists_backend()
        if fallback.available:
            fallback.reason = None
            backend = fallback
        else:
            backend = Backend(None, f"{backend.reason} | {fallback.reason}")
    _BACKENDS[net] = backend
    return backend


def _rect(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def _centers(region: np.ndarray, half: int, count: int, seed: int) -> list[tuple[int, int]]:
    h, w = region.shape
    valid = region.copy()
    valid[:half, :] = False
    valid[h - half:, :] = False
    valid[:, :half] = False
    valid[:, w - half:] = False
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return []
    if ys.size <= count:
        idx = np.arange(ys.size)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(ys.size, size=count, replace=False))
    return [(int(ys[i]), int(xs[i])) for i in idx]


def _crop(img: np.ndarray, center: tuple[int, int], half: int) -> np.ndarray:
    cy, cx = center
    return img[cy - half:cy + half, cx - half:cx + half]


def _pick_patches(
    hard: np.ndarray, seed: int, max_patches: int
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    hard_u8 = hard.astype(np.uint8)
    for size in PATCH_SIZES:
        half = size // 2
        interior = cv2.erode(hard_u8, _rect(2 * half + 1)) > 0
        no_fill = cv2.dilate(hard_u8, _rect(2 * half + 1)) == 0
        near = cv2.dilate(hard_u8, _rect(6 * half + 1)) > 0
        fill_centers = _centers(interior, half, max_patches, seed)
        ring_centers = _centers(no_fill & near, half, max_patches, seed + 1)
        if fill_centers and len(ring_centers) >= 2:
            return size, fill_centers, ring_centers
    return 0, [], []


def _batch(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    a = np.stack([p[0] for p in pairs]).astype(np.float32) / 255.0
    b = np.stack([p[1] for p in pairs]).astype(np.float32) / 255.0
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def perceptual_context_metrics(
    result_rgb: np.ndarray,
    hard: np.ndarray,
    *,
    net: str = "squeeze",
    seed: int = 7,
    max_patches: int = MAX_PATCHES,
) -> PerceptualResult:
    """Self-consistency of the fill against its surrounding context, in LPIPS space."""
    backend = get_backend(net)
    if not backend.available:
        return PerceptualResult(reason=backend.reason)
    if not hard.any():
        return PerceptualResult(backend=backend.name, reason="mask empty")

    size, fill_centers, ring_centers = _pick_patches(hard, seed, max_patches)
    if not fill_centers:
        return PerceptualResult(
            backend=backend.name,
            reason=f"mask too thin for perceptual patches (need a {PATCH_SIZES[-1]}px "
                   f"interior square plus surrounding context)",
        )

    half = size // 2
    fills = [_crop(result_rgb, c, half) for c in fill_centers]
    rings = [_crop(result_rgb, c, half) for c in ring_centers]

    cross = [(f, r) for f in fills for r in rings]
    ring_pairs = list(combinations(range(len(rings)), 2))[:MAX_CONTEXT_PAIRS]
    within = [(rings[i], rings[j]) for i, j in ring_pairs]
    blur_fill = [(f, cv2.GaussianBlur(f, (0, 0), BLUR_SIGMA)) for f in fills]
    blur_ring = [(r, cv2.GaussianBlur(r, (0, 0), BLUR_SIGMA)) for r in rings]

    groups = [cross, within, blur_fill, blur_ring]
    a, b = _batch([p for g in groups for p in g])
    dist = backend.distance(a, b)  # type: ignore[misc]

    out: list[float] = []
    at = 0
    for g in groups:
        out.append(float(np.median(dist[at:at + len(g)])))
        at += len(g)
    d_cross, d_within, d_blur_fill, d_blur_ring = out

    reason = None
    if d_within < CONTEXT_FLOOR:
        reason = (f"context contrast very low (context_vs_context={d_within:.4f}); "
                  f"context_score is floored and only weakly meaningful")
    score = d_cross / max(d_within, CONTEXT_FLOOR)
    blur_ratio = d_blur_fill / max(d_blur_ring, 1e-4)

    return PerceptualResult(
        backend=backend.name,
        reason=reason,
        patch_px=size,
        fill_vs_context=d_cross,
        context_vs_context=d_within,
        context_score=score,
        blur_ratio=blur_ratio,
    )


def perceptual_image_distance(
    a_rgb: np.ndarray, b_rgb: np.ndarray, *, net: str = "squeeze", max_side: int = 512
) -> tuple[float | None, str | None]:
    """Whole-image perceptual distance, used to show how far a run drifted from a baseline."""
    backend = get_backend(net)
    if not backend.available:
        return None, backend.reason
    if a_rgb.shape != b_rgb.shape:
        return None, f"shape mismatch {a_rgb.shape} vs {b_rgb.shape}"
    h, w = a_rgb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        dim = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        a_rgb = cv2.resize(a_rgb, dim, interpolation=cv2.INTER_AREA)
        b_rgb = cv2.resize(b_rgb, dim, interpolation=cv2.INTER_AREA)
    a, b = _batch([(a_rgb, b_rgb)])
    return float(backend.distance(a, b)[0]), None  # type: ignore[misc]
