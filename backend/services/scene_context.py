"""Cheap hole-scene classification for object removal routing.

Labels drive prompts, seed strategy, grow/feather, and post-fill texture —
so water-only heuristics stop dominating every background type.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

SceneKind = Literal["water", "sky", "vegetation", "structured", "generic"]


def classify_hole_scene(rgb: np.ndarray, hard: np.ndarray) -> SceneKind:
    """Classify the background around a removal hole from local ring + band cues."""
    import cv2

    hard = hard.astype(bool)
    if not hard.any():
        return "generic"

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    ys, xs = np.where(hard)
    y_top, y_bot = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    hh, _ww = hard.shape
    y0 = max(0, y_top - max(48, int((y_bot - y_top + 1) * 0.55)))
    band = np.zeros_like(hard, dtype=bool)
    band[y0:y_top, x0 : x1 + 1] = True
    band &= ~hard

    def _frac(mask: np.ndarray, region: np.ndarray) -> float:
        n = int(region.sum())
        if n < 64:
            return 0.0
        return float(mask[region].mean())

    # Water / coastal: blue-cyan OR grey mid-value bay (matches existing coastal heuristics).
    blue_water = (h >= 85) & (h <= 135) & (s >= 12) & (s <= 110) & (v > 95)
    grey_coast = (s < 48) & (v > 125) & (v < 235) & (h >= 40) & (h <= 160)
    water_map = blue_water | grey_coast
    water_score = max(_frac(water_map, ring), _frac(water_map, band) * 1.05)

    # Sky / haze: bright, low sat, low local edge energy.
    sky_map = (v > 175) & (s < 40)
    sky_score = max(_frac(sky_map, ring), _frac(sky_map, band) * 1.1)
    edge_ring = 0.0
    if int(ring.sum()) >= 64:
        edge_ring = float(cv2.Canny(gray, 40, 120)[ring].mean())
    if edge_ring > 22.0:
        sky_score *= 0.55

    # Vegetation / grass / foliage.
    green = (h >= 35) & (h <= 95) & (s >= 35) & (v > 40) & (v < 220)
    veg_score = max(_frac(green, ring), _frac(green, band) * 0.9)

    # Structured indoor / walls / floors: mid sat, strong edges, not water/sky.
    # Library aisles often have dark wood + bright busts (low mid-sat frac) but
    # very high edge energy — boost with wood/brown and edge density cues.
    structured_score = 0.0
    if int(ring.sum()) >= 64 and edge_ring > 14.0:
        mid = (s >= 8) & (s <= 90) & (v > 35) & (v < 240)
        wood = (h >= 5) & (h <= 40) & (s >= 12) & (s <= 95) & (v > 30) & (v < 210)
        mid_frac = float(mid[ring].mean())
        wood_frac = float(wood[ring].mean())
        structured_score = max(mid_frac, wood_frac * 1.15) * min(1.0, edge_ring / 26.0)
        if edge_ring > 24.0 and water_score < 0.20 and sky_score < 0.35:
            structured_score = max(structured_score, min(0.55, edge_ring / 55.0))

    # Prefer water when clear — coastal third-person is the hardest failure mode.
    if water_score >= 0.28:
        return "water"
    if sky_score >= 0.42 and sky_score >= veg_score and sky_score >= structured_score:
        return "sky"
    if veg_score >= 0.32 and veg_score >= structured_score:
        return "vegetation"
    if structured_score >= 0.26:
        return "structured"
    if water_score >= 0.18:
        return "water"
    return "generic"


def scene_prompts(scene: SceneKind) -> tuple[str, str]:
    """Positive / negative prompts for generative erase by scene."""
    negative = (
        "person, people, human, man, woman, child, face, body, silhouette, ghost, "
        "arms, legs, clothing, blurry, deformed, low quality, watermark, text, artifacts, "
        "flat, smear, plaque, muddy, soft focus, painted, watercolor, plastic"
    )
    if scene == "water":
        return (
            "photorealistic water with fine natural ripples and sunlit specular highlights, "
            "continuous shoreline context, overcast or natural daylight, seamless photograph, "
            "matching surrounding texture, no people",
            negative + ", smooth blob, ghost face, translucent person",
        )
    if scene == "sky":
        return (
            "photorealistic sky with natural soft cloud texture and smooth atmospheric gradient, "
            "consistent lighting, seamless photograph, matching surrounding sky, no people",
            negative + ", hard clouds, repeating tiles, horizon line through subject, bird, plane",
        )
    if scene == "vegetation":
        return (
            "photorealistic grass foliage and natural vegetation texture, organic detail, "
            "consistent lighting, seamless photograph, matching surrounding plants, no people",
            negative + ", plastic leaves, repeating pattern, smear, painted",
        )
    if scene == "structured":
        return (
            "photorealistic interior wall floor or architectural surface continuation, "
            "straight lines preserved, consistent perspective and lighting, seamless photograph, "
            "matching surrounding materials, no people",
            negative + ", warped perspective, melted edges, furniture ghost, door floating",
        )
    return (
        "high quality photograph, natural background continuation, seamless texture, "
        "photorealistic detail, consistent lighting, sharp focus",
        negative,
    )


def scene_is_open_fill(scene: SceneKind) -> bool:
    """Open soft backgrounds (water/sky) need different grow/feather than structured."""
    return scene in ("water", "sky")


def seed_hole_for_scene(rgb: np.ndarray, hard: np.ndarray, scene: SceneKind) -> np.ndarray:
    """Replace the hole with scene-appropriate context so diffusion isn't conditioned on a person."""
    import cv2

    hard = hard.astype(bool)
    if not hard.any():
        return rgb.copy()

    if scene == "water":
        # Full 2D water synthesis as SD seed — light refine keeps ripple energy.
        water = _water_source_map(rgb, hard)
        if int(water.sum()) >= 200:
            seeded = _synthesize_water_2d(rgb, hard, water)
            return _match_fill_hf_to_ring(rgb, seeded, hard, target_ratio=0.9, max_gain=3.2)
        return _seed_exemplar_from_map(rgb, hard, water)
    if scene == "sky":
        return _seed_sky_gradient(rgb, hard)
    if scene in ("vegetation", "structured", "generic"):
        # Stamp ring exemplars when textured; otherwise soft TELEA.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if int(ring.sum()) >= 256 and float(cv2.Canny(gray, 40, 120)[ring].mean()) > 10.0:
            return _seed_exemplar_from_map(rgb, hard, ring)
        return _seed_telea(rgb, hard)
    return rgb.copy()


def patchmatch_fill(
    rgb: np.ndarray,
    hard: np.ndarray,
    *,
    patch: int = 24,
    searches: int = 8,
    dense: bool = False,
) -> np.ndarray:
    """Multi-scale exemplar fill (OpenCV-only PatchMatch-lite) for structured/water holes.

    Copies best-matching patches from a dilated ring into the hole so floors/shelves/water
    keep real texture instead of LaMa/SD soft plaques.
    """
    import cv2

    hard = hard.astype(bool)
    if not hard.any():
        return rgb.copy()

    out = rgb.copy()
    h, w = rgb.shape[:2]
    # Wider source ring for water (ripples farther away still match locally).
    ring_r = max(61, patch * 4 + 1) if dense else max(31, patch * 2 + 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_r, ring_r))
    source = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    # Prefer water-like source when available (coastal) — wide ring, no people/glare.
    try:
        water_src = _water_source_map(rgb, hard)
        if int(water_src.sum()) >= 256:
            wide = _wide_water_source_ring(rgb, hard, water_src)
            if int(wide.sum()) >= 256:
                source = wide
            else:
                source = (source | water_src) & (~hard)
    except Exception:
        pass
    if int(source.sum()) < 256:
        return _seed_telea(rgb, hard)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ys_h, xs_h = np.where(hard)
    ys_s, xs_s = np.where(source)
    # Cap source set — full-frame water rings make dist2 untenable.
    if len(ys_s) > 4000:
        pick = np.linspace(0, len(ys_s) - 1, 4000).astype(np.int32)
        ys_s, xs_s = ys_s[pick], xs_s[pick]
    rng = np.random.default_rng(11)
    half = patch // 2

    # Denser sampling on water plaques / structured floors — keep Python loops bounded.
    target_samples = 900 if dense else 700
    step = max(1, int(np.sqrt(max(len(ys_h), 1) / float(target_samples))))
    # Hard cap: never walk more than ~1200 hole samples (latency).
    max_samples = 1200 if dense else 900
    indices = np.arange(0, len(ys_h), step)
    if len(indices) > max_samples:
        indices = indices[:: max(1, len(indices) // max_samples)][:max_samples]
    near_k = 48 if dense else 40
    cand_near = 12 if dense else 10
    n_search = searches + (2 if dense else 0)
    for yi in indices:
        y, x = int(ys_h[yi]), int(xs_h[yi])
        best_score = 1e18
        best_patch = None
        # Prefer nearby source samples; for water also bias upward (same column band).
        dist2 = (ys_s.astype(np.float32) - y) ** 2 + (xs_s.astype(np.float32) - x) ** 2
        if dense:
            # Prefer sources above the hole (real water continuity).
            above_penalty = np.where(ys_s < y, 0.0, 180.0)
            dist2 = dist2 + above_penalty
        near_idx = np.argpartition(dist2, min(near_k, len(dist2) - 1))[: min(near_k, len(dist2))]
        cand = list(near_idx[:cand_near])
        for _ in range(n_search):
            cand.append(int(rng.integers(0, len(ys_s))))
        for ci in cand:
            cy, cx = int(ys_s[ci]), int(xs_s[ci])
            y0, y1 = max(0, cy - half), min(h, cy + half)
            x0, x1 = max(0, cx - half), min(w, cx + half)
            if y1 - y0 < 8 or x1 - x0 < 8:
                continue
            # Compare grayscale neighborhood against hole neighborhood using TELEA guide
            ty0, ty1 = max(0, y - half), min(h, y + half)
            tx0, tx1 = max(0, x - half), min(w, x + half)
            sh, sw = min(y1 - y0, ty1 - ty0), min(x1 - x0, tx1 - tx0)
            if sh < 8 or sw < 8:
                continue
            src_g = gray[y0 : y0 + sh, x0 : x0 + sw]
            # Valid compare only where target ring exists (not deep hole)
            tgt_hard = hard[ty0 : ty0 + sh, tx0 : tx0 + sw]
            valid = ~tgt_hard
            if int(valid.sum()) < 16:
                score = float(np.abs(src_g - src_g.mean()).mean())  # prefer textured
            else:
                tgt_g = gray[ty0 : ty0 + sh, tx0 : tx0 + sw]
                score = float(np.abs(src_g[valid] - tgt_g[valid]).mean())
            if score < best_score:
                best_score = score
                best_patch = rgb[y0 : y0 + sh, x0 : x0 + sw]

        if best_patch is None:
            continue
        dy0, dy1 = max(0, y - half), min(h, y + half)
        dx0, dx1 = max(0, x - half), min(w, x + half)
        bh, bw_ = dy1 - dy0, dx1 - dx0
        if best_patch.shape[0] != bh or best_patch.shape[1] != bw_:
            best_patch = cv2.resize(best_patch, (bw_, bh), interpolation=cv2.INTER_LINEAR)
        region = hard[dy0:dy1, dx0:dx1]
        out[dy0:dy1, dx0:dx1][region] = best_patch[region]

    # Soft NS only at outer fringe (never soft TELEA plaque over the whole hole).
    try:
        guide = out.copy()
        soft = cv2.inpaint(guide, (hard.astype(np.uint8) * 255), 2, cv2.INPAINT_NS)
        dist = cv2.distanceTransform(hard.astype(np.uint8) * 255, cv2.DIST_L2, 5)
        fringe = (hard & (dist <= 2.5)).astype(np.float32)
        fringe = cv2.GaussianBlur(fringe, (0, 0), 0.7)[:, :, None]
        seam = 0.28 if dense else 0.40
        out = (
            out.astype(np.float32) * (1.0 - seam * fringe) + soft.astype(np.float32) * (seam * fringe)
        ).astype(np.uint8)
    except Exception:
        pass
    return out


def _region_hf_energy(rgb: np.ndarray, region: np.ndarray) -> float:
    """Mean |gray - blur| energy inside a boolean region."""
    import cv2

    if not region.any():
        return 0.0
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(g[region] - cv2.GaussianBlur(g, (0, 0), 2.0)[region]).mean())


def _match_fill_hf_to_ring(
    original: np.ndarray,
    filled: np.ndarray,
    hard: np.ndarray,
    *,
    target_ratio: float = 0.92,
    max_gain: float = 3.6,
    prefer_water_ref: bool = True,
) -> np.ndarray:
    """Boost (or gently damp) high-frequency residual toward ring energy."""
    import cv2

    hard = hard.astype(bool)
    if int(hard.sum()) < 64:
        return filled
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    water = _water_source_map(original, hard)
    # Only bias to water ref when water dominates the ring (coastal). Indoor wood
    # often matches the grey-coast heuristic and would block damp incorrectly.
    water_frac = float((ring & water).mean()) if int(ring.sum()) else 0.0
    if prefer_water_ref and water_frac >= 0.35 and int((ring & water).sum()) >= 64:
        ref = ring & water
    else:
        ref = ring
    if int(ref.sum()) < 64:
        return filled
    ring_hf = _region_hf_energy(original, ref)
    fill_hf = _region_hf_energy(filled, hard)
    if ring_hf < 1.5:
        return filled

    out = filled.astype(np.float32)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0)
    hf = out - blur

    # Too noisy (structured smear / moiré) — damp toward ring.
    if fill_hf > ring_hf * 1.15 and fill_hf > 8.0:
        damp = float(np.clip((ring_hf * 1.02) / max(fill_hf, 0.1), 0.30, 0.92))
        result = filled.copy().astype(np.float32)
        result[hard] = np.clip(blur[hard] + hf[hard] * damp, 0, 255)
        return result.astype(np.uint8)

    need = ring_hf * float(target_ratio)
    if fill_hf >= need * 0.88:
        return filled

    src = (water | ref) & (~hard) if water_frac >= 0.35 else ref
    if int(src.sum()) < 128:
        src = ref
    ys_h, xs_h = np.where(hard)
    ys_s, xs_s = np.where(src)
    if len(ys_s) < 16:
        return filled
    # Randomized HF injection — modulo tiling creates horizontal banding.
    rng = np.random.default_rng(19)
    injected = np.zeros_like(hf)
    orig_f = original.astype(np.float32)
    lap = orig_f - cv2.GaussianBlur(orig_f, (0, 0), 2.0)
    picks = rng.integers(0, len(ys_s), size=len(ys_h))
    # Shuffle with spatial hash so neighbors aren't identical columns.
    jitter = ((ys_h * 17 + xs_h * 31) % len(ys_s)).astype(np.int32)
    picks = (picks + jitter) % len(ys_s)
    injected[ys_h, xs_h] = lap[ys_s[picks], xs_s[picks]]
    hf = injected
    gain = float(np.clip(need / max(fill_hf, 0.35), 1.15, max_gain))
    textured = blur + hf * gain
    result = filled.copy().astype(np.float32)
    result[hard] = np.clip(textured[hard], 0, 255)
    fill2 = _region_hf_energy(result.astype(np.uint8), hard)
    if fill2 < need * 0.85 and fill2 > 0.1:
        extra = float(np.clip(need / fill2, 1.0, 1.9))
        blur2 = cv2.GaussianBlur(result, (0, 0), 2.0)
        result[hard] = np.clip(blur2[hard] + (result[hard] - blur2[hard]) * extra, 0, 255)
    return np.clip(result, 0, 255).astype(np.uint8)


def _wide_water_source_ring(
    original: np.ndarray,
    hard: np.ndarray,
    water_src: np.ndarray,
) -> np.ndarray:
    """Wide water exemplar sources via distance-from-hole (not AABB traps).

    AABB side-traps fail when the hard mask has a wide arm — they erase the
    entire bay. Prefer blue water with a modest silhouette collar.
    """
    import cv2

    hard = hard.astype(bool)
    h, w = hard.shape
    ys, xs = np.where(hard)
    if len(ys) == 0:
        return np.zeros_like(hard, dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    hole_h = max(y1 - y0 + 1, 8)
    hole_w = max(x1 - x0 + 1, 8)

    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = (hh >= 85) & (hh <= 135) & (ss >= 10) & (ss <= 120) & (vv > 95) & (vv < 245)
    grey_bay = (ss < 48) & (vv > 130) & (vv < 230) & (hh >= 50) & (hh <= 150)
    waterish = (water_src.astype(bool) | blue | grey_bay) & (~hard)

    dist = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, 5)
    # Modest collar — adjacent water inherits person-edge color.
    src = waterish & (dist >= 22.0)

    # Exclude people with a modest halo.
    try:
        from services.object_removal import _all_people_binary

        people = _all_people_binary(original, conf=0.18) > 0
        if people.any():
            people = (
                cv2.dilate(
                    people.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                )
                > 0
            )
            src = src & (~people)
    except Exception:
        pass

    # Prefer true blue water; keep grey bay as secondary.
    blue_src = src & blue
    if int(blue_src.sum()) >= 1200:
        src = blue_src

    above = src.copy()
    above[y0:, :] = False
    # Lateral / below: same bay band beside the silhouette (dist-gated already).
    lateral = src & (~above)
    # Prefer lateral water whose Y overlaps the hole (real ripple continuity)
    # but stays away from the couple — use a stronger dist for hole-depth rows.
    hole_depth = np.zeros_like(hard, dtype=bool)
    hole_depth[y0 : y1 + 1, :] = True
    deep_ok = lateral & ((~hole_depth) | (dist >= 40.0))

    if int(above.sum()) >= 600 and int(deep_ok.sum()) >= 800:
        # Mix above + lateral (subsample lateral to avoid rock foreshore dominance).
        ys_l, xs_l = np.where(deep_ok)
        keep = ((ys_l * 3 + xs_l) % 2) == 0
        lat2 = np.zeros_like(deep_ok)
        lat2[ys_l[keep], xs_l[keep]] = True
        src = above | lat2
    elif int(above.sum()) >= 600:
        src = above | (deep_ok if int(deep_ok.sum()) >= 200 else above)
    elif int(deep_ok.sum()) >= 800:
        src = deep_ok
    elif int(src.sum()) < 400:
        # Last resort: any waterish with tiny collar.
        src = waterish & (dist >= 12.0)
        try:
            from services.object_removal import _all_people_binary

            people = _all_people_binary(original, conf=0.18) > 0
            if people.any():
                src = src & (~people)
        except Exception:
            pass
    return src


def _inject_water_laplacian_hf(
    original: np.ndarray,
    filled: np.ndarray,
    hard: np.ndarray,
    source: np.ndarray,
    *,
    target_hf: float = 12.0,
) -> np.ndarray:
    """Inject high-frequency ripple (laplacian residual) matching ring HF (~12)."""
    import cv2

    hard = hard.astype(bool)
    source = source.astype(bool) & (~hard)
    if not hard.any() or int(source.sum()) < 64:
        return filled

    out = filled.astype(np.float32)
    orig_f = original.astype(np.float32)
    # Laplacian residual ≈ high-frequency ripple.
    blur = cv2.GaussianBlur(orig_f, (0, 0), 1.8)
    lap = orig_f - blur
    blur_f = cv2.GaussianBlur(out, (0, 0), 1.8)
    own = out - blur_f

    ring_hf = _region_hf_energy(original, source)
    fill_hf = _region_hf_energy(filled, hard)
    want = max(float(target_hf) * 0.92, ring_hf * 0.95) if ring_hf >= 2.0 else float(target_hf)

    ys_h, xs_h = np.where(hard)
    ys_s, xs_s = np.where(source)
    rng = np.random.default_rng(29)
    # Randomized HF stamps — avoid column-aligned banding.
    injected = np.zeros_like(own)
    # Bias sources to similar x but with strong jitter.
    x_min, x_max = float(xs_h.min()), float(xs_h.max())
    xs_s_f = xs_s.astype(np.float32)
    kth = min(24, len(xs_s_f) - 1)
    n_near = min(24, len(xs_s_f))
    # target_x collapses to a few hundred values (it is x plus an integer jitter),
    # so key on the exact float — rounding the key would change which sources win.
    near_by_target_x: dict[float, np.ndarray] = {}
    for i, (y, x) in enumerate(zip(ys_h, xs_h)):
        # Prefer nearby-x sources with random offset.
        x_norm = (float(x) - x_min) / max(x_max - x_min, 1.0)
        # Hash-based pick for stability + jitter.
        j = int((y * 131 + x * 17 + i * 7) % len(ys_s))
        j2 = int(rng.integers(0, len(ys_s)))
        # Mix two sources so ripples aren't vertically correlated.
        if (i % 2) == 0:
            sy, sx = int(ys_s[j]), int(xs_s[j])
        else:
            # Bias second pick toward similar column band.
            target_x = x_min + x_norm * max(x_max - x_min, 1.0) + float(rng.integers(-40, 41))
            near = near_by_target_x.get(target_x)
            if near is None:
                dist = np.abs(xs_s_f - target_x)
                near = np.argpartition(dist, kth)[:n_near]
                near_by_target_x[target_x] = near
            pick = int(near[int(rng.integers(0, len(near)))])
            sy, sx = int(ys_s[pick]), int(xs_s[pick])
            _ = j2  # keep rng advanced
        injected[y, x] = lap[sy, sx]

    if fill_hf < want * 0.85:
        # Replace soft fill HF with injected ripple; keep a little structure.
        gain = float(np.clip(want / max(fill_hf, 0.5), 1.1, 3.5))
        combined = own * 0.25 + injected * (0.85 * gain)
    else:
        # Already textured — mild top-up only.
        combined = own * 0.85 + injected * 0.35

    textured = blur_f + combined
    result = filled.copy().astype(np.float32)
    result[hard] = np.clip(textured[hard], 0, 255)
    # Final nudge toward target HF if still flat.
    fill2 = _region_hf_energy(result.astype(np.uint8), hard)
    if fill2 < want * 0.88:
        extra = float(np.clip(want / max(fill2, 0.4), 1.0, 2.2))
        b2 = cv2.GaussianBlur(result, (0, 0), 1.8)
        result[hard] = np.clip(b2[hard] + (result[hard] - b2[hard]) * extra, 0, 255)
    return np.clip(result, 0, 255).astype(np.uint8)


def _synthesize_water_2d(original: np.ndarray, hard: np.ndarray, water_src: np.ndarray) -> np.ndarray:
    """Fill water holes with distance-gated same-row strips + above for the head.

    Prefer slightly-wrong water texture over a person-shaped grey blob. Sample
    only water with dist>=36 from the hole silhouette (avoids couple/person edge).
    """
    import cv2

    hard = hard.astype(bool)
    if not hard.any():
        return original.copy()

    h, w = hard.shape
    ys_h, xs_h = np.where(hard)
    y_top, y_bot = int(ys_h.min()), int(ys_h.max())
    hole_h = max(y_bot - y_top + 1, 1)

    dist = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, 5)
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = (hh >= 85) & (hh <= 135) & (ss >= 10) & (ss <= 120) & (vv > 95) & (vv < 245)
    grey_bay = (ss < 50) & (vv > 125) & (vv < 235) & (hh >= 45) & (hh <= 155)
    waterish = (water_src.astype(bool) | blue | grey_bay) & (~hard)

    try:
        from services.object_removal import _all_people_binary

        people = _all_people_binary(original, conf=0.18) > 0
        people = (
            cv2.dilate(
                people.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            )
            > 0
        )
    except Exception:
        people = np.zeros_like(hard, dtype=bool)

    # Clean source: water away from hole + people.
    source = waterish & (dist >= 36.0) & (~people)
    if int((source & blue).sum()) >= 800:
        source = source & blue
    if int(source.sum()) < 200:
        source = waterish & (dist >= 20.0) & (~people)
    if int(source.sum()) < 80:
        mean_c = (
            original[waterish].mean(axis=0)
            if waterish.any()
            else np.array([160.0, 175.0, 190.0])
        )
        out = original.copy()
        out[hard] = mean_c
        return out

    skyish = (vv > 175) & (ss < 40) & (~hard) & (~people)
    landish = (
        ((hh < 85) | (hh > 140))
        & (ss > 20)
        & (vv > 40)
        & (vv < 200)
        & (~hard)
        & (~people)
    )

    # Per-row source x coords (distance-gated — safe same-row fill).
    row_xs: dict[int, np.ndarray] = {}
    for y in range(max(0, y_top - 60), min(h, y_bot + 40)):
        xs = np.where(source[y])[0]
        if len(xs) >= 3:
            row_xs[y] = xs

    out = original.copy().astype(np.float32)
    rng = np.random.default_rng(47)
    sy, sx = np.where(source)
    sy_f = sy.astype(np.float32)

    for y in range(y_top, y_bot + 1):
        xs = np.where(hard[y])[0]
        if len(xs) == 0:
            continue
        depth = float(y - y_top) / float(hole_h)
        x_min, x_max = int(xs.min()), int(xs.max())

        # Head band against sky/hills: sample upward only.
        ups_s = ups_l = ups_w = 0.0
        ups_n = 0
        for x in xs[:: max(1, len(xs) // 6)]:
            for dy in range(1, 50):
                yy = y - dy
                if yy < 0:
                    break
                if not hard[yy, x]:
                    ups_n += 1
                    ups_s += 1.0 if skyish[yy, x] else 0.0
                    ups_l += 1.0 if landish[yy, x] else 0.0
                    ups_w += 1.0 if source[yy, x] else 0.0
                    break
        if ups_n:
            ups_s, ups_l, ups_w = ups_s / ups_n, ups_l / ups_n, ups_w / ups_n
        use_sky = depth < 0.28 and (ups_s >= 0.30 or ups_l >= 0.30) and ups_w < 0.45

        if use_sky:
            for x in xs:
                picked = None
                for dy in (1, 2, 4, 8, 16, 28):
                    yy = y - dy
                    if yy < 0:
                        break
                    if not hard[yy, x] and not people[yy, x]:
                        picked = original[yy, x].astype(np.float32)
                        break
                if picked is None:
                    j = int(rng.integers(0, len(sy)))
                    picked = original[int(sy[j]), int(sx[j])].astype(np.float32)
                out[y, x] = picked
            continue

        # Body: same-row water strips from LEFT/RIGHT of hole (dist-gated).
        near_row = None
        for i, x in enumerate(xs):
            picked = None
            for jitter in (0, -1, 1, -2, 2, -4, 4, -8, 8):
                yy = min(max(y + jitter, 0), h - 1)
                wxs = row_xs.get(yy)
                if wxs is None or len(wxs) < 2:
                    continue
                left = wxs[wxs < x_min - 8]
                right = wxs[wxs > x_max + 8]
                if len(left) >= 2 and len(right) >= 2:
                    pool = left if ((i + y + jitter) % 2 == 0) else right
                elif len(left) >= 2:
                    pool = left
                elif len(right) >= 2:
                    pool = right
                else:
                    continue
                rel = float(x - x_min) / max(float(x_max - x_min), 1.0)
                idx = int(
                    np.clip(rel * (len(pool) - 1) + rng.integers(-2, 3), 0, len(pool) - 1)
                )
                picked = original[yy, int(pool[idx])].astype(np.float32)
                idx2 = int(np.clip(idx + int(rng.integers(-3, 4)), 0, len(pool) - 1))
                picked = 0.75 * picked + 0.25 * original[yy, int(pool[idx2])].astype(
                    np.float32
                )
                break
            if picked is None:
                if near_row is None:
                    y_dist = np.abs(sy_f - float(y))
                    near_row = np.argpartition(y_dist, min(24, len(y_dist) - 1))[
                        : min(24, len(y_dist))
                    ]
                near = near_row
                j = int(near[int(rng.integers(0, len(near)))])
                picked = original[int(sy[j]), int(sx[j])].astype(np.float32)
            out[y, x] = picked

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    ref = source & blue if int((source & blue).sum()) >= 200 else source
    src_mean = original[ref].astype(np.float32).mean(axis=0)
    src_std = original[ref].astype(np.float32).std(axis=0) + 1e-3
    dst = out_u8[hard].astype(np.float32)
    dst_mean, dst_std = dst.mean(axis=0), dst.std(axis=0) + 1e-3
    adj = (dst - dst_mean) * np.clip(src_std / dst_std, 0.7, 1.4) * 0.85 + src_mean
    dst_y = 0.299 * adj[:, 0] + 0.587 * adj[:, 1] + 0.114 * adj[:, 2]
    src_y = float(0.299 * src_mean[0] + 0.587 * src_mean[1] + 0.114 * src_mean[2])
    if float(dst_y.mean()) < src_y - 3.0:
        adj = adj * float(
            np.clip((src_y - 1.0) / max(float(dst_y.mean()), 1.0), 1.0, 1.30)
        )
    out_u8 = out_u8.copy()
    out_u8[hard] = np.clip(adj, 0, 255).astype(np.uint8)
    out_u8 = _inject_water_laplacian_hf(original, out_u8, hard, source, target_hf=10.5)

    result = original.copy()
    result[hard] = out_u8[hard]
    return result



def fill_for_scene(
    original: np.ndarray,
    prior: np.ndarray,
    mask: np.ndarray,
    scene: SceneKind | None = None,
) -> np.ndarray:
    """Authoritative scene fill: 2D water synthesis; structured TELEA + directional grain."""
    import cv2

    hard = mask > 127
    if not hard.any():
        return prior
    if scene is None:
        scene = classify_hole_scene(original, hard)

    if scene == "water":
        water_src = _water_source_map(original, hard)
        if int(water_src.sum()) >= 200:
            # Randomized exemplar is authoritative — never soft-blend a TELEA/LaMa plaque.
            base = _synthesize_water_2d(original, hard, water_src)
            ring_src = _wide_water_source_ring(original, hard, water_src)
            fill_hf = _region_hf_energy(base, hard)
            ring_hf = _region_hf_energy(original, ring_src) if int(ring_src.sum()) >= 64 else 12.0
            if fill_hf < ring_hf * 0.70 or fill_hf < 7.0:
                # Extra random stamps + HF — skip slow PatchMatch (was banding).
                base = _seed_exemplar_from_map(original, hard, ring_src if int(ring_src.sum()) >= 64 else water_src)
                base = _inject_water_laplacian_hf(
                    original, base, hard, ring_src if int(ring_src.sum()) >= 64 else water_src,
                    target_hf=max(12.0, ring_hf),
                )
        else:
            base = prior
        # Skip inject_ring_texture tiling for water (modulo tile = banding).
        mixed = _inject_water_laplacian_hf(
            original,
            base,
            hard,
            _wide_water_source_ring(original, hard, water_src)
            if int(water_src.sum()) >= 64
            else water_src,
            target_hf=12.0,
        )
        mixed = _match_fill_hf_to_ring(
            original, mixed, hard, target_ratio=0.95, max_gain=2.4, prefer_water_ref=True
        )
        result = prior.copy()
        # Hard composite: full exemplar replace inside hole (no soft plaque overlay).
        result[hard] = mixed[hard]
        return result

    if scene == "structured":
        # Wood-aware: PatchMatch first (preserves shelf lines), light TELEA only at fringe.
        pm = patchmatch_fill(original, hard, patch=28, searches=5, dense=False)
        try:
            guide = original.copy()
            guide[hard] = pm[hard]
            telea = cv2.inpaint(guide, (hard.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
            dist = cv2.distanceTransform(hard.astype(np.uint8) * 255, cv2.DIST_L2, 5)
            fringe = (hard & (dist <= 4.0)).astype(np.float32)[:, :, None]
            base = (
                pm.astype(np.float32) * (1.0 - 0.45 * fringe)
                + telea.astype(np.float32) * (0.45 * fringe)
            ).astype(np.uint8)
        except Exception:
            base = pm
        base = _continue_floor_texture(original, base, hard)
        # Mild ring texture only — heavy inject melts shelves into milky smear.
        mixed = inject_ring_texture(original, base, mask, scene=scene)
        mixed = _match_fill_hf_to_ring(
            original, mixed, hard, target_ratio=0.50, max_gain=1.02, prefer_water_ref=False
        )
        # Prefer PatchMatch base when inject washed / melted structure.
        g_base = float(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)[hard].mean())
        g_mix = float(cv2.cvtColor(mixed, cv2.COLOR_RGB2GRAY)[hard].mean())
        edge_base = float(cv2.Canny(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), 40, 120)[hard].mean())
        edge_mix = float(cv2.Canny(cv2.cvtColor(mixed, cv2.COLOR_RGB2GRAY), 40, 120)[hard].mean())
        if g_mix > g_base + 10.0 or edge_mix < edge_base * 0.75:
            mixed = base
        result = prior.copy()
        result[hard] = mixed[hard]
        return result

    return inject_ring_texture(original, prior, mask, scene=scene)


def _continue_floor_texture(original: np.ndarray, base: np.ndarray, hard: np.ndarray) -> np.ndarray:
    """Pull wood-grain from below/above the hole along the aisle direction."""
    import cv2

    hard = hard.astype(bool)
    if not hard.any():
        return base
    h, w = hard.shape
    ys, xs = np.where(hard)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # Prefer floor strip below the person feet.
    y_lo = min(h - 1, y1 + 1)
    y_hi = min(h, y1 + max(28, (y1 - y0) // 2))
    below = np.zeros_like(hard, dtype=bool)
    below[y_lo:y_hi, max(0, x0 - 8) : min(w, x1 + 9)] = True
    below &= ~hard
    src = below
    if int(src.sum()) < 80:
        y_a0 = max(0, y0 - max(24, (y1 - y0) // 3))
        above = np.zeros_like(hard, dtype=bool)
        above[y_a0:y0, max(0, x0 - 8) : min(w, x1 + 9)] = True
        above &= ~hard
        src = above
    if int(src.sum()) < 64:
        return base
    out = base.astype(np.float32)
    sy, sx = np.where(src)
    sy0, sy1 = int(sy.min()), int(sy.max())
    sx0, sx1 = int(sx.min()), int(sx.max())
    if sy1 <= sy0 or sx1 <= sx0:
        return base
    strip = original[sy0 : sy1 + 1, sx0 : sx1 + 1].astype(np.float32)
    sh, sw = strip.shape[:2]
    # Map hole columns onto strip columns (perspective-ish lengthwise grain).
    for y, x in zip(ys, xs):
        # Bias toward vertical neighbors in the strip (plank direction).
        tx = int(np.clip((x - x0) / max(x1 - x0, 1) * (sw - 1), 0, sw - 1))
        ty = int(((y - y0) * 3 + (x * 2)) % sh)
        # Mix with base so shelves above floor aren't forced to floor color.
        depth = float(y - y0) / float(max(y1 - y0, 1))
        # Stronger floor pull near bottom of hole (feet region).
        tw = 0.55 + 0.35 * depth
        out[y, x] = strip[ty, tx] * tw + out[y, x] * (1.0 - tw)
    # Light TELEA seam only.
    try:
        soft = cv2.inpaint(out.astype(np.uint8), (hard.astype(np.uint8) * 255), 3, cv2.INPAINT_TELEA)
        blend = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.5)[:, :, None]
        out = out * (1.0 - 0.18 * blend) + soft.astype(np.float32) * (0.18 * blend)
    except Exception:
        pass
    result = base.copy()
    result[hard] = np.clip(out[hard], 0, 255).astype(np.uint8)
    return result


def inject_ring_texture(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    scene: SceneKind | None = None,
) -> np.ndarray:
    """Inject high-frequency texture from nearby context into flat fills (any scene)."""
    import cv2

    hard = mask > 127
    if not hard.any():
        return inpainted

    if scene is None:
        scene = classify_hole_scene(original, hard)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    if int(ring.sum()) < 64:
        return inpainted

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    ring_edge_density = float(cv2.Canny(gray, 40, 120)[ring].mean())
    # Sky can be legitimately low-edge; still allow mild inject from band above.
    min_edge = 5.0 if scene == "sky" else 8.0
    if ring_edge_density < min_edge and scene != "sky":
        return inpainted

    fill = inpainted.astype(np.float32)
    orig_f = original.astype(np.float32)
    sigma = 2.0
    blur_o = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=sigma)
    blur_f = cv2.GaussianBlur(fill, (0, 0), sigmaX=sigma)
    hf_o = orig_f - blur_o

    source = ring.copy()
    if scene == "water":
        source = (_water_source_map(original, hard) | ring) & (~hard)
    elif scene == "sky":
        hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
        sky = (hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 45) & (~hard)
        source = (sky | ring) & (~hard)
    elif scene == "vegetation":
        hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
        green = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 30)
            & (~hard)
        )
        source = (green | ring) & (~hard)

    ys_h, xs_h = np.where(hard)
    y_top = int(ys_h.min())
    hf_fill = np.zeros_like(fill)
    used_tile = False

    above = source.copy()
    above[y_top:, :] = False
    if int(above.sum()) >= 400:
        ay, ax = np.where(above)
        ay0, ay1 = int(np.percentile(ay, 8)), int(np.percentile(ay, 92))
        ax0, ax1 = int(np.percentile(ax, 8)), int(np.percentile(ax, 92))
        if ay1 > ay0 + 12 and ax1 > ax0 + 24:
            tile_hf = hf_o[ay0 : ay1 + 1, ax0 : ax1 + 1]
            th, tw = tile_hf.shape[:2]
            x_min = int(xs_h.min())
            for y, x in zip(ys_h, xs_h):
                sy = ((y - y_top) + (x * 3) % 5) % th
                sx = ((x - x_min) + (y * 5) % 7) % tw
                hf_fill[y, x] = tile_hf[sy, sx]
            used_tile = float(np.abs(hf_fill[hard]).mean()) >= 0.5

    if not used_tile:
        src_ys, src_xs = np.where(source)
        if len(src_ys) < 32:
            src_ys, src_xs = np.where(ring)
        if len(src_ys) < 8:
            return inpainted
        rng = np.random.default_rng(17)
        for y, x in zip(ys_h, xs_h):
            i = int(rng.integers(0, len(src_ys)))
            hf_fill[y, x] = hf_o[int(src_ys[i]), int(src_xs[i])]

    # Water: never TELEA from original (person still in hole — bleeds skin/grey).
    # Prefer amplifying existing fill HF; tiled HF injection causes ripple banding.
    if scene == "water":
        own_hf = fill - blur_f
        own_e = float(np.abs(own_hf[hard]).mean())
        if own_e >= 2.0:
            # Base already has real water texture — light boost only.
            hf_combined = own_hf * 1.25
        elif own_e >= 1.0:
            hf_combined = own_hf * 1.45 + hf_fill * 0.25
        elif used_tile:
            rng = np.random.default_rng(23)
            src_ys, src_xs = np.where(source)
            if len(src_ys) >= 32:
                for y, x in zip(ys_h, xs_h):
                    if ((y + x) % 3) == 0:
                        i = int(rng.integers(0, len(src_ys)))
                        hf_fill[y, x] = hf_o[int(src_ys[i]), int(src_xs[i])]
            hf_combined = hf_fill * 1.35
        else:
            hf_combined = own_hf * 1.6 if own_e > 0.2 else hf_fill * 1.5
        textured = blur_f + hf_combined
        out = fill.copy()
        out[hard] = np.clip(textured[hard], 0, 255)
        return _match_fill_hf_to_ring(
            original, out.astype(np.uint8), hard, target_ratio=0.90, max_gain=2.2, prefer_water_ref=True
        )

    try:
        telea = cv2.inpaint(original, (hard.astype(np.uint8) * 255), 7, cv2.INPAINT_TELEA).astype(
            np.float32
        )
        blur_t = cv2.GaussianBlur(telea, (0, 0), sigmaX=sigma)
        dist = cv2.distanceTransform((hard.astype(np.uint8)) * 255, cv2.DIST_L2, 5)
        edge_w = np.clip(1.0 - dist / 18.0, 0.0, 1.0)[:, :, None]
        hf_combined = hf_fill * (1.0 - edge_w) + (telea - blur_t) * edge_w
        blur_f = blur_f * 0.92 + blur_t * 0.08
        hf_gain = 1.10 if scene == "structured" else 1.20
        hf_combined = hf_combined * hf_gain
    except Exception:
        hf_gain = 1.10 if scene == "structured" else 1.15
        hf_combined = hf_fill * hf_gain

    textured = blur_f + hf_combined
    out = fill.copy()
    out[hard] = np.clip(textured[hard], 0, 255)
    if scene == "structured":
        return _match_fill_hf_to_ring(
            original,
            out.astype(np.uint8),
            hard,
            target_ratio=0.65,
            max_gain=1.15,
            prefer_water_ref=False,
        )
    return out.astype(np.uint8)


def _water_source_map(rgb: np.ndarray, hard: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Coastal water — exclude warm wood/brown and bright sky.
    blue = (h >= 90) & (h <= 130) & (s >= 12) & (s <= 105) & (v > 115) & (v < 230)
    # Grey bay water: cool bias, but NOT bright sky (low sat + very bright).
    grey = (s < 48) & (v > 145) & (v < 210) & (h >= 75) & (h <= 145)
    # Explicit sky exclusion: bright + low sat (overcast / haze).
    sky = (v > 200) & (s < 35)
    # Very top of frame is almost always sky on coastal selfies — drop it.
    hh = hard.shape[0]
    top_band = np.zeros_like(hard, dtype=bool)
    top_band[: max(1, int(hh * 0.12)), :] = True
    return (blue | grey) & (~hard.astype(bool)) & (~sky) & (~top_band)


def _water_tile_above(rgb: np.ndarray, hard: np.ndarray, water_src: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return (tile RGB float32, source boolean) for water ABOVE the hole."""
    import cv2

    hard = hard.astype(bool)
    h, w = hard.shape
    ys, xs = np.where(hard)
    if len(ys) == 0:
        return None, None
    y_top = int(ys.min())
    x0, x1 = int(xs.min()), int(xs.max())
    # Geometric band directly above the hole — same bay, not distant dark rocks.
    y0 = max(0, y_top - max(56, int((ys.max() - y_top + 1) * 0.85)))
    band = np.zeros_like(hard, dtype=bool)
    band[y0:y_top, max(0, x0 - 48) : min(w, x1 + 49)] = True
    band &= ~hard
    src = band & water_src
    if int(src.sum()) < 300:
        src = band.copy()
        # Keep mid-bright, low-mid sat pixels even if water map missed them.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        ok = (hsv[:, :, 2] > 135) & (hsv[:, :, 2] < 235) & (hsv[:, :, 1] < 60)
        src = band & ok
    if int(src.sum()) < 200:
        src = water_src & band
    if int(src.sum()) < 150:
        # Fall back to any water above y_top.
        above = water_src.copy()
        above[y_top:, :] = False
        src = above
    if int(src.sum()) < 120:
        return None, None

    # Prefer mid-bright water but KEEP sparkle highlights (top percentiles matter).
    v = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if int(src.sum()) >= 400:
        lo = float(np.percentile(v[src], 15))
        hi = float(np.percentile(v[src], 96))
        mid = src & (v >= lo) & (v <= hi)
        if int(mid.sum()) >= 200:
            src = mid
        # Drop only the darkest rock-like pixels.
        dark = float(np.percentile(v[src], 12))
        bright = src & (v >= dark)
        if int(bright.sum()) >= 180:
            src = bright

    ay, ax = np.where(src)
    ay0, ay1 = int(np.percentile(ay, 3)), int(np.percentile(ay, 97))
    ax0, ax1 = int(np.percentile(ax, 3)), int(np.percentile(ax, 97))
    if ay1 <= ay0 + 6 or ax1 <= ax0 + 12:
        return None, None
    tile = rgb[ay0 : ay1 + 1, ax0 : ax1 + 1].astype(np.float32)
    return tile, src


def _seed_telea(rgb: np.ndarray, hard: np.ndarray) -> np.ndarray:
    import cv2

    try:
        return cv2.inpaint(rgb, (hard.astype(np.uint8) * 255), 9, cv2.INPAINT_TELEA)
    except Exception:
        out = rgb.copy()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
        if int(ring.sum()) >= 32:
            out[hard] = rgb[ring].mean(axis=0)
        return out


def _seed_sky_gradient(rgb: np.ndarray, hard: np.ndarray) -> np.ndarray:
    import cv2

    out = _seed_telea(rgb, hard)
    ys, xs = np.where(hard)
    if len(ys) == 0:
        return out
    y_top = int(ys.min())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y_top - max(24, int((ys.max() - y_top + 1) * 0.4)))
    band = np.zeros_like(hard, dtype=bool)
    band[y0:y_top, x0 : x1 + 1] = True
    band &= ~hard
    if int(band.sum()) < 64:
        return out
    # Vertical wash from band mean toward slightly brighter (sky).
    top_c = rgb[band].astype(np.float32).mean(axis=0)
    bot_c = np.clip(top_c * 1.04, 0, 255)
    y_bot = int(ys.max())
    span = max(1, y_bot - y_top)
    for y, x in zip(ys, xs):
        t = (y - y_top) / span
        out[y, x] = (top_c * (1.0 - t) + bot_c * t).astype(np.uint8)
    # Keep some TELEA edge detail.
    soft = _seed_telea(rgb, hard).astype(np.float32)
    blend = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 3.0)[:, :, None]
    mixed = out.astype(np.float32) * (1.0 - 0.35 * blend) + soft * (0.35 * blend)
    result = rgb.copy()
    result[hard] = np.clip(mixed[hard], 0, 255).astype(np.uint8)
    return result


def _seed_exemplar_from_map(rgb: np.ndarray, hard: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Stamp random source neighborhoods into the hole (keeps real texture energy)."""
    import cv2

    source = source.astype(bool) & (~hard.astype(bool))
    out = rgb.copy()
    if int(source.sum()) < 512:
        return _seed_telea(rgb, hard)

    # Prefer water above the hole (same bay band) — side/below samples smear rocks/people.
    ys_h, xs_h = np.where(hard)
    y_top = int(ys_h.min())
    x0, x1 = int(xs_h.min()), int(xs_h.max())
    h, w = hard.shape
    above = source.copy()
    above[y_top:, :] = False
    band = np.zeros_like(hard, dtype=bool)
    band[max(0, y_top - max(80, (int(ys_h.max()) - y_top + 1))) : y_top, max(0, x0 - 64) : min(w, x1 + 65)] = True
    prefer = above & band
    if int(prefer.sum()) >= 400:
        source = prefer
    elif int(above.sum()) >= 400:
        source = above

    # Bias source pick randomly across the bay — column-similar picks cause banding.
    ys_w, xs_w = np.where(source)
    rng = np.random.default_rng(42)
    patch = 32
    hh, ww = rgb.shape[:2]
    step = max(1, len(ys_h) // 1400)
    order = np.arange(0, len(ys_h), step)
    rng.shuffle(order)
    for yi in order:
        y, x = int(ys_h[yi]), int(xs_h[yi])
        # 70% fully random, 30% nearby-x (local continuity without streaks).
        if rng.random() < 0.70:
            yi_s = int(rng.integers(0, len(ys_w)))
        else:
            x_dist = np.abs(xs_w.astype(np.float32) - float(x))
            near = np.where(x_dist < max(64.0, (x1 - x0) * 0.8))[0]
            pool = near if len(near) >= 16 else np.arange(len(ys_w))
            yi_s = int(pool[int(rng.integers(0, len(pool)))])
        cy, cx = int(ys_w[yi_s]), int(xs_w[yi_s])
        y0, y1 = max(0, cy - patch // 2), min(hh, cy + patch // 2)
        x0p, x1p = max(0, cx - patch // 2), min(ww, cx + patch // 2)
        src = rgb[y0:y1, x0p:x1p]
        dy0, dy1 = max(0, y - patch // 2), min(hh, y + patch // 2)
        dx0, dx1 = max(0, x - patch // 2), min(ww, x + patch // 2)
        sh, sw = dy1 - dy0, dx1 - dx0
        if sh < 8 or sw < 8:
            continue
        src_r = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_LINEAR)
        region = hard[dy0:dy1, dx0:dx1]
        out[dy0:dy1, dx0:dx1][region] = src_r[region]

    msk = (hard.astype(np.uint8) * 255)
    try:
        # Fringe only — never soft TELEA plaque over the whole hole.
        soft = cv2.inpaint(out, msk, 2, cv2.INPAINT_NS)
        dist = cv2.distanceTransform(hard.astype(np.uint8) * 255, cv2.DIST_L2, 5)
        fringe = (hard & (dist <= 2.5)).astype(np.float32)[:, :, None]
        out = (
            out.astype(np.float32) * (1.0 - 0.35 * fringe) + soft.astype(np.float32) * (0.35 * fringe)
        ).astype(np.uint8)
    except Exception:
        pass
    return out
