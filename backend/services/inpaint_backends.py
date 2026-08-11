"""Inpainting backends — swap engines without changing the removal pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

RemovalMode = Literal["ai", "quick"]

_lama_torch_model = None
_lama_onnx_session = None
_sd_pipe = None
_sdxl_pipe = None
_powerpaint_pipe = None

# Object-removal prompts: synthesize plausible background, never people.
# Prefer scene_context.scene_prompts() at runtime; these remain as fallbacks.
_SD_REMOVE_PROMPT = (
    "high quality photograph, natural background continuation, seamless texture, "
    "photorealistic detail, consistent lighting, sharp focus"
)
_SD_REMOVE_NEGATIVE = (
    "person, people, human, man, woman, child, face, body, silhouette, ghost, "
    "arms, legs, clothing, blurry, deformed, low quality, watermark, text, artifacts, "
    "flat, smear, plaque, muddy, soft focus, painted, watercolor, plastic"
)
_SD_WATER_PROMPT = (
    "photorealistic ocean bay water with fine natural ripples and sunlit specular highlights, "
    "continuous rocky shoreline, distant green hills across the water, "
    "overcast daylight, seamless photograph, matching surrounding texture, no people"
)
_SD_WATER_NEGATIVE = (
    _SD_REMOVE_NEGATIVE
    + ", smooth blob, repeating pattern, stretched pixels, low-resolution patch"
)

# Threaded quality hint for generative passes (set by inpaint_objects).
_quality_tier: str = "balanced"


def set_removal_quality(tier: str | None) -> None:
    """best → generative routing + more steps / multi-seed; balanced → default cost;
    fast → classical engines and single-seed short generative passes."""
    global _quality_tier
    t = (tier or "balanced").strip().lower()
    if t in ("best", "best_quality"):
        _quality_tier = "best"
    elif t in ("fast", "quick", "speed"):
        _quality_tier = "fast"
    else:
        _quality_tier = "balanced"


def get_removal_quality() -> str:
    return _quality_tier


def _tier_num_steps(fast: int, balanced: int, best: int) -> int:
    tier = get_removal_quality()
    if tier == "fast":
        return fast
    return best if tier == "best" else balanced


def _tier_allows_multi_seed() -> bool:
    return get_removal_quality() != "fast"


def generative_engine_ready(scene: str | None = None) -> bool:
    """CUDA plus downloaded weights for the engine the erase router would pick.

    Mirrors _erase_engine_prefers_powerpaint so we never route a scene into a
    backend whose weights are absent (that would kick off a multi-GB download).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        from services.model_download import _powerpaint_present, _sdxl_inpaint_present
    except Exception:
        return False
    name = _erase_engine_name()
    if name in ("powerpaint", "ppt", "erase"):
        return bool(_powerpaint_present())
    if name in ("sdxl", "sd", "juggernaut"):
        return bool(_sdxl_inpaint_present())
    if scene is None:
        return bool(_powerpaint_present() or _sdxl_inpaint_present())
    if scene in ("structured", "generic", "vegetation"):
        return bool(_powerpaint_present())
    return bool(_sdxl_inpaint_present())


def generative_removal_readiness() -> dict:
    """Readiness of the Best Quality erase path, per scene family, for the UI."""
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    try:
        from services.model_download import _powerpaint_present, _sdxl_inpaint_present

        present = {
            "powerpaint": bool(_powerpaint_present()),
            "sdxl_inpaint": bool(_sdxl_inpaint_present()),
        }
    except Exception:
        present = {"powerpaint": False, "sdxl_inpaint": False}
    scenes = {scene: bool(generative_engine_ready(scene)) for scene in ("water", "structured")}
    return {
        "cuda": cuda,
        "engine": _erase_engine_name(),
        "scenes": scenes,
        "ready": all(scenes.values()),
        "models_present": present,
    }


class InpaintError(Exception):
    """Raised when inpainting fails or produces unusable output."""


def _ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def _prepare_torch_inputs(
    rgb: np.ndarray,
    mask: np.ndarray,
    device,
    pad_mod: int = 8,
) -> tuple:
    """CHW float tensors padded to modulo 8 (LaMa torch convention)."""
    import torch

    img_f = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    msk_f = (mask.astype(np.float32) / 255.0)
    if msk_f.ndim == 3:
        msk_f = msk_f[:, :, 0]
    msk_f = msk_f[np.newaxis, ...]

    ch, h, w = img_f.shape
    out_h = _ceil_modulo(h, pad_mod)
    out_w = _ceil_modulo(w, pad_mod)
    if out_h != h or out_w != w:
        img_f = np.pad(img_f, ((0, 0), (0, out_h - h), (0, out_w - w)), mode="symmetric")
        msk_f = np.pad(msk_f, ((0, 0), (0, out_h - h), (0, out_w - w)), mode="symmetric")

    img_t = torch.from_numpy(img_f).unsqueeze(0).to(device)
    msk_t = torch.from_numpy(msk_f).unsqueeze(0).to(device)
    msk_t = (msk_t > 0).float()
    return img_t, msk_t, h, w


def _get_lama_torch_model():
    global _lama_torch_model
    if _lama_torch_model is None:
        import torch

        from services.model_download import ensure_lama_big_torch

        path = ensure_lama_big_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        model.to(device)
        _lama_torch_model = (model, device)
    return _lama_torch_model


def _get_lama_onnx_session():
    global _lama_onnx_session
    if _lama_onnx_session is None:
        import onnxruntime as ort

        from services.model_download import ensure_lama_model

        path = ensure_lama_model()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = set(ort.get_available_providers())
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        _lama_onnx_session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    return _lama_onnx_session


def _lama_onnx_output_to_rgb(raw: np.ndarray) -> np.ndarray:
    hwc = raw.transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0:
        hwc = hwc * 255.0
    return np.clip(hwc, 0, 255).astype(np.uint8)


def inpaint_patch_torch(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Full-resolution LaMa big (PyTorch JIT) on a patch."""
    import torch

    model, device = _get_lama_torch_model()
    img_t, msk_t, orig_h, orig_w = _prepare_torch_inputs(rgb, mask, device)

    with torch.inference_mode():
        out = model(img_t, msk_t)
        result = out[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)[:orig_h, :orig_w]

    hard = (mask > 127)
    if hard.any():
        region = result[hard]
        if region.size and region.mean() > 252 and region.std() < 3:
            raise InpaintError("LaMa torch produced blank white fill in masked region")
    return result


def inpaint_patch_onnx(rgb: np.ndarray, mask: np.ndarray, size: int = 512) -> np.ndarray:
    """Fixed-size LaMa ONNX (fast) on a patch."""
    import cv2

    session = _get_lama_onnx_session()
    img_f = rgb.astype(np.float32) / 255.0
    msk_f = (mask.astype(np.float32) / 255.0)
    msk_f = (msk_f > 0).astype(np.float32)

    ph, pw = rgb.shape[:2]
    img_t = np.transpose(cv2.resize(img_f, (size, size)), (2, 0, 1))[None]
    msk_t = cv2.resize(msk_f, (size, size), interpolation=cv2.INTER_NEAREST)[None, None]

    out = session.run(None, {"image": img_t.astype(np.float32), "mask": msk_t.astype(np.float32)})[0]
    if out is None or out.size == 0:
        raise InpaintError("LaMa ONNX returned empty output")

    result = _lama_onnx_output_to_rgb(out[0])
    return cv2.resize(result, (pw, ph), interpolation=cv2.INTER_LANCZOS4)


def inpaint_patch_onnx_tiled(
    rgb: np.ndarray,
    mask: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Tiled LaMa ONNX for larger crops."""
    import cv2

    ch, cw = rgb.shape[:2]
    max_dim = max(ch, cw)
    if max_dim <= int(tile_size * 1.25):
        return inpaint_patch_onnx(rgb, mask, tile_size)

    tile = tile_size - overlap
    acc = np.zeros((ch, cw, 3), dtype=np.float32)
    wsum = np.zeros((ch, cw), dtype=np.float32)

    for y in range(0, ch, tile):
        for x in range(0, cw, tile):
            y1 = min(y + tile_size, ch)
            x1 = min(x + tile_size, cw)
            y0 = max(0, y1 - tile_size)
            x0 = max(0, x1 - tile_size)
            patch_rgb = rgb[y0:y1, x0:x1]
            patch_msk = mask[y0:y1, x0:x1]
            if patch_msk.max() == 0:
                continue

            ph, pw = patch_rgb.shape[:2]
            inpainted = cv2.resize(
                inpaint_patch_onnx(patch_rgb, patch_msk, tile_size),
                (pw, ph),
                interpolation=cv2.INTER_LANCZOS4,
            )

            py, px = np.mgrid[0:ph, 0:pw]
            wy = np.minimum(np.minimum(py + 1, ph - py), overlap)
            wx = np.minimum(np.minimum(px + 1, pw - px), overlap)
            weight = np.minimum(wy, wx).astype(np.float32)

            acc[y0:y1, x0:x1] += inpainted.astype(np.float32) * weight[:, :, None]
            wsum[y0:y1, x0:x1] += weight

    wsum = np.maximum(wsum, 1e-6)
    return (acc / wsum[:, :, None]).astype(np.uint8)


def inpaint_patch_opencv(
    rgb: np.ndarray,
    mask: np.ndarray,
    radius: int | None = None,
    method: str = "telea",
) -> np.ndarray:
    """OpenCV content-aware inpaint (TELEA or NS)."""
    import cv2

    hard = ((mask > 127).astype(np.uint8)) * 255
    ys, xs = np.where(hard > 0)
    if len(xs) == 0:
        return rgb.copy()
    hole = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    if radius is None:
        radius = max(5, int(min(rgb.shape[:2]) * 0.01), int(hole * 0.02))
        radius = min(radius, 18)
    else:
        radius = max(3, min(int(radius), 40))
    flag = cv2.INPAINT_NS if method.lower() == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(rgb, hard, radius, flag)


def _release_sd_pipe() -> None:
    """Free SD1.5 weights so SDXL / PowerPaint can claim VRAM."""
    global _sd_pipe
    if _sd_pipe is None:
        return
    try:
        import torch

        pipe, _device, _use_cuda = _sd_pipe
        del pipe
        _sd_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        _sd_pipe = None


def _release_sdxl_pipe() -> None:
    """Free SDXL weights (e.g. before falling back to SD1.5 on OOM)."""
    global _sdxl_pipe
    if _sdxl_pipe is None:
        return
    try:
        import torch

        pipe, _device, _use_cuda = _sdxl_pipe
        del pipe
        _sdxl_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        _sdxl_pipe = None


def _release_powerpaint_pipe() -> None:
    """Free PowerPaint weights before loading SDXL / SD1.5."""
    global _powerpaint_pipe
    if _powerpaint_pipe is None:
        return
    try:
        import torch

        pipe, _device, _use_cuda = _powerpaint_pipe
        del pipe
        _powerpaint_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        _powerpaint_pipe = None


def _get_sd_pipe():
    """Lazy-load Stable Diffusion 1.5 Inpainting (local weights under models/)."""
    global _sd_pipe
    if _sd_pipe is None:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        from services.model_download import ensure_sd_inpaint

        _release_sdxl_pipe()
        _release_powerpaint_pipe()
        model_dir = ensure_sd_inpaint()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        dtype = torch.float16 if use_cuda else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        _sd_pipe = (pipe, device, use_cuda)
    return _sd_pipe


def _get_sdxl_pipe(model_dir: str | None = None):
    """Lazy-load SDXL Inpainting — CUDA only (CPU is impractical for XL)."""
    global _sdxl_pipe
    if _sdxl_pipe is None or (
        model_dir is not None and getattr(_sdxl_pipe[0], "_pe_model_dir", None) != str(model_dir)
    ):
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline

        from services.model_download import ensure_sdxl_inpaint, _repair_sdxl_inpaint_snapshot

        if not torch.cuda.is_available():
            raise InpaintError("SDXL inpaint requires CUDA; falling back")

        if _sdxl_pipe is not None:
            _release_sdxl_pipe()

        _release_sd_pipe()
        _release_powerpaint_pipe()
        resolved = Path(model_dir) if model_dir else ensure_sdxl_inpaint()
        # Some community "inpainting" snapshots ship base-pipeline model_index / unet
        # in_channels=4 while weights are 9-channel inpaint — patch before load.
        _repair_sdxl_inpaint_snapshot(resolved)
        device = torch.device("cuda")
        dtype = torch.float16
        try:
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                str(resolved),
                torch_dtype=dtype,
                variant="fp16",
                local_files_only=True,
            )
        except Exception:
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                str(resolved),
                torch_dtype=dtype,
                local_files_only=True,
            )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        pipe._pe_model_dir = str(resolved)
        _sdxl_pipe = (pipe, device, True)
    return _sdxl_pipe


def _get_powerpaint_pipe():
    """Lazy-load PowerPaint V1 (SD1.5 inpaint + task tokens) — CUDA preferred."""
    global _powerpaint_pipe
    if _powerpaint_pipe is None:
        import importlib.util
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        from services.model_download import ensure_powerpaint

        if not torch.cuda.is_available():
            raise InpaintError("PowerPaint requires CUDA; falling back")

        _release_sdxl_pipe()
        _release_sd_pipe()
        model_dir = ensure_powerpaint()
        tok_path = model_dir / "power_paint_tokenizer.py"
        if not tok_path.is_file():
            raise InpaintError("PowerPaint tokenizer module missing")

        spec = importlib.util.spec_from_file_location("power_paint_tokenizer", tok_path)
        if spec is None or spec.loader is None:
            raise InpaintError("Failed to load PowerPaint tokenizer")
        tok_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tok_mod)

        device = torch.device("cuda")
        dtype = torch.float16
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )
        except Exception:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )
        pipe.tokenizer = tok_mod.PowerPaintTokenizer(pipe.tokenizer)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        _powerpaint_pipe = (pipe, device, True)
    return _powerpaint_pipe


def _detect_waterish_hole(rgb: np.ndarray, hard: np.ndarray) -> bool:
    """True when water/sky sits in the ring or in a band above the hole."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hs, ss, vs = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ring = cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0
    ring = ring & (~hard)
    if int(ring.sum()) >= 64:
        blueish = ((hs[ring] >= 85) & (hs[ring] <= 135) & (ss[ring] >= 12)).mean()
        grey_coast = ((ss[ring] < 48) & (vs[ring] > 125) & (vs[ring] < 235)).mean()
        if blueish > 0.28 or grey_coast > 0.35:
            return True
    ys, xs = np.where(hard)
    if len(ys) == 0:
        return False
    y_top = int(ys.min())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y_top - max(48, int((ys.max() - y_top + 1) * 0.5)))
    band = np.zeros_like(hard, dtype=bool)
    band[y0:y_top, x0 : x1 + 1] = True
    band &= ~hard
    if int(band.sum()) < 64:
        return False
    grey_coast = ((ss[band] < 48) & (vs[band] > 125) & (vs[band] < 235)).mean()
    blueish = ((hs[band] >= 85) & (hs[band] <= 135) & (ss[band] >= 12)).mean()
    return bool(blueish > 0.25 or grey_coast > 0.30)


def _seed_water_context(rgb: np.ndarray, hard: np.ndarray) -> np.ndarray:
    """Replace the hole with a TELEA water-ish seed so SD refines instead of inventing a blob."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    water = ((s < 48) & (v > 125) & (v < 235)) | ((h >= 85) & (h <= 135) & (s >= 12) & (v > 95))
    water = water & (~hard)
    guide = rgb.copy()
    if int(water.sum()) >= 256:
        mean_c = rgb[water].astype(np.float32).mean(axis=0)
        water_img = rgb.copy()
        water_img[~water] = mean_c
        water_blur = cv2.GaussianBlur(water_img, (0, 0), sigmaX=7.0)
        guide[~water & ~hard] = water_blur[~water & ~hard]
    msk = (hard.astype(np.uint8) * 255)
    try:
        return cv2.inpaint(guide, msk, 14, cv2.INPAINT_TELEA)
    except Exception:
        out = rgb.copy()
        if int(water.sum()) >= 64:
            out[hard] = rgb[water].mean(axis=0)
        return out


def _seed_water_texture_stamp(rgb: np.ndarray, hard: np.ndarray) -> np.ndarray:
    """Fill the hole by stamping real water patches (keeps ripple energy for SDXL)."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    water = ((s < 48) & (v > 125) & (v < 235)) | ((h >= 85) & (h <= 135) & (s >= 12) & (v > 95))
    water = water & (~hard)
    out = rgb.copy()
    if int(water.sum()) < 512:
        return _seed_water_context(rgb, hard)

    ys_w, xs_w = np.where(water)
    ys_h, xs_h = np.where(hard)
    # Tile random water neighborhoods into the hole.
    rng = np.random.default_rng(42)
    patch = 48
    hh, ww = rgb.shape[:2]
    for y, x in zip(ys_h[::max(1, len(ys_h) // 800)], xs_h[::max(1, len(xs_h) // 800)]):
        yi = int(rng.integers(0, len(ys_w)))
        cy, cx = int(ys_w[yi]), int(xs_w[yi])
        y0, y1 = max(0, cy - patch // 2), min(hh, cy + patch // 2)
        x0, x1 = max(0, cx - patch // 2), min(ww, cx + patch // 2)
        src = rgb[y0:y1, x0:x1]
        dy0, dy1 = max(0, y - patch // 2), min(hh, y + patch // 2)
        dx0, dx1 = max(0, x - patch // 2), min(ww, x + patch // 2)
        sh, sw = dy1 - dy0, dx1 - dx0
        if sh < 8 or sw < 8:
            continue
        src_r = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_LINEAR)
        region = hard[dy0:dy1, dx0:dx1]
        out[dy0:dy1, dx0:dx1][region] = src_r[region]

    # Soft classical pass only to kill hard stamp seams; keep high-freq energy.
    msk = (hard.astype(np.uint8) * 255)
    try:
        soft = cv2.inpaint(out, msk, 3, cv2.INPAINT_TELEA)
        blend = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 2.0)[:, :, None]
        out = (out.astype(np.float32) * (1.0 - 0.35 * blend) + soft.astype(np.float32) * (0.35 * blend)).astype(
            np.uint8
        )
    except Exception:
        pass
    return out


def _scene_for_hole(rgb: np.ndarray, hard: np.ndarray):
    from services.scene_context import classify_hole_scene

    return classify_hole_scene(rgb, hard)


def _prompts_for_scene(scene: str) -> tuple[str, str]:
    from services.scene_context import scene_prompts

    return scene_prompts(scene)  # type: ignore[arg-type]


def _seed_for_scene(rgb: np.ndarray, hard: np.ndarray, scene: str) -> np.ndarray:
    from services.scene_context import seed_hole_for_scene

    return seed_hole_for_scene(rgb, hard, scene)  # type: ignore[arg-type]


def inpaint_patch_sd(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_side: int | None = None,
    num_steps: int | None = None,
) -> np.ndarray:
    """Generative inpaint via Stable Diffusion 1.5 Inpainting.

    Runs at up to 640px (CPU) or 768px (CUDA) on the long side, then pastes
    the synthesized hole back into the original-resolution crop.

    Scene-aware seeding/prompts so water, sky, vegetation, and indoor fills
    don't all share the generic person-conditioned path.
    """
    import cv2
    import torch

    hard = mask > 127
    if not hard.any():
        return rgb.copy()

    pipe, device, use_cuda = _get_sd_pipe()
    scene = _scene_for_hole(rgb, hard)
    waterish = scene == "water" or _detect_waterish_hole(rgb, hard)
    open_fill = scene in ("water", "sky") or waterish

    # Water holes: CUDA 768 is the practical sweet spot for SD1.5 (1024 adds cost, little gain).
    if max_side is None:
        if use_cuda:
            max_side = 768
        else:
            max_side = 640
    if num_steps is None:
        base = 40 if use_cuda else 32
        num_steps = _tier_num_steps(int(base * 0.6), base, (base + 8) if use_cuda else base)

    # Grow the hole so hair fringes / silhouette edges get regenerated.
    grow_k = 21 if open_fill else 17
    grow_iters = 2 if open_fill else 1
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_k, grow_k))
    hard_grown = cv2.dilate(hard.astype(np.uint8), k_grow, iterations=grow_iters) > 0

    # Seed so SD is not staring at a person (which yields soft plaques).
    src_rgb = _seed_for_scene(rgb, hard_grown, "water" if waterish else scene)

    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    nw = _ceil_modulo(nw, 8)
    nh = _ceil_modulo(nh, 8)

    if nw != w or nh != h:
        img_small = cv2.resize(src_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        msk_small = cv2.resize(
            (hard_grown.astype(np.uint8) * 255),
            (nw, nh),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        img_small = src_rgb
        msk_small = (hard_grown.astype(np.uint8) * 255)

    # Binary mask only — soft SAM feathers cause semi-transparent person ghosts.
    msk_small = ((msk_small > 127).astype(np.uint8)) * 255

    pil_img = Image.fromarray(img_small)
    pil_msk = Image.fromarray(msk_small)

    prompt, negative = _prompts_for_scene("water" if waterish else scene)
    # Open fills: light refine on exemplar-seeded water so ripples survive.
    # High strength wiped texture and left soft plaques matching Quick.
    strength = 0.72 if open_fill else 1.0
    guidance = 6.5 if open_fill else (7.5 if use_cuda else 7.0)

    def _run_once(seed: int):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            image=pil_img,
            mask_image=pil_msk,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=generator,
        )
        try:
            if open_fill:
                return pipe(**kwargs).images[0]
            return pipe(**kwargs, padding_mask_crop=64).images[0]
        except TypeError:
            kwargs.pop("strength", None)
            return pipe(**kwargs).images[0]

    with torch.inference_mode():
        # Multi-seed on open / large / best-quality holes — keep most textured fill.
        large = int(hard_grown.sum()) > 4000
        if _tier_allows_multi_seed() and (
            open_fill or large or get_removal_quality() == "best"
        ) and use_cuda:
            seeds = (42, 77, 113)
        else:
            seeds = (42,)
        best_gen = None
        best_score = -1.0
        for seed in seeds:
            candidate = np.array(_run_once(seed).convert("RGB"))
            if candidate.shape[0] != h or candidate.shape[1] != w:
                candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_LANCZOS4)
            g = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
            tex = float(np.abs(g[hard_grown] - cv2.GaussianBlur(g, (0, 0), 2.0)[hard_grown]).mean())
            # Penalize washed / near-white plaques.
            mean_l = float(g[hard_grown].mean())
            wash_pen = max(0.0, mean_l - 210.0) * 0.08
            score = tex - wash_pen
            if score > best_score:
                best_score = score
                best_gen = candidate
        gen = best_gen

    # Hard paste: never leave original person faintly visible inside the hole.
    result = rgb.copy()
    result[hard_grown] = gen[hard_grown]

    region = result[hard]
    if region.size and region.mean() > 252 and region.std() < 3:
        raise InpaintError("SD inpaint produced blank white fill in masked region")
    return result


def inpaint_patch_sdxl(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_side: int | None = None,
    num_steps: int | None = None,
    model_dir: str | None = None,
) -> np.ndarray:
    """Generative inpaint via Stable Diffusion XL Inpainting (CUDA).

    Runs near 1024px on the long side — SDXL's native resolution — then pastes
    the synthesized hole back into the original-resolution crop.
    """
    import cv2
    import torch

    hard = mask > 127
    if not hard.any():
        return rgb.copy()

    pipe, device, use_cuda = _get_sdxl_pipe(model_dir=model_dir)
    if not use_cuda:
        raise InpaintError("SDXL inpaint requires CUDA")

    scene = _scene_for_hole(rgb, hard)
    waterish = scene == "water" or _detect_waterish_hole(rgb, hard)
    open_fill = scene in ("water", "sky") or waterish
    effective_scene = "water" if waterish else scene

    if max_side is None:
        max_side = 1024
    if num_steps is None:
        num_steps = _tier_num_steps(28, 40, 48)

    # Grow the hole so hair fringes / silhouette edges get regenerated.
    grow_k = 21 if open_fill else (19 if scene == "vegetation" else 17)
    grow_iters = 2 if open_fill else 1
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_k, grow_k))
    hard_grown = cv2.dilate(hard.astype(np.uint8), k_grow, iterations=grow_iters) > 0

    # Scene seed: stamp real texture so SDXL has energy to refine (not a person silhouette).
    src_rgb = _seed_for_scene(rgb, hard_grown, effective_scene)

    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    # SDXL prefers multiples of 8; keep aspect, pad later if needed via resize.
    nw = _ceil_modulo(nw, 8)
    nh = _ceil_modulo(nh, 8)

    if nw != w or nh != h:
        img_small = cv2.resize(src_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        msk_small = cv2.resize(
            (hard_grown.astype(np.uint8) * 255),
            (nw, nh),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        img_small = src_rgb
        msk_small = (hard_grown.astype(np.uint8) * 255)

    msk_small = ((msk_small > 127).astype(np.uint8)) * 255

    pil_img = Image.fromarray(img_small)
    pil_msk = Image.fromarray(msk_small)

    prompt, negative = _prompts_for_scene(effective_scene)
    # Seed carries real water texture; light refine harmonizes without soft plaques.
    # Prior high strength (~0.97) erased exemplar ripples → Best ≈ Quick grey plaque.
    strength = 0.68 if open_fill else 0.99
    guidance = 6.0 if open_fill else (7.0 if scene == "structured" else 7.5)

    def _run_once(seed: int):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            image=pil_img,
            mask_image=pil_msk,
            height=nh,
            width=nw,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=generator,
        )
        try:
            return pipe(**kwargs).images[0]
        except TypeError:
            kwargs.pop("height", None)
            kwargs.pop("width", None)
            return pipe(**kwargs).images[0]

    with torch.inference_mode():
        large = int(hard_grown.sum()) > 4000
        if _tier_allows_multi_seed() and (
            open_fill
            or large
            or get_removal_quality() == "best"
            or scene in ("vegetation", "structured")
        ):
            seeds = (42, 77, 113)
        else:
            seeds = (42,)
        best_gen = None
        best_score = -1.0
        for seed in seeds:
            candidate = np.array(_run_once(seed).convert("RGB"))
            if candidate.shape[0] != h or candidate.shape[1] != w:
                candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_LANCZOS4)
            g = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
            tex = float(np.abs(g[hard_grown] - cv2.GaussianBlur(g, (0, 0), 2.0)[hard_grown]).mean())
            mean_l = float(g[hard_grown].mean())
            wash_pen = max(0.0, mean_l - 210.0) * 0.08
            # Prefer fills whose mean luminance AND HF energy are near the ring.
            ring = cv2.dilate(hard_grown.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
            ring = ring & (~hard_grown)
            if int(ring.sum()) >= 64:
                ring_l = float(g[ring].mean())
                wash_pen += abs(mean_l - ring_l) * 0.05
                ring_tex = float(
                    np.abs(g[ring] - cv2.GaussianBlur(g, (0, 0), 2.0)[ring]).mean()
                )
                # Penalize soft plaques relative to ring ripple energy.
                if ring_tex > 1.0:
                    wash_pen += max(0.0, ring_tex * 0.85 - tex) * 0.35
            score = tex - wash_pen
            if score > best_score:
                best_score = score
                best_gen = candidate
        gen = best_gen

    result = rgb.copy()
    result[hard_grown] = gen[hard_grown]

    region = result[hard]
    if region.size and region.mean() > 252 and region.std() < 3:
        raise InpaintError("SDXL inpaint produced blank white fill in masked region")
    return result


def inpaint_patch_powerpaint(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    max_side: int | None = None,
    num_steps: int | None = None,
) -> np.ndarray:
    """Object erase via PowerPaint V1 (SD1.5 inpaint + P_ctxt / P_obj task tokens).

    Uses high CFG object-removal prompting so the model fills background context
    instead of regenerating a soft person-shaped plaque.
    """
    import cv2
    import torch

    hard = mask > 127
    if not hard.any():
        return rgb.copy()

    pipe, device, use_cuda = _get_powerpaint_pipe()
    if not use_cuda:
        raise InpaintError("PowerPaint requires CUDA")

    scene = _scene_for_hole(rgb, hard)
    waterish = scene == "water" or _detect_waterish_hole(rgb, hard)
    open_fill = scene in ("water", "sky") or waterish
    effective_scene = "water" if waterish else scene

    if max_side is None:
        max_side = 768
    if num_steps is None:
        num_steps = _tier_num_steps(28, 40, 48)

    grow_k = 21 if open_fill else 17
    grow_iters = 2 if open_fill else 1
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_k, grow_k))
    hard_grown = cv2.dilate(hard.astype(np.uint8), k_grow, iterations=grow_iters) > 0

    # Seed holes so the UNet is not conditioned on a person silhouette.
    src_rgb = _seed_for_scene(rgb, hard_grown, effective_scene)

    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    nw = _ceil_modulo(nw, 8)
    nh = _ceil_modulo(nh, 8)

    if nw != w or nh != h:
        img_small = cv2.resize(src_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        msk_small = cv2.resize(
            (hard_grown.astype(np.uint8) * 255),
            (nw, nh),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        img_small = src_rgb
        msk_small = (hard_grown.astype(np.uint8) * 255)

    msk_small = ((msk_small > 127).astype(np.uint8)) * 255
    pil_img = Image.fromarray(img_small)
    pil_msk = Image.fromarray(msk_small)

    # PowerPaint object-removal: P_ctxt positive + P_obj negative, CFG >= 10.
    base_prompt, base_neg = _prompts_for_scene(effective_scene)
    prompt = f"{base_prompt}, empty scene P_ctxt"
    negative = f"{base_neg} P_obj"
    strength = 0.70 if open_fill else 0.99
    guidance = 11.0 if open_fill else 12.0

    def _run_once(seed: int):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            image=pil_img,
            mask_image=pil_msk,
            height=nh,
            width=nw,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=generator,
        )
        try:
            return pipe(**kwargs).images[0]
        except TypeError:
            kwargs.pop("height", None)
            kwargs.pop("width", None)
            return pipe(**kwargs).images[0]

    with torch.inference_mode():
        seeds = (42, 77) if waterish and _tier_allows_multi_seed() else (42,)
        best_gen = None
        best_score = -1.0
        for seed in seeds:
            candidate = np.array(_run_once(seed).convert("RGB"))
            if candidate.shape[0] != h or candidate.shape[1] != w:
                candidate = cv2.resize(candidate, (w, h), interpolation=cv2.INTER_LANCZOS4)
            g = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
            tex = float(np.abs(g[hard_grown] - cv2.GaussianBlur(g, (0, 0), 2.0)[hard_grown]).mean())
            mean_l = float(g[hard_grown].mean())
            wash_pen = max(0.0, mean_l - 210.0) * 0.08
            ring = cv2.dilate(hard_grown.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
            ring = ring & (~hard_grown)
            if int(ring.sum()) >= 64:
                ring_l = float(g[ring].mean())
                wash_pen += abs(mean_l - ring_l) * 0.05
            score = tex - wash_pen
            if score > best_score:
                best_score = score
                best_gen = candidate
        gen = best_gen

    result = rgb.copy()
    result[hard_grown] = gen[hard_grown]

    region = result[hard]
    if region.size and region.mean() > 252 and region.std() < 3:
        raise InpaintError("PowerPaint produced blank white fill in masked region")
    return result


class InpaintBackend(ABC):
    is_generative: bool = False
    engine: str = "classical"

    @abstractmethod
    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ...


class GenerativeRemoveBackend(InpaintBackend):
    """SDXL (CUDA) or SD1.5 Inpainting — Best Quality / AI Remove path."""

    is_generative = True
    engine = "sdxl"

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if _prefer_powerpaint():
            try:
                import torch

                if torch.cuda.is_available():
                    self.engine = "powerpaint"
                    return inpaint_patch_powerpaint(rgb, mask)
            except InpaintError:
                pass
            except Exception:
                pass
        try:
            import torch

            if torch.cuda.is_available():
                self.engine = "sdxl"
                return inpaint_patch_sdxl(rgb, mask)
        except InpaintError:
            pass
        except Exception:
            pass
        try:
            self.engine = "sd15"
            return inpaint_patch_sd(rgb, mask)
        except InpaintError:
            raise
        except Exception as exc:
            raise InpaintError(str(exc)) from exc


class AiRemoveBackend(InpaintBackend):
    """Best Quality erase path (CUDA): scene-routed PowerPaint / SDXL → SD1.5 → LaMa.

    Default PHOTO_ENHANCER_ERASE_ENGINE=auto:
      - structured/indoor → PowerPaint (dedicated erase) then SDXL
      - water/sky/open → SDXL (better texture energy) then PowerPaint
    Force with powerpaint | sdxl | auto.
    """

    is_generative = True
    engine = "sdxl"

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hard = mask > 127
        prefer_ppt = _erase_engine_prefers_powerpaint(rgb, hard)
        order = ("powerpaint", "sdxl") if prefer_ppt else ("sdxl", "powerpaint")
        gen_exc: Exception | None = None

        for name in order:
            try:
                import torch

                if not torch.cuda.is_available():
                    break
                if name == "powerpaint":
                    self.engine = "powerpaint"
                    return inpaint_patch_powerpaint(rgb, mask)
                self.engine = "sdxl"
                return inpaint_patch_sdxl(rgb, mask)
            except Exception as exc:
                if gen_exc is None:
                    gen_exc = exc
                try:
                    if name == "powerpaint":
                        _release_powerpaint_pipe()
                    else:
                        _release_sdxl_pipe()
                except Exception:
                    pass

        try:
            self.engine = "sd15"
            return inpaint_patch_sd(rgb, mask)
        except Exception as sd15_exc:
            if gen_exc is None:
                gen_exc = sd15_exc
            try:
                out = inpaint_patch_torch(rgb, mask)
                self.is_generative = False
                self.engine = "lama"
                return out
            except InpaintError:
                raise
            except Exception as lama_exc:
                raise InpaintError(
                    f"Generative inpaint failed ({gen_exc}); LaMa fallback failed ({lama_exc})"
                ) from lama_exc


def _erase_engine_name() -> str:
    import os

    return os.environ.get("PHOTO_ENHANCER_ERASE_ENGINE", "auto").strip().lower() or "auto"


def _prefer_powerpaint() -> bool:
    return _erase_engine_name() in ("powerpaint", "ppt", "erase")


def _erase_engine_prefers_powerpaint(rgb: np.ndarray, hard: np.ndarray) -> bool:
    """Scene-aware default: PowerPaint for structured; SDXL for open water/sky."""
    name = _erase_engine_name()
    if name in ("powerpaint", "ppt", "erase"):
        return True
    if name in ("sdxl", "sd", "juggernaut"):
        return False
    # auto
    try:
        from services.scene_context import classify_hole_scene

        scene = classify_hole_scene(rgb, hard)
        # Structured indoor: PowerPaint erase is better than SDXL mega-hole smears.
        return scene in ("structured", "generic", "vegetation")
    except Exception:
        return False


class StructuredClassicalBackend(InpaintBackend):
    """LaMa + PatchMatch path used when structured mega-holes would melt floors."""

    is_generative = False
    engine = "lama_pm"

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            return inpaint_patch_torch(rgb, mask)
        except Exception:
            return inpaint_patch_onnx_tiled(rgb, mask)


def warm_inpaint_engines() -> dict:
    """Load the primary erase engine onto GPU (one pipe — avoid VRAM thrash)."""
    import torch

    status: dict = {
        "cuda": bool(torch.cuda.is_available()),
        "sdxl": False,
        "powerpaint": False,
        "sd15": False,
        "warmed": None,
    }
    if not status["cuda"]:
        # CPU: warm Quick LaMa ONNX only.
        try:
            _get_lama_onnx_session()
            status["warmed"] = "lama_onnx"
        except Exception:
            pass
        return status

    # Prefer PowerPaint (dedicated erase) when present; else SDXL.
    try:
        from services.model_download import _powerpaint_present, _sdxl_inpaint_present

        if _powerpaint_present() or _prefer_powerpaint():
            _get_powerpaint_pipe()
            status["powerpaint"] = True
            status["warmed"] = "powerpaint"
            return status
        if _sdxl_inpaint_present():
            _get_sdxl_pipe()
            status["sdxl"] = True
            status["warmed"] = "sdxl"
            return status
    except Exception:
        pass
    try:
        _get_sd_pipe()
        status["sd15"] = True
        status["warmed"] = "sd15"
    except Exception:
        pass
    return status


class LamaTorchBackend(InpaintBackend):
    """PyTorch big-LaMa at native crop resolution (legacy / fallback)."""

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            return inpaint_patch_torch(rgb, mask)
        except InpaintError:
            raise
        except Exception as exc:
            raise InpaintError(str(exc)) from exc


class QuickRemoveBackend(InpaintBackend):
    """ONNX LaMa 512 — fast tiled inference."""

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            return inpaint_patch_onnx_tiled(rgb, mask)
        except InpaintError:
            raise
        except Exception as exc:
            raise InpaintError(str(exc)) from exc


def get_backend(mode: RemovalMode) -> InpaintBackend:
    if mode == "quick":
        return QuickRemoveBackend()
    return AiRemoveBackend()
