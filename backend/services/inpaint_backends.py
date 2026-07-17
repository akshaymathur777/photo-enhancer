"""Inpainting backends — swap engines without changing the removal pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from PIL import Image

RemovalMode = Literal["ai", "quick"]

_lama_torch_model = None
_lama_onnx_session = None
_sd_pipe = None

# Object-removal prompts: synthesize plausible background, never people.
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
            wy = np.minimum(py + 1, ph - py, overlap)
            wx = np.minimum(px + 1, pw - px, overlap)
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


def _get_sd_pipe():
    """Lazy-load Stable Diffusion 1.5 Inpainting (local weights under models/)."""
    global _sd_pipe
    if _sd_pipe is None:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        from services.model_download import ensure_sd_inpaint

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

    On coastal/water holes, seeds the mask with a classical water fill and runs
    SD at reduced strength so it refines ripples instead of inventing a soft plaque.
    """
    import cv2
    import torch

    hard = mask > 127
    if not hard.any():
        return rgb.copy()

    pipe, device, use_cuda = _get_sd_pipe()
    waterish = _detect_waterish_hole(rgb, hard)

    # Water holes: CUDA 768 is the practical sweet spot for SD1.5 (1024 adds cost, little gain).
    if max_side is None:
        if use_cuda:
            max_side = 768
        else:
            max_side = 640
    if num_steps is None:
        num_steps = 40 if use_cuda else 32

    # Grow the hole so hair fringes / silhouette edges get regenerated.
    grow_k = 21 if waterish else 17
    grow_iters = 2 if waterish else 1
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_k, grow_k))
    hard_grown = cv2.dilate(hard.astype(np.uint8), k_grow, iterations=grow_iters) > 0

    # Water: seed the hole so SD is not staring at a person (which yields white plaques).
    src_rgb = _seed_water_context(rgb, hard_grown) if waterish else rgb

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

    prompt = _SD_WATER_PROMPT if waterish else _SD_REMOVE_PROMPT
    negative = _SD_WATER_NEGATIVE if waterish else _SD_REMOVE_NEGATIVE
    # Water: mostly generative on a water seed (person gone) so ripples can appear.
    # Too-low strength just keeps the soft TELEA plaque.
    strength = 0.88 if waterish else 1.0
    guidance = 7.0 if waterish else (7.5 if use_cuda else 7.0)

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
            if waterish:
                return pipe(**kwargs).images[0]
            return pipe(**kwargs, padding_mask_crop=64).images[0]
        except TypeError:
            kwargs.pop("strength", None)
            return pipe(**kwargs).images[0]

    with torch.inference_mode():
        # Water+CUDA: two seeds, keep the more textured / less washed fill.
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


class InpaintBackend(ABC):
    is_generative: bool = False

    @abstractmethod
    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ...


class GenerativeRemoveBackend(InpaintBackend):
    """Stable Diffusion 1.5 Inpainting — Best Quality / AI Remove path."""

    is_generative = True

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            return inpaint_patch_sd(rgb, mask)
        except InpaintError:
            raise
        except Exception as exc:
            raise InpaintError(str(exc)) from exc


class AiRemoveBackend(InpaintBackend):
    """Generative SD1.5 inpaint with LaMa torch fallback."""

    is_generative = True

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            return inpaint_patch_sd(rgb, mask)
        except Exception as gen_exc:
            try:
                out = inpaint_patch_torch(rgb, mask)
                # Mark that we fell back so callers can apply LaMa-oriented postprocess.
                self.is_generative = False
                return out
            except InpaintError:
                raise
            except Exception as lama_exc:
                raise InpaintError(
                    f"Generative inpaint failed ({gen_exc}); LaMa fallback failed ({lama_exc})"
                ) from lama_exc


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
