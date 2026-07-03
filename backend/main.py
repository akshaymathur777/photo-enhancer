from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
import json

from dotenv import load_dotenv

from paths import configure_runtime_paths, get_env_file, get_frontend_dist

configure_runtime_paths()
load_dotenv(get_env_file())

from services.ai_modes import (
    clamp_scale_for_mode,
    list_modes,
    normalize_mode,
    resolve_bg_model,
    resolve_inpaint_mode,
)
from services.ai_styles import apply_style, list_styles, normalize_style
from services.image_io import open_image_bytes
from services.image_processing import (
    BG_MODELS,
    adjust_image,
    auto_enhance,
    crop_rotate,
    denoise,
    image_to_bytes,
    remove_background,
    replace_background,
    sharpen,
    upscale,
)
from services.effects import apply_blur, apply_bokeh, apply_grain, apply_prism, apply_vignette
from services.relight import apply_relight, list_relight_presets
from services.repair import ai_repair, photo_restore_lite
from services.portrait import (
    fix_red_eye,
    hair_tint,
    lip_blush_overlay,
    portrait_magic,
    skin_smooth,
    whiten_teeth,
)
from services.chat_agent import (
    action_meta,
    chat_backend_status,
    execute_action,
    parse_intent,
    result_to_base64,
)
from services.object_removal import (
    auto_detect_people_mask,
    inpaint_objects,
    mask_overlay_preview,
    segment_from_points,
)
from services.upscale_limits import MAX_OUTPUT_MP, max_allowed_scale

app = FastAPI(title="Photo Enhancer", version="2.0.0")

DIST = get_frontend_dist()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE = 20 * 1024 * 1024  # 20 MB


def _processing_error(exc: Exception) -> HTTPException:
    """Map processing failures to client-friendly HTTP errors instead of opaque 500s."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()

    if isinstance(exc, ValueError):
        return HTTPException(400, message)
    if isinstance(exc, FileNotFoundError):
        return HTTPException(503, f"Required model is not available: {message}")
    if "download" in lowered or "model" in lowered and "load" in lowered:
        return HTTPException(503, f"Model is still loading or unavailable: {message}")
    if "cuda" in lowered or "out of memory" in lowered:
        return HTTPException(503, f"GPU memory error — try a smaller image: {message}")
    return HTTPException(500, f"Processing failed: {message}")


async def _load_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        return open_image_bytes(data, file.filename or "")
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models/status")
def api_models_status():
    from services.model_download import get_models_status

    return get_models_status()


@app.get("/api/models/download-progress")
def api_models_download_progress():
    from services.model_download import get_download_progress

    return get_download_progress()


@app.post("/api/models/download-all")
def api_models_download_all():
    from services.model_download import download_all_models

    return download_all_models()


@app.post("/api/models/download/{model_id}")
def api_models_download_one(model_id: str):
    from services.model_download import download_model

    try:
        return download_model(model_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/open")
async def api_open(file: UploadFile = File(...)):
    """Open any image (including HEIC) and return a browser-friendly JPEG preview."""
    image = await _load_image(file)
    if image.mode == "RGBA":
        fmt, media = "PNG", "image/png"
        out = image
    else:
        fmt, media = "JPEG", "image/jpeg"
        out = image.convert("RGB")
    return Response(content=image_to_bytes(out, fmt), media_type=media)


@app.get("/api/modes")
def api_list_modes():
    return {"modes": list_modes(), "default": "balanced"}


@app.get("/api/styles")
def api_list_styles():
    return {"styles": list_styles()}


@app.get("/api/models")
def list_models():
    return {
        "background_removal": [
            {"id": k, "name": v, "label": k.replace("_", " ").title()}
            for k, v in BG_MODELS.items()
        ],
        "modes": list_modes(),
        "styles": list_styles(),
        "tools": [
            {"id": "remove-bg", "name": "Remove Background", "model": "BiRefNet Massive"},
            {"id": "replace-bg", "name": "Replace Background", "model": "BiRefNet Massive"},
            {"id": "remove-objects", "name": "Remove Objects", "model": "SAM 2.1 + LaMa AI/Quick"},
            {"id": "ai-styles", "name": "AI Styles", "model": "Creative presets"},
            {"id": "auto-enhance", "name": "Auto Enhance"},
            {"id": "adjust", "name": "Adjust (brightness/contrast/saturation)"},
            {"id": "denoise", "name": "Denoise"},
            {"id": "sharpen", "name": "Sharpen"},
            {"id": "upscale", "name": "Upscale", "model": "Real-ESRGAN"},
        ],
        "upscale": {
            "max_output_megapixels": MAX_OUTPUT_MP // 1_000_000,
            "scales": [2, 4],
        },
    }


@app.post("/api/apply-style")
async def api_apply_style(
    file: UploadFile = File(...),
    style: str = Form("cinematic"),
    mode: str = Form("balanced"),
):
    image = await _load_image(file)
    style_id = normalize_style(style)
    _ = normalize_mode(mode)
    result = apply_style(image, style_id)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/remove-background")
async def api_remove_background(
    file: UploadFile = File(...),
    model: str = Form(None),
    mode: str = Form("balanced"),
):
    image = await _load_image(file)
    model_key = resolve_bg_model(normalize_mode(mode), model or None)
    try:
        result = remove_background(image, model_key)
    except Exception as exc:
        raise _processing_error(exc) from exc
    return Response(content=image_to_bytes(result), media_type="image/png")


@app.post("/api/replace-background")
async def api_replace_background(
    file: UploadFile = File(...),
    model: str = Form(None),
    mode: str = Form("balanced"),
    bg_type: str = Form("color"),
    color: str = Form("#ffffff"),
    blur_radius: int = Form(20),
    bg_image: UploadFile = File(None),
):
    image = await _load_image(file)
    model_key = resolve_bg_model(normalize_mode(mode), model or None)
    bg_img = None
    if bg_type == "image" and bg_image is not None:
        bg_img = await _load_image(bg_image)
    result = replace_background(image, bg_type, color, blur_radius, model_key, bg_img)
    fmt = "PNG" if bg_type == "transparent" else "JPEG"
    media = "image/png" if fmt == "PNG" else "image/jpeg"
    return Response(content=image_to_bytes(result, fmt), media_type=media)


@app.post("/api/crop-rotate")
async def api_crop_rotate(
    file: UploadFile = File(...),
    left: int = Form(0),
    top: int = Form(0),
    width: int = Form(0),
    height: int = Form(0),
    angle: float = Form(0.0),
    flip_h: bool = Form(False),
    flip_v: bool = Form(False),
):
    image = await _load_image(file)
    w, h = image.size
    if width <= 0:
        width = w
    if height <= 0:
        height = h
    result = crop_rotate(image, left, top, width, height, angle, flip_h, flip_v)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/bokeh")
async def api_bokeh(
    file: UploadFile = File(...),
    blur_radius: int = Form(25),
    shape: str = Form("circle"),
    depth_aware: bool = Form(False),
    model: str = Form(None),
    mode: str = Form("balanced"),
):
    image = await _load_image(file)
    model_key = resolve_bg_model(normalize_mode(mode), model or None)
    try:
        result = apply_bokeh(image, blur_radius, shape, depth_aware, model_key)
    except Exception as exc:
        raise _processing_error(exc) from exc
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/blur")
async def api_blur(
    file: UploadFile = File(...),
    radius: int = Form(15),
    selective: bool = Form(False),
    model: str = Form(None),
    mode: str = Form("balanced"),
):
    image = await _load_image(file)
    model_key = resolve_bg_model(normalize_mode(mode), model or None)
    result = apply_blur(image, radius, selective, model_key)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/prism")
async def api_prism(
    file: UploadFile = File(...),
    strength: float = Form(0.6),
):
    image = await _load_image(file)
    result = apply_prism(image, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/vignette")
async def api_vignette(
    file: UploadFile = File(...),
    strength: float = Form(0.5),
):
    image = await _load_image(file)
    result = apply_vignette(image, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/grain")
async def api_grain(
    file: UploadFile = File(...),
    amount: float = Form(0.04),
):
    image = await _load_image(file)
    result = apply_grain(image, amount)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.get("/api/relight/presets")
def api_relight_presets():
    return {"presets": list_relight_presets()}


@app.post("/api/relight")
async def api_relight(
    file: UploadFile = File(...),
    preset: str = Form("even_light"),
    strength: float = Form(1.0),
):
    image = await _load_image(file)
    result = apply_relight(image, preset, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/ai-repair")
async def api_ai_repair(
    file: UploadFile = File(...),
    auto_detect: bool = Form(False),
    mask_file: UploadFile = File(None),
):
    image = await _load_image(file)
    mask = None
    if mask_file is not None:
        mask_data = await mask_file.read()
        mask = Image.open(__import__("io").BytesIO(mask_data)).convert("L")
    try:
        result = ai_repair(image, mask, auto_detect)
    except Exception as exc:
        raise _processing_error(exc) from exc
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/photo-restore")
async def api_photo_restore(
    file: UploadFile = File(...),
    strength: float = Form(1.0),
):
    image = await _load_image(file)
    result = photo_restore_lite(image, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/portrait-magic")
async def api_portrait_magic(
    file: UploadFile = File(...),
    strength: float = Form(1.0),
):
    image = await _load_image(file)
    try:
        result = portrait_magic(image, strength)
    except Exception as exc:
        raise _processing_error(exc) from exc
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/skin-smooth")
async def api_skin_smooth(
    file: UploadFile = File(...),
    strength: float = Form(0.6),
):
    image = await _load_image(file)
    result = skin_smooth(image, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/fix-red-eye")
async def api_fix_red_eye(file: UploadFile = File(...)):
    image = await _load_image(file)
    result = fix_red_eye(image)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/whiten-teeth")
async def api_whiten_teeth(
    file: UploadFile = File(...),
    strength: float = Form(0.5),
):
    image = await _load_image(file)
    result = whiten_teeth(image, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/hair-tint")
async def api_hair_tint(
    file: UploadFile = File(...),
    hue_shift: float = Form(0.0),
    saturation: float = Form(0.3),
):
    image = await _load_image(file)
    try:
        result = hair_tint(image, hue_shift, saturation)
    except Exception as exc:
        raise _processing_error(exc) from exc
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/lip-blush")
async def api_lip_blush(
    file: UploadFile = File(...),
    lip_color: str = Form("#cc4444"),
    blush_color: str = Form("#ff88aa"),
    intensity: float = Form(0.35),
):
    image = await _load_image(file)
    result = lip_blush_overlay(image, lip_color, blush_color, intensity)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/batch/enhance")
async def api_batch_enhance(
    input_dir: str = Form(...),
    output_dir: str = Form(...),
    operation: str = Form("auto_enhance"),
):
    from services.batch import process_folder

    ops = {
        "auto_enhance": lambda img: auto_enhance(img),
        "portrait_magic": lambda img: portrait_magic(img),
        "photo_restore": lambda img: photo_restore_lite(img),
    }
    fn = ops.get(operation, ops["auto_enhance"])
    try:
        result = process_folder(input_dir, output_dir, fn)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@app.post("/api/auto-enhance")
async def api_auto_enhance(
    file: UploadFile = File(...),
    strength: float = Form(1.0),
):
    image = await _load_image(file)
    result = auto_enhance(image, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/adjust")
async def api_adjust(
    file: UploadFile = File(...),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    saturation: float = Form(1.0),
    exposure: float = Form(1.0),
    sharpness: float = Form(1.0),
):
    image = await _load_image(file)
    result = adjust_image(image, brightness, contrast, saturation, exposure, sharpness)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/denoise")
async def api_denoise(
    file: UploadFile = File(...),
    strength: int = Form(10),
):
    image = await _load_image(file)
    result = denoise(image, strength)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/sharpen")
async def api_sharpen(
    file: UploadFile = File(...),
    amount: float = Form(1.5),
):
    image = await _load_image(file)
    result = sharpen(image, amount)
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/upscale")
async def api_upscale(
    file: UploadFile = File(...),
    scale: int = Form(2),
    mode: str = Form("balanced"),
):
    if scale not in (2, 4):
        raise HTTPException(400, "Scale must be 2 or 4")
    image = await _load_image(file)
    w, h = image.size
    try:
        effective_scale, warning = clamp_scale_for_mode(w, h, scale, normalize_mode(mode))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        result = upscale(image, effective_scale)
    except Exception as exc:
        raise _processing_error(exc) from exc
    headers: dict[str, str] = {"X-Upscale-Scale": str(effective_scale)}
    if warning:
        headers["X-Upscale-Warning"] = warning
    return Response(
        content=image_to_bytes(result, "PNG"),
        media_type="image/png",
        headers=headers,
    )


@app.get("/api/upscale/limits")
def upscale_limits(width: int, height: int):
    if width < 1 or height < 1:
        raise HTTPException(400, "width and height must be positive")
    allowed = max_allowed_scale(width, height)
    return {
        "max_output_megapixels": MAX_OUTPUT_MP // 1_000_000,
        "max_scale": allowed,
        "can_upscale": allowed is not None,
    }


def _parse_points_labels(points_json: str, labels_json: str) -> tuple[list[list[float]], list[int]]:
    try:
        points = json.loads(points_json)
        labels = json.loads(labels_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid points or labels JSON") from exc

    if not isinstance(points, list) or not isinstance(labels, list):
        raise HTTPException(400, "points and labels must be JSON arrays")
    if len(points) != len(labels) or len(points) == 0:
        raise HTTPException(400, "points and labels must be non-empty and equal length")

    parsed_points: list[list[float]] = []
    parsed_labels: list[int] = []
    for pt, lbl in zip(points, labels):
        if not isinstance(pt, list) or len(pt) != 2:
            raise HTTPException(400, "Each point must be [x, y]")
        parsed_points.append([float(pt[0]), float(pt[1])])
        parsed_labels.append(1 if int(lbl) else 0)

    return parsed_points, parsed_labels


@app.post("/api/segment")
async def api_segment(
    file: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...),
    preview: bool = Form(True),
):
    """Segment objects from click points. Returns mask PNG or overlay preview."""
    image = await _load_image(file)
    parsed_points, parsed_labels = _parse_points_labels(points, labels)
    try:
        mask = segment_from_points(image, parsed_points, parsed_labels)
    except Exception as exc:
        raise _processing_error(exc) from exc

    if preview:
        result = mask_overlay_preview(image, mask)
        return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")

    return Response(content=image_to_bytes(mask, "PNG"), media_type="image/png")


@app.get("/api/chat/status")
def chat_status():
    return chat_backend_status()


@app.post("/api/chat")
async def api_chat(
    file: UploadFile = File(...),
    message: str = Form(...),
    history: str = Form("[]"),
):
    image = await _load_image(file)
    try:
        hist = json.loads(history)
        if not isinstance(hist, list):
            hist = []
    except json.JSONDecodeError:
        hist = []

    status = chat_backend_status()
    action = parse_intent(message, hist)
    try:
        result_img, reply, fmt = execute_action(image, action)
    except Exception as exc:
        return {
            "reply": f"Could not apply that edit: {exc}",
            "image_base64": None,
            "tool": action.tool if action.tool != "none" else None,
            "needs_clicks": action.needs_clicks,
            "ai_enabled": status["ai_enabled"],
            "backend": status["backend"],
            "suggestions": action.suggestions or ["Try again", "Make it brighter"],
            "clarify": action.clarify,
            "meta": action_meta(action),
        }

    image_base64 = None
    if result_img is not None and fmt:
        image_base64 = result_to_base64(result_img, fmt)

    return {
        "reply": reply,
        "image_base64": image_base64,
        "tool": action.tool if action.tool != "none" else None,
        "needs_clicks": action.needs_clicks,
        "ai_enabled": status["ai_enabled"],
        "backend": status["backend"],
        "suggestions": action.suggestions,
        "clarify": action.clarify,
        "meta": action_meta(action),
    }


@app.post("/api/auto-segment-people")
async def api_auto_segment_people(
    file: UploadFile = File(...),
    preview: bool = Form(True),
):
    """Detect all people and return mask overlay or raw mask PNG."""
    image = await _load_image(file)
    try:
        mask = auto_detect_people_mask(image)
    except Exception as exc:
        raise _processing_error(exc) from exc

    if not mask.getbbox():
        raise HTTPException(400, "No people detected in this image")

    if preview:
        result = mask_overlay_preview(image, mask)
        return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")

    return Response(content=image_to_bytes(mask, "PNG"), media_type="image/png")


@app.post("/api/auto-remove-people")
async def api_auto_remove_people(
    file: UploadFile = File(...),
    mode: str = Form("balanced"),
    inpaint_mode: str = Form(None),
    edge_softness: float = Form(1.0),
):
    """Detect all people, build a mask, and inpaint them away in one step."""
    image = await _load_image(file)
    removal = resolve_inpaint_mode(normalize_mode(mode), inpaint_mode)
    softness = max(0.5, min(float(edge_softness), 2.0))

    try:
        mask = auto_detect_people_mask(image)
    except Exception as exc:
        raise _processing_error(exc) from exc

    if not mask.getbbox():
        raise HTTPException(400, "No people detected in this image")

    try:
        result = inpaint_objects(image, mask, mode=removal, edge_softness=softness)
    except Exception as exc:
        raise _processing_error(exc) from exc
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.post("/api/remove-objects")
async def api_remove_objects(
    file: UploadFile = File(...),
    points: str = Form(None),
    labels: str = Form(None),
    mask_file: UploadFile = File(None),
    edge_softness: float = Form(1.0),
    mode: str = Form("balanced"),
    inpaint_mode: str = Form(None),
):
    """Remove selected objects via inpainting. Provide click points, brush mask, or mask file."""
    image = await _load_image(file)

    if mask_file is not None:
        mask_data = await mask_file.read()
        mask = Image.open(__import__("io").BytesIO(mask_data)).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
    elif points and labels:
        parsed_points, parsed_labels = _parse_points_labels(points, labels)
        try:
            mask = segment_from_points(image, parsed_points, parsed_labels)
        except Exception as exc:
            raise _processing_error(exc) from exc
    else:
        raise HTTPException(400, "Provide points+labels or a mask file")

    if not mask.getbbox():
        raise HTTPException(400, "No object selected — click, paint, or detect a region first")

    softness = max(0.5, min(float(edge_softness), 2.0))
    removal = resolve_inpaint_mode(normalize_mode(mode), inpaint_mode)
    try:
        result = inpaint_objects(image, mask, mode=removal, edge_softness=softness)
    except Exception as exc:
        raise _processing_error(exc) from exc
    return Response(content=image_to_bytes(result, "JPEG"), media_type="image/jpeg")


@app.get("/{full_path:path}")
async def serve_app(full_path: str):
    """Serve the built React GUI (SPA fallback)."""
    if full_path.startswith("api/"):
        raise HTTPException(404)
    if not DIST.exists():
        raise HTTPException(
            503,
            "GUI not built. Run: cd frontend && npm install && npm run build",
        )
    target = DIST / full_path
    if target.is_file():
        return FileResponse(target)
    return FileResponse(DIST / "index.html")
