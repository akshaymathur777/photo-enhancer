from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from services.ai_modes import (
    clamp_scale_for_mode,
    normalize_mode,
    resolve_bg_model,
)
from services.ai_styles import apply_style, normalize_style
from services.image_processing import (
    adjust_image,
    auto_enhance,
    denoise,
    image_to_bytes,
    remove_background,
    replace_background,
    sharpen,
    upscale,
)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "apply_photo_edit",
        "description": "Apply a photo editing tool to the user's image.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": [
                        "remove-bg",
                        "replace-bg",
                        "auto-enhance",
                        "adjust",
                        "denoise",
                        "sharpen",
                        "upscale",
                        "remove-objects",
                        "apply-style",
                        "none",
                        "clarify",
                    ],
                },
                "reply": {"type": "string", "description": "Short friendly reply to the user."},
                "mode": {
                    "type": "string",
                    "enum": ["best_quality", "balanced", "fast"],
                    "description": "Processing quality/speed preset.",
                },
                "style": {
                    "type": "string",
                    "description": "Creative style preset id or name (cinematic, vintage, vivid, portrait, bw_dramatic, golden_hour, cool_moody).",
                },
                "model": {
                    "type": "string",
                    "enum": ["massive", "general", "portrait", "matting", "bria", "lite"],
                },
                "bg_type": {"type": "string", "enum": ["transparent", "color", "blur"]},
                "bg_color": {"type": "string", "description": "Hex color like #ffffff"},
                "brightness": {"type": "number"},
                "contrast": {"type": "number"},
                "saturation": {"type": "number"},
                "sharpness": {"type": "number"},
                "strength": {"type": "number"},
                "sharpen_amount": {"type": "number"},
                "denoise_strength": {"type": "integer"},
                "scale": {"type": "integer", "enum": [2, 4]},
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 short follow-up suggestions for the user.",
                },
            },
            "required": ["tool", "reply"],
        },
    },
}

DEFAULT_SUGGESTIONS = [
    "Make it brighter",
    "Remove the background",
    "Make it cinematic",
    "Auto enhance",
]

_VALID_TOOLS = {
    "remove-bg",
    "replace-bg",
    "auto-enhance",
    "adjust",
    "denoise",
    "sharpen",
    "upscale",
    "remove-objects",
    "apply-style",
    "none",
    "clarify",
}


@dataclass
class ChatAction:
    tool: str
    reply: str
    options: dict[str, Any] = field(default_factory=dict)
    needs_clicks: bool = False
    suggestions: list[str] = field(default_factory=list)
    clarify: bool = False


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Ollama (local, on-device, fully offline LLM backend)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Cache Ollama reachability so we don't probe the local server on every request.
_OLLAMA_CACHE_TTL = 15.0
_ollama_cache: dict[str, Any] = {"checked_at": 0.0, "reachable": False, "models": []}


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip().rstrip("/") or DEFAULT_OLLAMA_HOST


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def ollama_available(force: bool = False) -> bool:
    """Best-effort, short-timeout probe of a local Ollama server.

    Never raises — if Ollama isn't installed/running we simply report False and
    the caller falls back to the rule-based parser. Results are cached briefly.
    """
    now = time.monotonic()
    if not force and (now - _ollama_cache["checked_at"]) < _OLLAMA_CACHE_TTL:
        return bool(_ollama_cache["reachable"])

    reachable = False
    models: list[str] = []
    try:
        import httpx

        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{ollama_host()}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            reachable = True
    except Exception:
        reachable = False

    _ollama_cache.update({"checked_at": now, "reachable": reachable, "models": models})
    return reachable


def _ollama_models() -> list[str]:
    return list(_ollama_cache.get("models") or [])


def ollama_model_present() -> bool:
    """True if the configured model is actually pulled on the local server."""
    want = ollama_model().lower()
    models = [m.lower() for m in _ollama_models()]
    if not models:
        return False
    want_base = want.split(":", 1)[0]
    for name in models:
        if name == want or name == f"{want}:latest":
            return True
        # Allow matching by base name (e.g. "llama3.2" matches "llama3.2:3b").
        if ":" not in want and name.split(":", 1)[0] == want_base:
            return True
    return False


def ollama_ready() -> bool:
    """Reachable AND the configured model is available to serve requests."""
    return ollama_available() and ollama_model_present()


def active_backend() -> str:
    """Resolve which chat backend is active. Privacy-first: never the cloud by default.

    Configurable via CHAT_BACKEND = auto (default) | rules | ollama | openai.
    In "auto" mode the priority is: local Ollama (offline) → OpenAI (only if a
    key is set) → rule-based. Rule-based is always the ultimate fallback.
    """
    pref = os.environ.get("CHAT_BACKEND", "auto").strip().lower()
    if pref == "rules":
        return "rules"
    if pref == "ollama":
        return "ollama" if ollama_ready() else "rules"
    if pref == "openai":
        return "openai" if has_openai_key() else "rules"
    # auto
    if ollama_ready():
        return "ollama"
    if has_openai_key():
        return "openai"
    return "rules"


def chat_backend_status() -> dict[str, Any]:
    """Structured status for /api/chat/status."""
    backend = active_backend()
    reachable = ollama_available()
    return {
        # Backwards-compatible flag: any LLM-backed engine is "ai".
        "ai_enabled": backend in ("ollama", "openai"),
        "backend": backend,
        "ollama_reachable": reachable,
        "ollama_model_present": reachable and ollama_model_present(),
        "ollama_model": ollama_model() if reachable else None,
        "openai_available": has_openai_key(),
    }


def parse_intent(message: str, history: list[dict] | None = None) -> ChatAction:
    hist = history or []
    backend = active_backend()

    if backend == "ollama":
        try:
            action = _parse_with_ollama(message, hist)
            if action is not None:
                if not action.suggestions:
                    action.suggestions = _contextual_suggestions(action, hist)
                return action
        except Exception:
            pass
    elif backend == "openai":
        try:
            action = _parse_with_openai(message, hist)
            if not action.suggestions:
                action.suggestions = _contextual_suggestions(action, hist)
            return action
        except Exception:
            pass

    return _parse_with_rules(message, hist)


_OLLAMA_SYSTEM_PROMPT = (
    "You are a photo editing assistant in an interactive chat for an offline, "
    "privacy-first desktop app. Map the user's natural-language request to ONE of "
    "the app's tools and return ONLY a single JSON object (no prose, no markdown).\n\n"
    "JSON fields:\n"
    "- tool (required): one of remove-bg, replace-bg, auto-enhance, adjust, denoise, "
    "sharpen, upscale, remove-objects, apply-style, clarify, none\n"
    "- reply (required): short friendly message to the user\n"
    "- mode: best_quality | balanced | fast\n"
    "- style: cinematic | vintage | vivid | portrait | bw_dramatic | golden_hour | cool_moody "
    "(only for apply-style)\n"
    "- bg_type: transparent | color | blur, bg_color: hex like #ffffff (only for replace-bg)\n"
    "- brightness, contrast, saturation, sharpness: numbers ~0.5-1.5 (only for adjust)\n"
    "- strength: number (auto-enhance), sharpen_amount: number (sharpen), "
    "denoise_strength: integer (denoise), scale: 2 or 4 (upscale)\n"
    "- suggestions: array of 2-4 short follow-up ideas\n\n"
    "Guidance:\n"
    "- Removing the whole background -> remove-bg. Solid/blurred background -> replace-bg.\n"
    "- Removing a specific person/object (user clicks to select) -> remove-objects.\n"
    "- Creative looks -> apply-style with the matching style id.\n"
    "- Vague requests like 'fix it'/'make it better' -> tool=clarify with a short question.\n"
    "- Use conversation history for follow-ups like 'more', 'brighter', 'again'.\n"
    "Return strictly valid JSON."
)


def _parse_with_ollama(message: str, history: list[dict]) -> ChatAction | None:
    import httpx

    messages = [{"role": "system", "content": _OLLAMA_SYSTEM_PROMPT}]
    for item in history[-10:]:
        role = item.get("role", "user")
        if role in ("user", "assistant"):
            content = item.get("content", "")
            meta = item.get("meta")
            if meta and role == "assistant":
                content = f"{content}\n[meta: {json.dumps(meta)}]"
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{ollama_host()}/api/chat",
            json={
                "model": ollama_model(),
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    content = (data.get("message") or {}).get("content", "").strip()
    if not content:
        return None

    args = _extract_json(content)
    if not args or not isinstance(args, dict):
        return None

    if args.get("tool") not in _VALID_TOOLS:
        return None

    if args.get("tool") == "clarify":
        suggestions = args.get("suggestions") or DEFAULT_SUGGESTIONS
        return ChatAction(
            "none",
            args.get("reply", "What would you like me to do?"),
            clarify=True,
            suggestions=suggestions[:4],
        )

    action = _action_from_args(args)
    if args.get("suggestions"):
        action.suggestions = args["suggestions"][:4]
    return action


def _extract_json(content: str) -> dict[str, Any] | None:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: grab the first {...} block in case the model wrapped it in prose.
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _parse_with_openai(message: str, history: list[dict]) -> ChatAction:
    import httpx

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a photo editing assistant in an interactive chat. Interpret the user's "
                "request using conversation history. Call apply_photo_edit with the best tool and "
                "parameters. For removing specific people/objects by description, use remove-objects "
                "(user must click to select). For background removal use remove-bg. For colored/blurred "
                "backgrounds use replace-bg. If the request is vague (e.g. 'fix it', 'make it better'), "
                "use tool=clarify and ask a short clarifying question with suggestions. For follow-ups "
                "like 'more', 'brighter', 'yes', use history to continue the last edit. "
                "For creative looks use apply-style with style=cinematic|vintage|vivid|portrait|"
                "bw_dramatic|golden_hour|cool_moody. Keep replies concise and conversational. "
                "Always include 2-4 short suggestions."
            ),
        },
    ]
    for item in history[-10:]:
        role = item.get("role", "user")
        if role in ("user", "assistant"):
            content = item.get("content", "")
            meta = item.get("meta")
            if meta and role == "assistant":
                content = f"{content}\n[meta: {json.dumps(meta)}]"
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "tools": [TOOL_SCHEMA],
                "tool_choice": {"type": "function", "function": {"name": "apply_photo_edit"}},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]["message"]
    tool_calls = choice.get("tool_calls") or []
    if not tool_calls:
        return ChatAction(
            "none",
            choice.get("content") or "I'm not sure how to help with that.",
            suggestions=DEFAULT_SUGGESTIONS,
        )

    args = json.loads(tool_calls[0]["function"]["arguments"])
    if args.get("tool") == "clarify":
        suggestions = args.get("suggestions") or DEFAULT_SUGGESTIONS
        return ChatAction(
            "none",
            args.get("reply", "What would you like me to do?"),
            clarify=True,
            suggestions=suggestions[:4],
        )
    action = _action_from_args(args)
    if args.get("suggestions"):
        action.suggestions = args["suggestions"][:4]
    return action


def _action_from_args(args: dict[str, Any]) -> ChatAction:
    tool = args.get("tool", "none")
    reply = args.get("reply", "Done.")
    mode = normalize_mode(args.get("mode"))
    options: dict[str, Any] = {
        "model": args.get("model") or resolve_bg_model(mode),
        "mode": mode,
    }

    if tool == "replace-bg":
        options["bg_type"] = args.get("bg_type", "color")
        options["bg_color"] = args.get("bg_color", "#ffffff")
        options["blur_radius"] = 20
    elif tool == "adjust":
        options["brightness"] = float(args.get("brightness", 1.0))
        options["contrast"] = float(args.get("contrast", 1.0))
        options["saturation"] = float(args.get("saturation", 1.0))
        options["sharpness"] = float(args.get("sharpness", 1.0))
    elif tool == "auto-enhance":
        options["strength"] = float(args.get("strength", 1.0))
    elif tool == "sharpen":
        options["sharpen_amount"] = float(args.get("sharpen_amount", 1.5))
    elif tool == "denoise":
        options["denoise_strength"] = int(args.get("denoise_strength", 10))
    elif tool == "upscale":
        options["scale"] = int(args.get("scale", 2))
    elif tool == "apply-style":
        options["style"] = normalize_style(args.get("style", "cinematic"))

    needs_clicks = tool == "remove-objects"
    suggestions = (args.get("suggestions") or [])[:4]
    return ChatAction(
        tool=tool,
        reply=reply,
        options=options,
        needs_clicks=needs_clicks,
        suggestions=suggestions,
    )


def _last_assistant_meta(history: list[dict]) -> dict[str, Any] | None:
    for item in reversed(history):
        if item.get("role") == "assistant":
            meta = item.get("meta")
            if isinstance(meta, dict):
                return meta
    return None


def _parse_with_rules(message: str, history: list[dict]) -> ChatAction:
    text = message.lower().strip()
    last_meta = _last_assistant_meta(history)

    # Undo / revert — handled mostly on frontend; guide user
    if _matches(text, r"^(undo|revert|go back|undo that|roll back)(\s+it)?\.?$"):
        return ChatAction(
            "none",
            "Use the Revert button in the menu bar to restore your original image.",
            suggestions=["Make it brighter", "Remove the background", "Try again"],
        )

    # Affirmative follow-up to a clarifying question
    if last_meta and last_meta.get("clarify") and _matches(
        text, r"^(yes|yeah|yep|sure|ok|okay|do it|go ahead|please|sounds good)\.?$"
    ):
        pending = last_meta.get("pending_tool")
        if pending:
            return _replay_action(pending, last_meta.get("pending_options") or {})

    # Relative follow-ups referencing last edit
    follow_up = _parse_follow_up(text, last_meta)
    if follow_up:
        return follow_up

    # Ambiguous / vague requests
    if _is_ambiguous(text):
        return ChatAction(
            "none",
            "I'd love to help! What kind of edit are you looking for?",
            clarify=True,
            suggestions=["Make it brighter", "Remove the background", "Auto enhance", "Sharpen"],
        )

    if _matches(text, r"\b(cinematic|film look|movie look|teal and orange)\b"):
        return ChatAction(
            "apply-style",
            "Applied a cinematic look with teal shadows and warm highlights.",
            options={"style": "cinematic", "mode": "balanced"},
            suggestions=["Make it vintage", "More vivid", "Try again"],
        )

    if _matches(text, r"\b(vintage|retro|old photo|film grain)\b"):
        return ChatAction(
            "apply-style",
            "Applied a vintage look with faded tones and grain.",
            options={"style": "vintage", "mode": "balanced"},
            suggestions=["Make it cinematic", "Golden hour", "Try again"],
        )

    if _matches(text, r"\b(vivid|vibrant|pop the colors|more colorful)\b"):
        return ChatAction(
            "apply-style",
            "Boosted saturation and clarity for a vivid look.",
            options={"style": "vivid", "mode": "balanced"},
            suggestions=["Make it cinematic", "Sharpen", "Try again"],
        )

    if _matches(text, r"\b(portrait look|soft skin|skin soft|portrait style)\b"):
        return ChatAction(
            "apply-style",
            "Applied a portrait style with skin softening.",
            options={"style": "portrait", "mode": "balanced"},
            suggestions=["Make it brighter", "Remove the background", "Try again"],
        )

    if _matches(text, r"\b(black and white|b&w|monochrome|noir)\b"):
        return ChatAction(
            "apply-style",
            "Converted to dramatic black and white.",
            options={"style": "bw_dramatic", "mode": "balanced"},
            suggestions=["More contrast", "Make it cinematic", "Try again"],
        )

    if _matches(text, r"\b(golden hour|sunset glow|warm glow)\b"):
        return ChatAction(
            "apply-style",
            "Applied a golden-hour warm glow.",
            options={"style": "golden_hour", "mode": "balanced"},
            suggestions=["Make it cinematic", "Cool moody", "Try again"],
        )

    if _matches(text, r"\b(cool moody|moody blue|blue tones|cool look)\b"):
        return ChatAction(
            "apply-style",
            "Applied a cool, moody blue-toned look.",
            options={"style": "cool_moody", "mode": "balanced"},
            suggestions=["Golden hour", "Cinematic", "Try again"],
        )

    if _matches(text, r"\b(make it|apply|use)\s+(cinematic|vintage|vivid|portrait|moody|golden)\b"):
        style_word = re.search(
            r"\b(cinematic|vintage|vivid|portrait|moody|golden)\b", text
        )
        style_map = {
            "cinematic": "cinematic",
            "vintage": "vintage",
            "vivid": "vivid",
            "portrait": "portrait",
            "moody": "cool_moody",
            "golden": "golden_hour",
        }
        sid = style_map.get(style_word.group(1), "cinematic") if style_word else "cinematic"
        return ChatAction(
            "apply-style",
            f"Applied the {sid.replace('_', ' ')} style.",
            options={"style": sid, "mode": "balanced"},
            suggestions=["Try another style", "Make it brighter", "Upscale 2x"],
        )

    if _matches(text, r"\b(remove|erase|delete).*(person|people|man|woman|object|item|someone)\b"):
        return ChatAction(
            "remove-objects",
            "I'll switch you to Remove Objects — click on what you want to erase, then confirm.",
            needs_clicks=True,
            suggestions=["I clicked on it", "Try a different area"],
        )

    if _matches(text, r"\b(remove|cut out|delete).*(background|bg)\b") or text in (
        "remove background",
        "remove bg",
        "transparent background",
        "make background transparent",
    ):
        transparent = "transparent" in text
        mode = "balanced"
        if transparent:
            action = ChatAction(
                "replace-bg",
                "Made the background transparent.",
                options={"model": resolve_bg_model(mode), "bg_type": "transparent", "mode": mode},
            )
        else:
            action = ChatAction(
                "remove-bg",
                "Background removed.",
                options={"model": resolve_bg_model(mode), "mode": mode},
            )
        action.suggestions = ["Make it brighter", "Upscale 2x", "Try again"]
        return action

    if _matches(text, r"\bchange\b.*\bbackground\b") and not _matches(
        text, r"\b(white|black|blue|red|green|pink|gray|grey|blur)\b"
    ):
        return ChatAction(
            "none",
            "How would you like to change the background?",
            clarify=True,
            suggestions=["Remove the background", "White background", "Blur the background"],
        )

    if _matches(text, r"\b(white|black|blue|red|green|pink|gray|grey)\s+background\b"):
        color = "#ffffff"
        for name, hex_c in _COLOR_MAP.items():
            if name in text:
                color = hex_c
                break
        action = ChatAction(
            "replace-bg",
            "Replaced the background with a solid color.",
            options={
                "model": resolve_bg_model("balanced"),
                "bg_type": "color",
                "bg_color": color,
                "mode": "balanced",
            },
        )
        action.suggestions = ["Make it brighter", "Upscale 2x", "Try again"]
        return action

    if _matches(text, r"\bblur(red)?\s+background\b"):
        action = ChatAction(
            "replace-bg",
            "Applied a blurred background.",
            options={
                "model": resolve_bg_model("balanced"),
                "bg_type": "blur",
                "blur_radius": 25,
                "mode": "balanced",
            },
        )
        action.suggestions = ["Make it brighter", "Remove the background", "Try again"]
        return action

    if _matches(text, r"\b(upscale|enlarge|bigger|scale)\b.*\b4\b") or "4x" in text:
        action = ChatAction("upscale", "Upscaled the image 4× with Real-ESRGAN.", options={"scale": 4})
        action.suggestions = ["Sharpen", "Make it brighter", "Try again"]
        return action

    if _matches(text, r"\b(upscale|enlarge|bigger|scale|resolution)\b") or "2x" in text:
        action = ChatAction("upscale", "Upscaled the image 2× with Real-ESRGAN.", options={"scale": 2})
        action.suggestions = ["Sharpen", "Make it brighter", "Try again"]
        return action

    if _matches(text, r"\b(auto[\s-]?enhance|enhance|improve|fix)\b") and not _matches(
        text, r"\b(background|brightness|contrast)\b"
    ):
        strength = 1.3 if _matches(text, r"\b(a lot|more|strong)\b") else 1.0
        action = ChatAction(
            "auto-enhance",
            "Auto-enhanced your photo.",
            options={"strength": strength},
        )
        action.suggestions = ["Make it brighter", "Sharpen", "Try again"]
        return action

    if _matches(text, r"\b(sharpen|sharper|crisp|crispy)\b"):
        action = ChatAction("sharpen", "Sharpened the image.", options={"sharpen_amount": 1.8})
        action.suggestions = ["Make it brighter", "Upscale 2x", "Try again"]
        return action

    if _matches(text, r"\b(denoise|noise|grainy|grain)\b"):
        action = ChatAction("denoise", "Reduced noise.", options={"denoise_strength": 12})
        action.suggestions = ["Sharpen", "Auto enhance", "Try again"]
        return action

    brightness = 1.0
    contrast = 1.0
    saturation = 1.0
    changed = False

    if _matches(text, r"\b(brighten|brighter|bright|lighter)\b"):
        brightness = 1.25 if _matches(text, r"\b(a lot|more|much)\b") else 1.15
        changed = True
    if _matches(text, r"\b(darken|darker|dark)\b"):
        brightness = 0.8
        changed = True
    if _matches(text, r"\b(contrast|punchy|pop)\b"):
        contrast = 1.25
        changed = True
    if _matches(text, r"\b(saturat|vivid|colorful|colourful)\b"):
        saturation = 1.3
        changed = True
    if _matches(text, r"\b(desaturat|muted|less color)\b"):
        saturation = 0.7
        changed = True

    if changed:
        action = ChatAction(
            "adjust",
            "Adjusted brightness, contrast, and color.",
            options={
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "sharpness": 1.0,
            },
        )
        action.suggestions = ["A bit brighter", "More contrast", "Try again"]
        return action

    return ChatAction(
        "none",
        "I'm not sure what you mean. Try one of the suggestions below, or describe the edit you want.",
        suggestions=DEFAULT_SUGGESTIONS,
    )


def _parse_follow_up(text: str, last_meta: dict[str, Any] | None) -> ChatAction | None:
    if not last_meta or not last_meta.get("tool"):
        # Standalone short follow-ups without context
        if _matches(text, r"^(more|a bit more|again|stronger|another)\.?$"):
            return None
        if text in ("brighter", "lighter", "more brightness"):
            return ChatAction(
                "adjust",
                "Brightened the image a bit more.",
                options={"brightness": 1.15, "contrast": 1.0, "saturation": 1.0, "sharpness": 1.0},
                suggestions=["A bit brighter", "More contrast", "Try again"],
            )
        return None

    last_tool = last_meta.get("tool")
    opts = dict(last_meta.get("options") or {})

    if _matches(text, r"^(more|a bit more|again|stronger|another|keep going)\.?$"):
        return _intensify_last(last_tool, opts)

    if _matches(text, r"\b(brighter|lighter)\b"):
        if last_tool == "adjust":
            opts["brightness"] = min(float(opts.get("brightness", 1.0)) + 0.12, 1.5)
            return ChatAction(
                "adjust",
                "Made it brighter.",
                options=opts,
                suggestions=["A bit brighter", "More contrast", "Try again"],
            )
        return ChatAction(
            "adjust",
            "Brightened the image.",
            options={"brightness": 1.15, "contrast": 1.0, "saturation": 1.0, "sharpness": 1.0},
            suggestions=["A bit brighter", "Auto enhance", "Try again"],
        )

    if _matches(text, r"\b(darker|darken)\b") and last_tool == "adjust":
        opts["brightness"] = max(float(opts.get("brightness", 1.0)) - 0.12, 0.5)
        return ChatAction(
            "adjust",
            "Made it darker.",
            options=opts,
            suggestions=["A bit darker", "Try again"],
        )

    if _matches(text, r"\b(sharper|more sharp)\b"):
        amt = float(opts.get("sharpen_amount", 1.5)) + 0.3
        return ChatAction(
            "sharpen",
            "Sharpened a bit more.",
            options={"sharpen_amount": min(amt, 3.0)},
            suggestions=["Try again", "Make it brighter"],
        )

    if _matches(text, r"\b(try again|redo|one more time)\b"):
        return _replay_action(last_tool, opts)

    return None


def _intensify_last(tool: str, opts: dict[str, Any]) -> ChatAction:
    if tool == "adjust":
        opts = dict(opts)
        opts["brightness"] = min(float(opts.get("brightness", 1.0)) + 0.1, 1.5)
        opts["contrast"] = min(float(opts.get("contrast", 1.0)) + 0.05, 1.5)
        opts["saturation"] = min(float(opts.get("saturation", 1.0)) + 0.05, 2.0)
        return ChatAction(
            "adjust",
            "Pushed the adjustments a bit further.",
            options=opts,
            suggestions=["A bit brighter", "Try again"],
        )
    if tool == "auto-enhance":
        strength = min(float(opts.get("strength", 1.0)) + 0.2, 2.0)
        return ChatAction(
            "auto-enhance",
            "Enhanced a bit more.",
            options={"strength": strength},
            suggestions=["More", "Sharpen", "Try again"],
        )
    if tool == "sharpen":
        amt = min(float(opts.get("sharpen_amount", 1.5)) + 0.3, 3.0)
        return ChatAction(
            "sharpen",
            "Sharpened a bit more.",
            options={"sharpen_amount": amt},
            suggestions=["Try again", "Make it brighter"],
        )
    if tool == "denoise":
        strength = min(int(opts.get("denoise_strength", 10)) + 3, 21)
        return ChatAction(
            "denoise",
            "Denoised a bit more.",
            options={"denoise_strength": strength},
            suggestions=["Sharpen", "Try again"],
        )
    return _replay_action(tool, opts)


def _replay_action(tool: str, opts: dict[str, Any]) -> ChatAction:
    labels = {
        "remove-bg": "Re-applied background removal.",
        "replace-bg": "Re-applied background change.",
        "auto-enhance": "Re-applied auto enhance.",
        "adjust": "Re-applied adjustments.",
        "denoise": "Re-applied denoise.",
        "sharpen": "Re-applied sharpening.",
        "upscale": "Re-applied upscale.",
        "apply-style": "Re-applied style.",
    }
    return ChatAction(
        tool,
        labels.get(tool, "Done."),
        options=dict(opts),
        suggestions=["Try again", "Make it brighter", "Sharpen"],
    )


def _is_ambiguous(text: str) -> bool:
    vague = (
        "fix it",
        "fix this",
        "make it better",
        "improve it",
        "help",
        "edit this",
        "edit it",
        "do something",
        "enhance it",
        "make it look good",
        "what can you do",
    )
    if text in vague:
        return True
    if len(text.split()) <= 3 and _matches(
        text, r"^(fix|help|edit|improve|enhance)(\s+(it|this|photo|image))?\.?$"
    ):
        return True
    return False


def _contextual_suggestions(action: ChatAction, history: list[dict]) -> list[str]:
    if action.suggestions:
        return action.suggestions[:4]
    if action.clarify:
        return action.suggestions or DEFAULT_SUGGESTIONS
    if action.needs_clicks:
        return ["I clicked on it", "Remove the background", "Make it brighter"]
    if action.tool == "remove-bg" or action.tool == "replace-bg":
        return ["Make it brighter", "Upscale 2x", "Sharpen", "Try again"]
    if action.tool == "upscale":
        return ["Sharpen", "Make it brighter", "Try again"]
    if action.tool == "adjust":
        return ["A bit brighter", "More contrast", "Make it cinematic"]
    if action.tool == "apply-style":
        return ["Make it vintage", "Golden hour", "Try again"]
    if action.tool in ("auto-enhance", "sharpen", "denoise"):
        return ["More", "Make it brighter", "Try again"]
    if not history:
        return DEFAULT_SUGGESTIONS
    return DEFAULT_SUGGESTIONS


_COLOR_MAP = {
    "white": "#ffffff",
    "black": "#000000",
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "pink": "#ec4899",
    "gray": "#9ca3af",
    "grey": "#9ca3af",
}


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text))


def execute_action(image: Image.Image, action: ChatAction) -> tuple[Image.Image | None, str, str]:
    """Returns (result_image, reply, mime_subtype)."""
    if action.clarify or action.needs_clicks or action.tool == "remove-objects":
        return None, action.reply, ""

    if action.tool == "none":
        return None, action.reply, ""

    opts = action.options
    tool = action.tool
    mode = normalize_mode(opts.get("mode"))

    if tool == "remove-bg":
        result = remove_background(image, opts.get("model") or resolve_bg_model(mode))
        return result, action.reply, "png"

    if tool == "replace-bg":
        result = replace_background(
            image,
            opts.get("bg_type", "color"),
            opts.get("bg_color", "#ffffff"),
            int(opts.get("blur_radius", 20)),
            opts.get("model") or resolve_bg_model(mode),
        )
        fmt = "png" if opts.get("bg_type") == "transparent" else "jpeg"
        return result, action.reply, fmt

    if tool == "apply-style":
        style_id = normalize_style(opts.get("style", "cinematic"))
        return apply_style(image, style_id), action.reply, "jpeg"

    if tool == "auto-enhance":
        return auto_enhance(image, float(opts.get("strength", 1.0))), action.reply, "jpeg"

    if tool == "adjust":
        return (
            adjust_image(
                image,
                float(opts.get("brightness", 1.0)),
                float(opts.get("contrast", 1.0)),
                float(opts.get("saturation", 1.0)),
                float(opts.get("sharpness", 1.0)),
            ),
            action.reply,
            "jpeg",
        )

    if tool == "denoise":
        return denoise(image, int(opts.get("denoise_strength", 10))), action.reply, "jpeg"

    if tool == "sharpen":
        return sharpen(image, float(opts.get("sharpen_amount", 1.5))), action.reply, "jpeg"

    if tool == "upscale":
        scale = int(opts.get("scale", 2))
        if scale not in (2, 4):
            scale = 2
        w, h = image.size
        try:
            effective_scale, warning = clamp_scale_for_mode(w, h, scale, mode)
        except ValueError as exc:
            return None, str(exc), ""
        reply = action.reply
        if warning:
            reply = f"{action.reply} ({warning})"
        return upscale(image, effective_scale), reply, "png"

    return None, action.reply, ""


def result_to_base64(image: Image.Image, fmt: str) -> str:
    import base64

    data = image_to_bytes(image, fmt.upper())
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def action_meta(action: ChatAction) -> dict[str, Any]:
    """Metadata stored in chat history for follow-up context."""
    meta: dict[str, Any] = {
        "tool": action.tool if action.tool != "none" else None,
        "options": action.options if action.options else None,
        "clarify": action.clarify,
        "needs_clicks": action.needs_clicks,
    }
    if action.clarify and action.suggestions:
        # Map first suggestion to a pending action for "yes" follow-ups
        first = action.suggestions[0].lower()
        pending = _quick_suggestion_to_action(first)
        if pending:
            meta["pending_tool"] = pending.tool
            meta["pending_options"] = pending.options
    return {k: v for k, v in meta.items() if v is not None}


def _quick_suggestion_to_action(text: str) -> ChatAction | None:
    return _parse_with_rules(text, [])
