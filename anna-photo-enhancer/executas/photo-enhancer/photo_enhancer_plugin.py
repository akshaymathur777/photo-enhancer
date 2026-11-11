"""Photo Enhancer Executa — chat-facing tools + reverse-RPC image/edit (Option A)."""

from __future__ import annotations

import json
import sys
import threading
import uuid
from typing import Any

NAME = "tool-dev-photo-enhancer"
VERSION = "0.1.0"

PROMPTS = {
    "enhance": (
        "Enhance this photograph: improve clarity, natural detail, color balance, "
        "and mild sharpening. Keep the subject, composition, and identity unchanged. "
        "Do not add or remove objects. Photorealistic output."
    ),
    "remove_bg": (
        "Remove the background completely. Keep the main subject with clean edges. "
        "Output a transparent PNG when possible; otherwise a clean cutout on a plain "
        "white background. Do not alter the subject."
    ),
}

MANIFEST = {
    "name": NAME,
    "version": VERSION,
    "protocolVersion": "2.0",
    "client_capabilities": {
        "image": True,
        "image.edit": True,
    },
    "tools": [
        {
            "name": "list_modes",
            "description": "List Photo Enhancer modes available in the App UI and via tools.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "enhance_photo",
            "description": "Enhance an existing photo URL via Anna host image.edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "HTTPS URL of the source image (e.g. R2 download_url).",
                    }
                },
                "required": ["image_url"],
                "additionalProperties": False,
            },
        },
        {
            "name": "remove_background",
            "description": "Remove the background from a photo URL via Anna host image.edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "HTTPS URL of the source image.",
                    }
                },
                "required": ["image_url"],
                "additionalProperties": False,
            },
        },
        {
            "name": "remove_object",
            "description": "Remove a described object/region from a photo URL via Anna host image.edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "HTTPS URL of the source image.",
                    },
                    "what_to_remove": {
                        "type": "string",
                        "description": "Short description of what to remove and inpaint.",
                    },
                },
                "required": ["image_url", "what_to_remove"],
                "additionalProperties": False,
            },
        },
        {
            "name": "ping",
            "description": "Smoke-test method.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ],
}

_pending: dict[str, dict[str, Any]] = {}
_pending_cv = threading.Condition()
_stdout_lock = threading.Lock()


def _write(msg: dict[str, Any]) -> None:
    with _stdout_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def _reverse_rpc(method: str, params: dict[str, Any], timeout_s: float = 120.0) -> Any:
    """Emit a reverse JSON-RPC request and wait for the matching response on stdin."""
    req_id = f"rev-{uuid.uuid4().hex[:12]}"
    _write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
    with _pending_cv:
        _pending[req_id] = {"event": threading.Event(), "response": None}
        ev = _pending[req_id]["event"]
    if not ev.wait(timeout_s):
        with _pending_cv:
            _pending.pop(req_id, None)
        raise TimeoutError(f"reverse-RPC timeout: {method}")
    with _pending_cv:
        resp = _pending.pop(req_id, {}).get("response") or {}
    if "error" in resp:
        err = resp["error"]
        raise RuntimeError(f"{err.get('code')}: {err.get('message')}")
    return resp.get("result")


# Forum #400 interim until GPT Image 2 edit route recovers (~2026-10-30).
EDIT_MODEL = "fal-ai/nano-banana-2"


def _edit_image(image_url: str, prompt: str, invoke_id: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "image_url": image_url,
        "prompt": prompt,
        "n": 1,
        "mask_url": None,
        "model": EDIT_MODEL,
    }
    if invoke_id:
        params["metadata"] = {"executa_invoke_id": invoke_id}
    result = _reverse_rpc("image/edit", params, timeout_s=150.0)
    images = (result or {}).get("images") or []
    if not images:
        return {"success": False, "error": "no images returned from image/edit"}
    return {
        "success": True,
        "data": {
            "images": images,
            "model": (result or {}).get("model"),
            "prompt": prompt,
        },
    }


def invoke(method: str, args: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    invoke_id = context.get("invoke_id") or context.get("executa_invoke_id")

    if method == "ping":
        return {"success": True, "data": {"pong": True, "version": VERSION}}

    if method == "list_modes":
        return {
            "success": True,
            "data": {
                "modes": [
                    {"id": "enhance", "label": "Enhance", "tool": "enhance_photo"},
                    {"id": "remove_bg", "label": "Remove background", "tool": "remove_background"},
                    {"id": "remove_object", "label": "Remove object", "tool": "remove_object"},
                ],
                "ui_hint": "Prefer open_app_view('main') when the user needs to upload a local file.",
            },
        }

    if method == "enhance_photo":
        url = (args.get("image_url") or "").strip()
        if not url:
            return {"success": False, "error": "image_url required"}
        return _edit_image(url, PROMPTS["enhance"], invoke_id)

    if method == "remove_background":
        url = (args.get("image_url") or "").strip()
        if not url:
            return {"success": False, "error": "image_url required"}
        return _edit_image(url, PROMPTS["remove_bg"], invoke_id)

    if method == "remove_object":
        url = (args.get("image_url") or "").strip()
        what = (args.get("what_to_remove") or "").strip()
        if not url:
            return {"success": False, "error": "image_url required"}
        if not what:
            return {"success": False, "error": "what_to_remove required"}
        prompt = (
            f"Remove the following from this photograph and inpaint the area so it looks "
            f"natural and continuous with the surroundings: {what}. Preserve everything else. "
            f"Photorealistic, no artifacts."
        )
        return _edit_image(url, prompt, invoke_id)

    return {"success": False, "error": f"unknown method: {method}"}


def _handle_request(req: dict[str, Any]) -> None:
    # Responses to our reverse-RPC calls arrive as JSON-RPC responses (have id, no method).
    if "method" not in req and "id" in req:
        rid = req.get("id")
        with _pending_cv:
            slot = _pending.get(str(rid)) if rid is not None else None
            if slot is not None:
                slot["response"] = req
                slot["event"].set()
        return

    try:
        method = req.get("method")
        if method == "describe":
            result: Any = MANIFEST
        elif method == "initialize":
            # v2 handshake — echo negotiated capabilities.
            result = {
                "protocolVersion": "2.0",
                "serverInfo": {"name": NAME, "version": VERSION},
                "client_capabilities": MANIFEST["client_capabilities"],
            }
        elif method == "health":
            result = {"status": "ready"}
        elif method == "invoke":
            params = req.get("params") or {}
            tool = params.get("tool") or params.get("name")
            args = params.get("arguments") or params.get("args") or {}
            context = params.get("context") or {}
            result = invoke(str(tool), args, context)
        else:
            raise ValueError(f"unknown rpc: {method}")
        _write({"jsonrpc": "2.0", "id": req.get("id"), "result": result})
    except Exception as e:  # noqa: BLE001
        _write(
            {
                "jsonrpc": "2.0",
                "id": req.get("id"),
                "error": {"code": -32601, "message": str(e)},
            }
        )


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Handle reverse-RPC replies inline; run invoke work that may block on reverse-RPC
        # in a daemon thread so stdin keeps draining.
        if "method" not in req and "id" in req:
            _handle_request(req)
            continue
        if req.get("method") == "invoke":
            threading.Thread(target=_handle_request, args=(req,), daemon=True).start()
        else:
            _handle_request(req)


if __name__ == "__main__":
    main()
