"""Option B — local classical Photo Enhancer Executa (CPU OpenCV/PIL; optional rembg)."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

import classical_ops as ops

NAME = "tool-dev-photo-enhancer-local"
VERSION = "0.1.0"

MANIFEST = {
    "name": NAME,
    "version": VERSION,
    "protocolVersion": "2.0",
    "tools": [
        {
            "name": "ping",
            "description": "Smoke-test method for the local classical Executa.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "local_enhance",
            "description": "Classical CPU enhance (color/contrast/sharpen). Pass image_png_b64.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_png_b64": {"type": "string"},
                    "strength": {"type": "number", "default": 1.0},
                },
                "required": ["image_png_b64"],
                "additionalProperties": False,
            },
        },
        {
            "name": "local_inpaint",
            "description": "OpenCV TELEA/NS inpaint. Pass RGB image + mask as PNG base64.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_png_b64": {"type": "string"},
                    "mask_png_b64": {"type": "string"},
                    "method": {"type": "string", "enum": ["telea", "ns"], "default": "telea"},
                },
                "required": ["image_png_b64", "mask_png_b64"],
                "additionalProperties": False,
            },
        },
        {
            "name": "local_remove_bg",
            "description": "Background removal via rembg (optional dependency). Pass image_png_b64.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_png_b64": {"type": "string"},
                    "model_key": {
                        "type": "string",
                        "enum": ["lite", "general", "portrait"],
                        "default": "lite",
                    },
                },
                "required": ["image_png_b64"],
                "additionalProperties": False,
            },
        },
        {
            "name": "capabilities",
            "description": "Report which local classical backends are importable on this host.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ],
}


def _capabilities() -> dict[str, Any]:
    caps = {
        "enhance": True,
        "inpaint_opencv": False,
        "remove_bg_rembg": False,
        "cuda": False,
        "notes": "Option B classical path. Generative CUDA (SDXL/PowerPaint) remains desktop-only for now.",
    }
    try:
        import cv2  # noqa: F401

        caps["inpaint_opencv"] = True
    except Exception as e:  # noqa: BLE001
        caps["opencv_error"] = str(e)
    try:
        import rembg  # noqa: F401

        caps["remove_bg_rembg"] = True
    except Exception as e:  # noqa: BLE001
        caps["rembg_error"] = str(e)
    try:
        import torch

        caps["cuda"] = bool(torch.cuda.is_available())
        caps["torch"] = torch.__version__
    except Exception:
        pass
    return caps


def invoke(method: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        if method == "ping":
            return {"success": True, "data": {"pong": True, "version": VERSION, "track": "B"}}

        if method == "capabilities":
            return {"success": True, "data": _capabilities()}

        if method == "local_enhance":
            b64 = (args.get("image_png_b64") or "").strip()
            if not b64:
                return {"success": False, "error": "image_png_b64 required"}
            strength = float(args.get("strength", 1.0))
            img = ops.decode_image_b64(b64)
            out = ops.auto_enhance(img, strength=strength)
            return {
                "success": True,
                "data": ops.result_payload(out, {"engine": "classical_enhance", "strength": strength}),
            }

        if method == "local_inpaint":
            ib = (args.get("image_png_b64") or "").strip()
            mb = (args.get("mask_png_b64") or "").strip()
            if not ib or not mb:
                return {"success": False, "error": "image_png_b64 and mask_png_b64 required"}
            method_name = (args.get("method") or "telea").lower()
            img = ops.decode_image_b64(ib)
            mask = ops.decode_image_b64(mb)
            out = ops.inpaint_opencv(img, mask, method=method_name)
            return {
                "success": True,
                "data": ops.result_payload(out, {"engine": "opencv_inpaint", "method": method_name}),
            }

        if method == "local_remove_bg":
            b64 = (args.get("image_png_b64") or "").strip()
            if not b64:
                return {"success": False, "error": "image_png_b64 required"}
            model_key = args.get("model_key") or "lite"
            img = ops.decode_image_b64(b64)
            out = ops.remove_bg_rembg(img, model_key=str(model_key))
            return {
                "success": True,
                "data": ops.result_payload(out, {"engine": "rembg", "model_key": model_key}),
            }

        return {"success": False, "error": f"unknown method: {method}"}
    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc(limit=6),
        }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            method = req.get("method")
            if method == "describe":
                result: Any = MANIFEST
            elif method == "initialize":
                result = {
                    "protocolVersion": "2.0",
                    "serverInfo": {"name": NAME, "version": VERSION},
                }
            elif method == "health":
                result = {"status": "ready", "capabilities": _capabilities()}
            elif method == "invoke":
                params = req.get("params") or {}
                tool = params.get("tool") or params.get("name")
                args = params.get("arguments") or params.get("args") or {}
                result = invoke(str(tool), args)
            else:
                raise ValueError(f"unknown rpc: {method}")
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}) + "\n")
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req.get("id"),
                        "error": {"code": -32601, "message": str(e)},
                    }
                )
                + "\n"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
