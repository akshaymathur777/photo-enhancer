"""Unit + Executa smoke tests for Option B classical ops."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "executas" / "photo-enhancer-local"
sys.path.insert(0, str(LOCAL))

import classical_ops as ops  # noqa: E402


def _sample_png_b64(w: int = 64, h: int = 48) -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (40, 90, 140))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 8, 40, 38), fill=(220, 70, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_png_b64(w: int = 64, h: int = 48) -> str:
    from PIL import Image, ImageDraw

    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    d.ellipse((10, 8, 40, 38), fill=255)
    buf = io.BytesIO()
    m.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ClassicalOpsTests(unittest.TestCase):
    def test_enhance_changes_pixels(self):
        img = ops.decode_image_b64(_sample_png_b64())
        out = ops.auto_enhance(img, strength=1.5)
        import numpy as np

        self.assertEqual(out.size, img.size)
        self.assertFalse(np.array_equal(np.asarray(out), np.asarray(img)))

    def test_inpaint_fills_mask(self):
        img = ops.decode_image_b64(_sample_png_b64())
        mask = ops.decode_image_b64(_mask_png_b64())
        out = ops.inpaint_opencv(img, mask, method="telea")
        self.assertEqual(out.size, img.size)
        # Center of ellipse should no longer be pure red-ish after inpaint
        px = out.getpixel((25, 23))
        self.assertNotEqual(px, (220, 70, 60))

    def test_encode_roundtrip(self):
        img = ops.decode_image_b64(_sample_png_b64())
        b64 = ops.encode_png_b64(img)
        back = ops.decode_image_b64(b64)
        self.assertEqual(back.size, img.size)

    def test_remove_bg_lite_when_rembg_present(self):
        try:
            import rembg  # noqa: F401
        except ImportError:
            self.skipTest("rembg not installed")
        img = ops.decode_image_b64(_sample_png_b64())
        out = ops.remove_bg_rembg(img, model_key="lite")
        self.assertEqual(out.size, img.size)
        self.assertIn(out.mode, ("RGBA", "LA", "PA"))


class LocalExecutaSmoke(unittest.TestCase):
    def _invoke(self, tool: str, args: dict) -> dict:
        """Invoke the local plugin directly.

        Prefer the project .venv so optional rembg is visible. Avoid
        ``anna-app executa dev`` while a live harness holds
        ``Scripts/tool-dev-photo-enhancer-local.exe`` (uv rebuild → Access denied).
        """
        venv_py = LOCAL / ".venv" / "Scripts" / "python.exe"
        py = str(venv_py if venv_py.exists() else Path(sys.executable))
        plugin = LOCAL / "photo_enhancer_local_plugin.py"
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "invoke",
            "params": {"tool": tool, "arguments": args},
        }
        p = subprocess.run(
            [py, str(plugin)],
            input=json.dumps(req) + "\n",
            capture_output=True,
            text=True,
            cwd=str(LOCAL),
        )
        lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip().startswith("{")]
        self.assertTrue(lines, msg=p.stderr or p.stdout or f"exit={p.returncode}")
        payload = json.loads(lines[-1])
        if "result" in payload:
            return payload["result"]
        if "error" in payload:
            return {"success": False, "error": payload["error"]}
        return payload

    def test_ping_and_capabilities(self):
        ping = self._invoke("ping", {})
        self.assertTrue(ping.get("success"))
        self.assertEqual(ping["data"]["track"], "B")
        caps = self._invoke("capabilities", {})
        self.assertTrue(caps.get("success"))
        self.assertTrue(caps["data"]["enhance"])
        self.assertTrue(caps["data"]["inpaint_opencv"])

    def test_local_enhance_via_executa(self):
        out = self._invoke("local_enhance", {"image_png_b64": _sample_png_b64(), "strength": 1.2})
        self.assertTrue(out.get("success"), out)
        self.assertIn("image_png_b64", out["data"])
        self.assertEqual(out["data"]["engine"], "classical_enhance")

    def test_local_inpaint_via_executa(self):
        out = self._invoke(
            "local_inpaint",
            {
                "image_png_b64": _sample_png_b64(),
                "mask_png_b64": _mask_png_b64(),
                "method": "telea",
            },
        )
        self.assertTrue(out.get("success"), out)
        self.assertEqual(out["data"]["engine"], "opencv_inpaint")

    def test_local_enhance_requires_image(self):
        out = self._invoke("local_enhance", {"image_png_b64": ""})
        self.assertFalse(out.get("success"))


if __name__ == "__main__":
    unittest.main()
