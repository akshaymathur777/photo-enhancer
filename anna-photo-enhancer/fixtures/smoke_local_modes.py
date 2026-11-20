"""PC smoke: local_enhance + local_remove_bg on fixtures/sample.png."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "executas" / "photo-enhancer-local"))

import classical_ops as ops  # noqa: E402

SAMPLE = ROOT / "fixtures" / "sample.png"
OUT = ROOT / ".anna" / "smoke-local-modes.json"


def main() -> int:
    raw = SAMPLE.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    img = ops.decode_image_b64(b64)

    enhance = ops.auto_enhance(img, strength=1.0)
    rembg = ops.remove_bg_rembg(img, model_key="lite")

    report = {
        "enhance": {
            "ok": True,
            "size": enhance.size,
            "mode": enhance.mode,
        },
        "remove_bg": {
            "ok": True,
            "size": rembg.size,
            "mode": rembg.mode,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
        raise SystemExit(2)
