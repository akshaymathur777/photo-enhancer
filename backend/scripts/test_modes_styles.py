"""Quick smoke test for AI modes and styles API."""
import json
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from services.chat_agent import execute_action, parse_intent

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"


def main() -> None:
    modes = json.loads(urllib.request.urlopen(f"{BASE}/api/modes", timeout=5).read())
    styles = json.loads(urllib.request.urlopen(f"{BASE}/api/styles", timeout=5).read())
    assert len(modes["modes"]) == 3
    assert len(styles["styles"]) == 7
    print("GET /api/modes and /api/styles OK")

    img = Image.new("RGB", (64, 64), (100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()

    boundary = "----bound"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="photo.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="style"\r\n\r\n'
        f"cinematic\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="mode"\r\n\r\n'
        f"balanced\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/api/apply-style",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = urllib.request.urlopen(req, timeout=10)
    out = resp.read()
    assert len(out) > 100
    print(f"POST /api/apply-style OK ({len(out)} bytes)")

    action = parse_intent("make it cinematic", [])
    assert action.tool == "apply-style"
    result, reply, fmt = execute_action(img, action)
    assert result is not None and fmt == "jpeg"
    print(f"Chat style intent OK: {reply}")

    print("All smoke tests passed")


if __name__ == "__main__":
    main()
