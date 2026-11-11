"""Thorough offline Executa suite for Option A (mock-image) + error paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = str(ROOT / "executas" / "photo-enhancer")
MOCK = str(ROOT / "fixtures" / "image-mocks.jsonl")


def run(args: list[str]) -> tuple[int, dict | None, str]:
    cmd = ["anna-app", "executa", "dev", "--dir", DIR, "--json", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, shell=True, cwd=ROOT)
    blob = (p.stdout or "") + (p.stderr or "")
    lines = [ln for ln in blob.splitlines() if ln.strip().startswith("{")]
    parsed = None
    if lines:
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError:
            parsed = None
    return p.returncode, parsed, blob


def expect(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def main() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        try:
            expect(name, cond, detail)
        except AssertionError:
            failures += 1

    plugin_src = (ROOT / "executas" / "photo-enhancer" / "photo_enhancer_plugin.py").read_text(
        encoding="utf-8"
    )
    check(
        "edit pins fal-ai/nano-banana-2",
        'EDIT_MODEL = "fal-ai/nano-banana-2"' in plugin_src
        and '"model": EDIT_MODEL' in plugin_src,
    )

    code, data, _ = run(["--describe"])
    check("describe", code == 0 and data is not None and "tools" in (data or {}))
    tools = {t["name"] for t in (data or {}).get("tools", [])}
    check(
        "describe tools",
        tools >= {"ping", "list_modes", "enhance_photo", "remove_background", "remove_object"},
        str(sorted(tools)),
    )
    check("describe v2 image.edit cap", (data or {}).get("client_capabilities", {}).get("image.edit") is True)

    code, data, _ = run(["--health"])
    check("health", code == 0 and (data or {}).get("status") == "ready")

    code, data, _ = run(["--invoke", "ping", "--args", "{}", "--no-image"])
    check("ping", code == 0 and (data or {}).get("success") is True)

    code, data, _ = run(["--invoke", "list_modes", "--args", "{}", "--no-image"])
    modes = ((data or {}).get("data") or {}).get("modes") or []
    check("list_modes count", code == 0 and len(modes) == 3, str(modes))

    cases = [
        (
            "enhance_photo",
            {"image_url": "https://example.com/a.png"},
            "mock-enhance",
            "Enhance this photograph",
        ),
        (
            "remove_background",
            {"image_url": "https://example.com/a.png"},
            "mock-remove-bg",
            "Remove the background",
        ),
        (
            "remove_object",
            {"image_url": "https://example.com/a.png", "what_to_remove": "the red ball"},
            "mock-remove-object",
            "the red ball",
        ),
    ]
    for tool, args, model, needle in cases:
        code, data, raw = run(
            ["--invoke", tool, "--args", json.dumps(args), "--mock-image", MOCK]
        )
        ok = (
            code == 0
            and (data or {}).get("success") is True
            and ((data or {}).get("data") or {}).get("model") == model
            and needle in (((data or {}).get("data") or {}).get("prompt") or "")
        )
        check(f"invoke {tool} mock", ok, f"model={((data or {}).get('data') or {}).get('model')} raw_tail={raw[-200:]}")

    code, data, _ = run(
        ["--invoke", "enhance_photo", "--args", json.dumps({"image_url": ""}), "--mock-image", MOCK]
    )
    check("enhance empty url", code == 0 and (data or {}).get("success") is False)

    code, data, _ = run(
        [
            "--invoke",
            "remove_object",
            "--args",
            json.dumps({"image_url": "https://x", "what_to_remove": ""}),
            "--mock-image",
            MOCK,
        ]
    )
    check("remove_object empty what", code == 0 and (data or {}).get("success") is False)

    code, data, blob = run(
        [
            "--invoke",
            "enhance_photo",
            "--args",
            json.dumps({"image_url": "https://example.com/a.png"}),
            "--no-image",
        ]
    )
    check("enhance denied without image", code != 0 or (data or {}).get("success") is False, blob[-180:])

    code, data, _ = run(["--invoke", "nope", "--args", "{}", "--no-image"])
    check("unknown tool", code == 0 and (data or {}).get("success") is False)

    # Option B local classical present?
    local = ROOT / "executas" / "photo-enhancer-local" / "photo_enhancer_local_plugin.py"
    check("option B plugin file exists", local.is_file(), str(local))

    print()
    print(f"executa suite failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
