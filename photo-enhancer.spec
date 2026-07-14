# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Photo Enhancer Windows desktop app (onedir)."""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
ROOT = Path(SPECPATH)

backend = ROOT / "backend"
frontend_dist = ROOT / "frontend" / "dist"
assets = ROOT / "assets"

if not frontend_dist.joinpath("index.html").is_file():
    raise SystemExit(
        "frontend/dist not found. Run: cd frontend && npm install && npm run build"
    )

datas = [
    (str(frontend_dist), "frontend/dist"),
    (str(backend / "services"), "backend/services"),
    (str(backend / "paths.py"), "backend"),
    (str(backend / "main.py"), "backend"),
]
if assets.joinpath("icon.ico").is_file():
    datas.append((str(assets / "icon.ico"), "assets"))

hiddenimports = [
    "paths",
    "main",
    "services.ai_modes",
    "services.ai_styles",
    "services.chat_agent",
    "services.image_io",
    "services.image_processing",
    "services.inpaint_backends",
    "services.model_download",
    "services.object_removal",
    "services.upscale_limits",
    "services.settings",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.__main__",
    "webview",
    "clr_loader",
    "pythonnet",
    "PIL._tkinter_finder",
    "multipart",
    "dotenv",
    "certifi",
    "pooch",
]

for pkg in (
    "fastapi",
    "starlette",
    "uvicorn",
    "rembg",
    "onnxruntime",
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "spandrel",
    "httpx",
    "numpy",
):
    try:
        hiddenimports.extend(collect_submodules(pkg))
    except Exception:
        pass

binaries = []
for pkg in ("onnxruntime", "torch", "cv2", "rembg"):
    try:
        _datas, _binaries, _hidden = collect_all(pkg)
        datas.extend(_datas)
        binaries.extend(_binaries)
        hiddenimports.extend(_hidden)
    except Exception:
        pass

hiddenimports = list(dict.fromkeys(hiddenimports))

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(backend), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "notebook",
        "jupyter",
        "IPython",
        "pytest",
        "setuptools",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PhotoEnhancer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(assets / "icon.ico") if assets.joinpath("icon.ico").is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PhotoEnhancer",
)
