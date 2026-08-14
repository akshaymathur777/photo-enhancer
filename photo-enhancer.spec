# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Photo Enhancer Windows desktop app (onedir)."""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules

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
    "services.scene_context",
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
    "diffusers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "transformers",
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
    "diffusers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "transformers",
):
    try:
        hiddenimports.extend(collect_submodules(pkg))
    except Exception:
        pass

binaries = []
# Full collect_all for packages that need data files; torch uses dynamic libs only
# (collect_all(torch) pulls ~3.9GB including CUDA + headers — we filter below).
for pkg in ("onnxruntime", "cv2", "rembg", "certifi", "diffusers", "accelerate", "huggingface_hub"):
    try:
        _datas, _binaries, _hidden = collect_all(pkg)
        datas.extend(_datas)
        binaries.extend(_binaries)
        hiddenimports.extend(_hidden)
    except Exception:
        pass

try:
    binaries.extend(collect_dynamic_libs("torch"))
    binaries.extend(collect_dynamic_libs("torchvision"))
except Exception:
    pass

hiddenimports = list(dict.fromkeys(hiddenimports))

# Optional / unused CUDA + packaging junk. Safe for SDXL/PowerPaint inference.
_SKIP_SUBSTR = (
    ".lib",
    ".pdb",
    "cudnn_adv",
    "cusolvermg",
    "nvrtc64_120_0.alt",
    "cufftw",
    "cupti",
    "_polars_runtime",
    "torch/include",
    "torch\\include",
    "/include/",
    "\\include\\",
)


def _keep_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return not any(s.lower().replace("\\", "/") in p for s in _SKIP_SUBSTR)


binaries = [b for b in binaries if _keep_path(str(b[0])) and _keep_path(str(b[1]))]
datas = [d for d in datas if _keep_path(str(d[0])) and _keep_path(str(d[1]))]

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
        "notebook",
        "jupyter",
        "IPython",
        "pytest",
        "setuptools",
        "polars",
        "matplotlib",
        "numba",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Second pass: Analysis may re-add torch junk via hooks.
a.binaries = [b for b in a.binaries if _keep_path(str(b[0])) and _keep_path(str(b[1]))]
a.datas = [d for d in a.datas if _keep_path(str(d[0])) and _keep_path(str(d[1]))]

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
