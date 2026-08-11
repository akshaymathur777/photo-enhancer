# Rebuild notes (Windows)

After changing the erase stack (PowerPaint / SDXL / object-removal UX), rebuild the desktop app so the packaged `.exe` picks up backend + frontend.

## Dev verify

```bat
cd backend
venv\Scripts\activate
python -c "from services.inpaint_backends import get_backend; print(get_backend('ai').__class__.__name__)"
python scripts\eval_removal_bench.py --only coastal --mode quick --max-side 960
```

Frontend:

```bat
cd frontend
npm install
npm run build
```

## Packaged exe

From repo root:

```bat
build.bat
```

Output: `dist\PhotoEnhancer\PhotoEnhancer.exe`

Models are **not** bundled — they download once to `%APPDATA%\PhotoEnhancer\models\` (PowerPaint ~6 GB, SDXL ~6.5 GB, LaMa, SAM/YOLO, BiRefNet, Real-ESRGAN).

## Models panel

In-app **Models** lists PowerPaint V1 (dedicated erase) and SDXL Inpaint, and reports whether Best Quality object removal can actually reach them on this machine. Scene-routed erase runs only on CUDA with the weights already downloaded (`PHOTO_ENHANCER_ERASE_ENGINE=auto|powerpaint|sdxl`); otherwise every tier falls back to the classical fill. Readiness comes from `generative_engine_ready()` via `GET /api/removal/generative-status` (also embedded in `/api/models/status` as `generative_removal`).

## Do not claim

Do not advertise “better than Apple Photos Clean Up” until `eval_removal_bench.py` coastal + library cases PASS and you have visually signed off.
