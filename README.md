# Photo Enhancer

**Local AI photo studio for Windows** — background removal, object removal, upscale, portrait retouch, and 30+ creative filters. **100% offline by default.** No account, no credits, no cloud upload.

![Screenshot placeholder — add docs/screenshots/main.png after capture](docs/screenshots/.gitkeep)

> Screenshots coming soon. Run `python launcher.py` or build the desktop app to try it locally.

## Features

### Enhance (core AI)
- **Background removal** — BiRefNet Massive (default), BRIA RMBG 2.0, Portrait, Matting, Anime, Lite
- **Replace background** — solid color, blurred original, **custom image**, or transparent PNG
- **Object removal** — SAM 2.1 smart-select, brush mask, YOLO auto-people; LaMa AI/Quick inpaint
- **Upscale** — Real-ESRGAN 2× / 4× (configurable output limit, default 150 MP)
- Auto enhance, adjust, denoise, sharpen

### Edit
- **Before/after compare slider** + multi-step **undo/redo**
- **Crop & rotate**
- **Bokeh** — subject-aware blur with circle/heart/hexagon shapes + depth falloff
- **Blur** — global or background-only
- **Eraser** shortcut → object removal
- **Clone stamp**
- **Prism**, vignette, film grain effects
- **Relight** — 9 lighting presets (Flash, Golden Hour, Blue Hour, …)
- **AI Repair** — LaMa inpaint (auto scratch detect or brush)
- **Photo restore lite** — denoise + sharpen chain

### Portrait
- **Magic** — one-tap face-aware polish
- **Skin smooth** — MediaPipe face mask + bilateral filter
- Red-eye fix, teeth whiten
- Simple hair tint, basic lip/blush overlay (quality varies — not generative makeup)

### Looks
- **30 filter presets** with thumbnail gallery (Cinematic, Vintage, Velvia, Noir, …)

### Batch
- Folder batch processing (auto enhance, portrait magic, photo restore)

### AI Chat
- Rule-based (offline) → Ollama (offline) → OpenAI (opt-in cloud)

## Privacy-first

- Photos stay on your machine unless you explicitly configure OpenAI for chat.
- Models download once to `%APPDATA%\PhotoEnhancer\models\`.
- No telemetry, no ads, no subscription.

## Quick start

### One-click (Windows)

```bat
start.bat
```

Or:

```bash
python launcher.py
```

Opens **http://127.0.0.1:8000** after building the GUI on first run.

### Manual setup

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
npm run build
```

### Dev mode

```bash
# Terminal 1
cd backend && venv\Scripts\uvicorn main:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
```

Open http://localhost:5173

## Build Windows desktop app

```bat
build.bat
```

Output: `dist\PhotoEnhancer\PhotoEnhancer.exe` (not committed — too large).

See README build section for WebView2, PyInstaller, and optional Inno Setup installer.

## System requirements

| | Minimum | Recommended |
|---|---------|-------------|
| OS | Windows 10/11 (64-bit) | Windows 11 |
| RAM | 8 GB | 16 GB+ |
| Disk | 10 GB free (for AI models) | 20 GB+ |
| GPU | Optional (CPU works) | NVIDIA CUDA for faster AI |
| Runtime | WebView2 (pre-installed on Windows 11) | Latest WebView2 runtime |
| Network | Internet required only for initial model download | Stable broadband for first setup |

## First run

On first launch, Photo Enhancer shows a welcome screen explaining that **AI models download once to your computer (~6–8 GB total)**. After that, **everything works offline** — your photos never leave your device.

You can choose:

- **Download all now** — prefetch every model for the fastest experience afterward
- **Download as I go** — models download automatically the first time you use each feature

Progress appears in the app UI (not just server logs). Open **Models** in the toolbar anytime to check status, disk usage, or download remaining models.

Models are stored in:

```
%APPDATA%\PhotoEnhancer\models\
```

## Model downloads

Models are **not** in git. First use of each feature triggers download (~200 MB – 1 GB per model, ~6–8 GB total if everything is used). You can also prefetch all models from the in-app **Models** panel.

## Configuration

Copy `.env.example` to `.env` in the project root (optional):

- `OLLAMA_HOST`, `OLLAMA_MODEL` — local chat LLM
- `OPENAI_API_KEY` — opt-in cloud chat only
- `CHAT_BACKEND=auto|rules|ollama|openai`
- `MAX_OUTPUT_MP` — upscale output cap in megapixels (50–400, default **150**). Override the server default; the in-app **Upscale** tool and **Models** panel let you pick Conservative (100 MP), Standard (150 MP), High (200 MP), or a custom value. Preferences sync to `%APPDATA%\PhotoEnhancer\settings.json` on the desktop app.

## ⚠️ AGPL notice (commercial use)

Object selection uses **Ultralytics** (SAM 2.1 + YOLOv8), which is **AGPL-3.0**. Commercial redistribution may require:

1. Complying with AGPL source-offer requirements, **or**
2. An [Ultralytics enterprise license](https://www.ultralytics.com/license), **or**
3. Replacing SAM/YOLO with permissively licensed alternatives

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full attribution.

## License

Application source code: [MIT](LICENSE)

Third-party libraries and AI models: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and PRs welcome.

## Security

- Local-only processing for all image tools.
- OpenAI is opt-in; images are only sent if you set `OPENAI_API_KEY`.
- No hardcoded secrets in the repository — use `.env` (gitignored).

## What this is not

Generative features (AI expand/replace, face reshape, virtual hairstyles, full makeup, headshots) are **out of scope** for the fully-local promise. Photo Enhancer targets **offline cleanup + portrait-friendly retouch**, not a full Airbrush clone.
