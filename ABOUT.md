# About Photo Enhancer

Photo Enhancer started as a **personal learning project**: I wanted a desktop photo editor that runs entirely on my Windows PC — no subscription, no cloud upload, no account — and uses the same modern ML models people talk about online (segmentation, inpainting, upscaling).

It grew from "can I wire BiRefNet + LaMa + Real-ESRGAN into one UI?" into something I actually use and want to share.

## Why local / privacy-first?

Not as a marketing slogan — I personally do not like sending family photos to random APIs. Keeping inference on-device is a design choice that matches how I want to edit. Optional cloud chat (OpenAI) exists only if you add a key; the default path is offline.

## What's working

- Background removal (multiple rembg models)
- Object removal: SAM click-select, brush, auto-people, LaMa inpaint — plus an optional CUDA-only generative erase (PowerPaint / SDXL) on the Best Quality tier, if you download those weights
- Upscale (Real-ESRGAN), auto-enhance, 30+ AI style presets
- Crop, bokeh, relight, AI repair, basic portrait tools
- Undo/redo, before/after compare slider
- AI chat: rule-based offline default; optional Ollama; optional OpenAI
- Windows desktop build (PyInstaller + WebView2) and browser launcher

## What's limited or not planned (for now)

- **Generative expand/replace** (text-prompt inpainting) — too heavy for CPU-first local use; not implemented. The only generative path is the object-removal erase, and it needs CUDA plus multi-GB weights, so most machines stay on the classical fill
- **Studio-grade face/hair/makeup** — would need cloud models or much larger local weights; only simple local approximations today
- **Video** — out of scope
- **Perfect parity with commercial apps** — unrealistic for a side project; see [AIRBRUSH_PC_REPLICATION.md](AIRBRUSH_PC_REPLICATION.md) for an honest matrix

## Honest expectations

- First run downloads models per feature (~8 GB if you use everything)
- CPU inference is slow on large images for the heaviest tools
- Some antivirus tools flag PyInstaller + PyTorch bundles; unsigned builds may trigger false positives
- AGPL applies to parts of the object-removal stack — read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing modified builds

## Roadmap (informal)

Rough priorities if I keep hacking on this:

1. Polish UX (screenshots, docs, first-run onboarding)
2. More filter presets and batch folder processing
3. Replace or relicense AGPL segmentation/detection where feasible
4. Maybe Android local — see [ANDROID_LOCAL_PLAN.md](ANDROID_LOCAL_PLAN.md)

No promises on timeline — this is nights-and-weekends work.

## Get involved

Questions, ideas, and pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
