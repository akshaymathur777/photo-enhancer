# Creating a GitHub Release

Photo Enhancer **v0.1.0** is a personal beta. The Windows desktop bundle is too large for git (~1–3 GB with PyTorch). Ship binaries through **GitHub Releases** instead.

## Prerequisites

- Built desktop app at `dist\PhotoEnhancer\PhotoEnhancer.exe` (run `build.bat` from repo root)
- Private repo pushed to GitHub (do not make public unless you intend to share widely)
- [GitHub CLI](https://cli.github.com/) (`gh`) optional but convenient

## 1. Tag the release locally

From the repo root, on a clean `main` that matches what you want to ship:

```powershell
git tag -a v0.1.0 -m "Photo Enhancer v0.1.0 — first shareable beta"
git push origin v0.1.0
```

## 2. Create the release on GitHub

### Option A — GitHub website

1. Open your repo → **Releases** → **Draft a new release**
2. Choose tag **v0.1.0**
3. Title: `Photo Enhancer v0.1.0`
4. Paste release notes (see template below)
5. Attach **`PhotoEnhancer.exe`** or zip the whole `dist\PhotoEnhancer\` folder
6. Publish (or keep as draft while repo stays private)

### Option B — GitHub CLI

```powershell
# Zip the folder (recommended — preserves WebView2 helper files)
Compress-Archive -Path dist\PhotoEnhancer\* -DestinationPath PhotoEnhancer-v0.1.0-win64.zip

gh release create v0.1.0 PhotoEnhancer-v0.1.0-win64.zip `
  --title "Photo Enhancer v0.1.0" `
  --notes-file docs/release-notes-v0.1.0.txt
```

Create `docs/release-notes-v0.1.0.txt` locally if you use `--notes-file` (that file is optional and need not be committed).

## Release notes template

```markdown
## Photo Enhancer v0.1.0 (beta)

Local AI photo studio for Windows — offline background removal, object removal, upscale, portrait retouch, and 30+ filters.

### Install

1. Download `PhotoEnhancer-v0.1.0-win64.zip` (or `PhotoEnhancer.exe` + bundled folder)
2. Extract anywhere (avoid syncing folders like OneDrive during first launch)
3. Run `PhotoEnhancer.exe`
4. On first run, AI models download once (~6–8 GB total) to `%APPDATA%\PhotoEnhancer\models\`

### Requirements

- Windows 10/11 (64-bit), 8 GB+ RAM, 10 GB+ free disk
- Internet only for initial model download
- WebView2 (pre-installed on Windows 11)

### Notes

- **Unsigned build** — Windows SmartScreen may warn; this is expected for personal builds
- **AGPL** — object removal uses Ultralytics SAM/YOLO; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) before commercial redistribution
- **Optional cloud** — OpenAI chat only if you add `OPENAI_API_KEY`; all image tools stay local

### Source

Build from source: see [README.md](../README.md)
```

## What not to commit

| Item | Reason |
|------|--------|
| `dist/` | PyInstaller output, huge |
| `*.pt`, `*.pth` weights | Downloaded at runtime |
| `.env` | Secrets |
| Test JPG/PNG outputs | Local dev artifacts |

## After publishing

- Replace placeholder screenshots in `docs/screenshots/` with real captures
- Smoke-test the release zip on a clean Windows VM without Python/Node installed
- Optional: code-sign the `.exe` to reduce SmartScreen friction
