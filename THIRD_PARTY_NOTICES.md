# Third-party notices

Photo Enhancer bundles or downloads models and libraries with their own licenses. This file summarizes the most important ones for **legal compliance** and **AGPL awareness**. It is not exhaustive — check each upstream project before redistribution or commercial use.

## AGPL-licensed components (share source if you modify & distribute)

These are used in **object removal** and **auto-people detection**:

| Component | Use in Photo Enhancer | License |
|-----------|----------------------|---------|
| [SAM 2](https://github.com/facebookresearch/sam2) (Meta) | Click-to-segment for Smart Select | AGPL-3.0 |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Auto-people segmentation (YOLOv8-seg) | AGPL-3.0 |

If you **modify** Photo Enhancer and **distribute** it (including as a network service under AGPL terms), you may need to provide corresponding source for AGPL-covered parts. The project as a whole is licensed under [AGPL-3.0](LICENSE).

**Personal project note:** Replacing these with permissively licensed alternatives is an informal roadmap item — not yet done.

## Other ML models (downloaded at runtime)

| Component | Use | Typical license |
|-----------|-----|-----------------|
| [rembg](https://github.com/danielgatis/rembg) / BiRefNet, BRIA, etc. | Background removal | Check rembg model cards |
| [LaMa](https://github.com/advimman/lama) | Inpainting / object removal | Apache-2.0 (upstream); ONNX ports vary |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Upscale | BSD-3-Clause |
| Various ONNX weights | Quick inpaint, segmentation helpers | Per-model (see Hugging Face / GitHub releases) |

Model files are cached under `%APPDATA%\PhotoEnhancer\models\` (packaged app) or the backend model directory (dev) — **not** committed to git.

## Application dependencies

Major Python and JavaScript libraries (FastAPI, React, PyTorch, OpenCV, Pillow, pywebview, etc.) are subject to their respective licenses in `backend/requirements.txt` and `frontend/package.json`. PyInstaller bundles many of these in the Windows executable.

## Trademarks

**Airbrush**, **Topaz**, **Adobe**, **Photoshop**, **Affinity**, and other names are trademarks of their owners. Photo Enhancer is an **independent personal project** and is not affiliated with, endorsed by, or sponsored by any of them.

## Questions

If you need clarification for a derivative work or commercial deployment, consult a lawyer — this file is informational, not legal advice.
