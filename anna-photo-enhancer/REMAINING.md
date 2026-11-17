# Reliclear — remaining desktop parity gaps

Shipped in Anna this pass (browser canvas + existing local enhance/rembg):

| Desktop tool | Anna path |
| --- | --- |
| Auto Enhance | `enhance` → `local_enhance` (+ optional host edit) |
| Remove BG | `remove_bg` → `local_remove_bg` / rembg |
| Crop / Rotate / Flip | UI canvas |
| Adjust | UI canvas |
| Sharpen / Denoise / Restore | UI canvas (denoise is lite box-blur, not OpenCV NLMeans) |
| Looks (subset of 12) | UI canvas |
| Blur / Vignette / Grain / Prism | UI canvas |

## Class C — still desktop-only

| Feature | Why deferred |
| --- | --- |
| Remove Objects (SAM + PowerPaint / SDXL) | Multi-GB CUDA models; host `image.edit` still unreliable |
| Upscale (Real-ESRGAN 2×/4×) | Heavy weights; needs GPU or large CPU package |
| Portrait Magic / skin / red-eye / teeth / hair / lips | Face models + CUDA portrait pipeline |
| Bokeh (subject-aware) / selective blur | Needs rembg/BiRefNet mask quality + OpenCV depth blend in packaged binary |
| Replace BG (color/blur/image) | Depends on high-quality matting models |
| Stamp (clone stamp) | Complex brush UI; no generative model |
| Eraser / AI Repair | OpenCV TELEA exists locally but needs mask paint UI; AI repair needs models |
| Relight presets | Desktop relight service not packaged |
| Batch folder processing | Desktop filesystem + queue; Anna App has no folder picker |
| Full 30+ Looks catalog | Subset of 12 shipped; rest are trivial to add later |
| Generative inpaint via host | Blocked on platform `image.edit` reliability |

## Next packaging steps (optional)

1. Rebuild `photo-enhancer-local` binaries (all platforms) with PIL `local_adjust` / `local_style` / OpenCV denoise for Agent chat tools.
2. Mask-paint UI → `local_inpaint` for a non-generative eraser.
3. Lite bokeh via rembg alpha + Gaussian when rembg is present.
4. Revisit host `image.edit` for remove_object when Anna recovers the edit gateway.
