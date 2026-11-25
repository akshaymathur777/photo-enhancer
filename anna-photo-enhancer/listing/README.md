# Reliclear — listing assets (App Review prep)

Local assets for Developer Console logo / screenshot upload. **Not** bundled into the App SPA; upload manually before `apps cut` / App Review.

## Files

| File | Use |
| --- | --- |
| `reliclear-logo-256.png` | **Preferred logo** — clean programmatic teal aperture on warm cream (matches `bundle/styles.css`) |
| `reliclear-logo-256.webp` | Same mark as WebP |
| `reliclear-logo-ai-256.png` / `.webp` | Alternate AI-generated mark (warm/teal; optional) |
| `harness-enhance-local-fallback.png` | Harness UI capture — Enhance **local** fallback success; acceptable store shot for v0.1 local-first cut |
| `harness-v01-ready.png` | Fresh harness dashboard capture (2026-11-11) for Console screenshot upload |

## Screenshot checklist (expanded editing cut)

Capture from live harness (`anna-app dev`) or installed draft, desktop viewport (~1280×720 or Console guidance):

1. **Hero / ready** — Reliclear brand + Clean/Edit/Looks/Effects categories + empty dropzone  
2. **Enhance or Remove BG success** — Before/After (local classical / rembg OK)  
3. **Edit or Looks** — Adjust/Crop or a Looks preset with Before/After  
4. Do **not** claim Remove object / Upscale / Portrait (desktop-only; see `REMAINING.md`)

## Upload in Developer Console

1. Open [Anna Developer Console](https://anna.partners) → **Apps** → **Reliclear** (`reliclear`, id **726**).  
2. Open the draft / listing / store-assets section (wording varies: Listing, Store, or Media).  
3. **Logo:** upload `reliclear-logo-256.png` (or WebP). Square **256×256** (or larger square Console accepts).  
4. **Screenshots:** upload the checklist shots (PNG/JPEG/WebP per Console limits). Local Enhance / rembg BG success is fine for v0.1.  
5. Save / sync listing. Meta (name, tagline, description, category) can also be refreshed from repo via:

```bash
cd anna-photo-enhancer
anna-app apps sync-meta
```

6. After rembg + Enhance smoke PASS: `apps cut 0.1.0` then `apps submit-review`.
