# Reliclear — listing assets (App Review prep)

Local assets for Developer Console logo / screenshot upload. **Not** bundled into the App SPA; upload manually before `apps cut` / App Review.

## Files

| File | Use |
| --- | --- |
| `reliclear-logo-256.png` | **Preferred logo** — clean programmatic teal aperture on warm cream (matches `bundle/styles.css`) |
| `reliclear-logo-256.webp` | Same mark as WebP |
| `reliclear-logo-ai-256.png` / `.webp` | Alternate AI-generated mark (warm/teal; optional) |
| `harness-enhance-local-fallback.png` | Harness UI capture (~2026-10-29) — Reliclear Ready + Enhance local fallback; use as **dev reference**, not as the final store screenshot until host `image.edit` recovers |

## Screenshot checklist (capture when host edit recovers)

Capture from live harness (`anna-app dev`) or installed draft, desktop viewport (~1280×720 or Console guidance):

1. **Hero / ready** — Reliclear brand + Enhance selected + empty dropzone + Ready status  
2. **Enhance success (host)** — Before/After after host `image.edit` PASS (not local fallback)  
3. **Remove background success** — transparent/cutout After stage  
4. **Remove object** — object prompt filled + success After (host path)  
5. Optional: error/unavailable state is fine for internal QA, **avoid** as primary store shot

Until host edit recovers, keep `harness-enhance-local-fallback.png` as the UI reference only.

## Upload in Developer Console

1. Open [Anna Developer Console](https://anna.partners) → **Apps** → **Reliclear** (`reliclear`, id **726**).  
2. Open the draft / listing / store-assets section (wording varies: Listing, Store, or Media).  
3. **Logo:** upload `reliclear-logo-256.png` (or WebP). Square **256×256** (or larger square Console accepts).  
4. **Screenshots:** upload the checklist shots (PNG/JPEG/WebP per Console limits). Prefer host-success flows.  
5. Save / sync listing. Meta (name, tagline, description, category) can also be refreshed from repo via:

```bash
cd anna-photo-enhancer
anna-app apps sync-meta
```

6. Do **not** `apps cut` or `submit-review` until host `image.edit` smoke PASS (or you explicitly accept local-only demos).
