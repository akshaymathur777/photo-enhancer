# Reliclear — Anna App

**Reliclear** enhances photos, removes backgrounds, and erases objects — as an Anna Marketplace App (Anna edition of the desktop Photo Enhancer project).

| Track | What | Where |
| --- | --- | --- |
| **A (default)** | Host `upload.*` + `image.edit` App UI + chat tools | `executas/photo-enhancer` + `bundle/` |
| **B (optional)** | Classical CPU enhance / OpenCV inpaint / optional rembg | `executas/photo-enhancer-local` |

Generative CUDA (SDXL / PowerPaint) stays on the desktop app for now; B is the sandbox-safe classical slice.

## Quick start

```bash
cd anna-photo-enhancer
npm i -g @anna-ai/cli            # once
anna-app doctor
anna-app login --host https://anna.partners   # if needed
anna-app whoami                  # expect https://anna.partners + aps:dev
npm run validate                 # or: anna-app validate --strict
anna-app dev --no-llm            # UI harness offline (no host image/upload)
anna-app dev                     # live host APIs → http://localhost:5180/
npm test                         # node + executa + classical suites
```

**Prereqs:** Node 22+, [uv](https://docs.astral.sh/uv/), `@anna-ai/cli`. After Developer activate, set handle once: `anna-app account set-handle <handle>`.

**Status (2026-11-02 ~11:39 UTC):** Listing name **Reliclear**, slug **`reliclear`** (id=726, draft **rev 4**). Host `image.edit` still Cloudflare **502 HTML** on full A1–A3 smoke (pinned `fal-ai/nano-banana-2`; Rays `a542d9aba8021387` / `a542dabc5c5856d2` / `a542db4bb979be32`) — past Oct 30 `1.1.0-beta.201` day. Exit criterion **NOT MET**. Escalate forum draft [`fixtures/FORUM-IMAGE-EDIT-502.md`](./fixtures/FORUM-IMAGE-EDIT-502.md). Do **not** submit App Review. Listing logo + screenshot checklist: [`listing/`](./listing/).

If host `image.edit` returns 502, **Enhance** falls back to `local_enhance` and **Remove background** tries `local_remove_bg` when rembg is installed. **Remove object** shows a clear error (no fake inpaint) until host edit recovers:

```bash
cd executas/photo-enhancer-local
uv pip install "rembg[cpu]>=2.0"   # preferred while harness holds Scripts/*.exe
# optional: uv sync --extra rembg  (when console script is not locked)
```

Lite rembg uses the small `u2netp` ONNX (~5MB, cached under `~/.rembg`).

## Thorough testing

```bash
cd anna-photo-enhancer
npm test
```

| Layer | Command | Covers |
| --- | --- | --- |
| Manifest / ACL | `npm run validate` + `tests/host_acl.test.mjs` | schema, `host_api`, CSP, mountBundle deny paths |
| Prompt / bundle | `tests/prompts.test.mjs` | mode prompts, upload purpose, edit call shape |
| Option A Executa | `npm run test:executa` | describe/health/ping/list_modes + mock-image enhance/bg/object + error paths |
| Option B classical | `npm run test:local` | PIL enhance, OpenCV inpaint, Executa invoke smoke |
| UI harness | `anna-app dev --no-llm` + manual/browser | dropzone, modes, connecting→ready, process error without grants |

### Test matrix (A vs B)

| Capability | A (host APIs) | B (local classical) |
| --- | --- | --- |
| Enhance | `image.edit` prompt | `local_enhance` (PIL) |
| Remove background | `image.edit` prompt | `local_remove_bg` (rembg, optional extra) |
| Remove / inpaint object | `image.edit` prompt | `local_inpaint` (OpenCV TELEA/NS + mask) |
| Needs Anna image grant | Yes | No |
| Needs local Python deps | No (chat Executa is thin) | numpy, Pillow, opencv-headless |
| CUDA generative | N/A | Not in B yet (desktop) |

Fixtures: `fixtures/sample.png`, `fixtures/image-mocks.jsonl`.

## Layout

```
anna-photo-enhancer/
├── app.json / manifest.json
├── bundle/                 # Option A UI
├── executas/
│   ├── photo-enhancer/     # Option A host image.edit tools
│   └── photo-enhancer-local/  # Option B classical
├── fixtures/
└── tests/
```

## Publish

```bash
anna-app login
anna-app validate --strict
anna-app apps push          # refresh working draft Reliclear (id=726, slug reliclear); does not cut a release
anna-app apps sync-meta     # push name/tagline/description/category from app.json
# anna-app apps cut 0.1.0   # only after host image.edit recovers + smoke PASS
# anna-app apps submit-review   # only with explicit product go-ahead
```

`bundled:photo-enhancer` is required; `bundled:photo-enhancer-local` is optional. Platform resolves handles to real `tool_id`s on publish. For shipping B as a multi-file Agent binary later, see Anna’s binary / `distribution_type: local` docs.

### Listing assets (logo / screenshots)

Prep files live under [`listing/`](./listing/) — see that folder’s README for the screenshot checklist.

**Upload in Developer Console** (manual; not via `apps push`):

1. Open https://anna.partners → Developer → **Apps** → **Reliclear** (id **726**, slug `reliclear`).
2. In Listing / Store / Media, upload **logo** `listing/reliclear-logo-256.png` (256×256 PNG or WebP).
3. Upload store **screenshots** once host Enhance/Remove BG/Remove Object succeed (checklist in `listing/README.md`). The harness capture `listing/harness-enhance-local-fallback.png` is a UI reference only until host edit recovers.
4. Refresh text meta anytime: `anna-app apps sync-meta`.
5. Do **not** cut / submit-review until host `image.edit` smoke PASS.

### Next actions

1. Re-probe live Enhance when Anna confirms edit recovery (Oct 30 probe still **502 HTML**).
2. If host edit PASS: full A1–A3 (+ A4–A6) → update STATUS / TEST-REPORT → Console logo/screenshots → consider `apps cut` → App Review (explicit go-ahead only).
3. Optional: install draft id=726 from Developer Console; post `fixtures/FORUM-IMAGE-EDIT-502.md` if escalating (502 persists past scheduled fix day).
