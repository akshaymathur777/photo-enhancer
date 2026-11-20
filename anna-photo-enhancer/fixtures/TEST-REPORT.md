# Photo Enhancer Anna App — test report

Branch: `feature/anna-app-slim`  
Date: **2026-10-28** (continue ~22:15–22:20 UTC — Enhance host still **502** / local PASS; Remove Object copy polish; `apps push` → **rev 2**; earlier full matrix ~20:11–20:15)  
CLI: `@anna-ai/cli` **0.1.61** · Node **v24.22.1** · Python **3.14.3**  
Scope: Option A production-proof + **graceful Enhance + Remove BG fallbacks** while host edit 502s; polished Remove Object empty/error copy; draft **rev 2**; **no Option B/CUDA expansion**; **no App Review submit**

## Environment / auth

| Check | Result |
| --- | --- |
| `git branch` | `feature/anna-app-slim` |
| `anna-app doctor` | **FAIL** (only): `~/.anna-app/dev.key` mode 666 vs expected 0600 — Windows NTFS ACL is owner R/W; harness still starts |
| `anna-app whoami` | **LOGGED IN** — host `https://anna.partners`, scopes `aps:dev`, PAT ~89d (`~/.config/anna/credentials.json` present; loose-perms warning on Windows) |
| Developer handle | **`@akshay`** via Developer Console activate + `anna-app account set-handle akshay` |
| Dev app on nexus | **[DRAFT] `anna-photo-enhancer` (id=726)** |
| Live host APIs | **Partial** — `upload.inline` OK; `image.edit` still **HTTP 502** (retry ~22:18 UTC 2026-10-28); `image.generate` last OK ~20:15 |
| Draft push | **PASS** ~22:19 UTC — working draft **rev 2** |

## Automated

| Suite | Result | Evidence |
| --- | --- | --- |
| `anna-app validate --strict` | **PASS** | `✓ validate passed` (dispatcher `@anna-ai/app-schema` 0.30.0) — re-run 2026-10-28 ~22:19 UTC |
| `npm test` (node + executa + classical) | **PASS** | exit 0 (16 node + 15 executa + classical; rembg unit skipped on system Python without rembg) — re-run ~22:19 UTC |
| `npm run test:node` | **PASS** | 16/16 |
| `npm run test:executa` | **PASS** | 15/15 |
| `python tests/test_local_classical.py` | **PASS** | 7/7 (+ rembg when using local `.venv`) |

## Harness UI (browser CDP) — offline

Harness: `anna-app dev --no-llm` → http://localhost:5180/  
(Prior session 2026-10-27)

| Check | Result | Evidence |
| --- | --- | --- |
| Connect → status Ready | **PASS** | UI Ready; harness `connected` |
| Mode pills Enhance / Remove BG / Remove object | **PASS** | `aria-pressed` toggles |
| Object prompt visible only for remove_object | **PASS** | |
| File pick enables Process | **PASS** | `sample.png` |
| Empty object text → describe-what-to-remove validation | **PASS** | copy now includes example hint |
| Process under `--no-llm` → clear Error | **PASS** | `llm_disabled` |

## Live production proof (Option A) — 2026-10-27 ~12:22–12:31 UTC+0 / local

Login device code that succeeded: **`RTBF-3788`** (prior `UCN8-PRGT` poll expired). Confirmed in Edge → “Device authorized”.

Harness: `anna-app validate --strict` then `anna-app dev` **without** `--no-llm` → `http://localhost:5180/`  
`llm bridge real → app_slug=anna-photo-enhancer` · harness left **running** for retries.

Fixture: `fixtures/sample.png` (1354 bytes) injected via CDP into App iframe.

| Check | Result | Evidence |
| --- | --- | --- |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~90d |
| Developer activate + handle `@akshay` | **PASS** | Console “Become a developer” + CLI set-handle |
| `anna-app validate --strict` | **PASS** | |
| `anna-app dev` (live LLM) | **PASS** | dashboard up; first attempt needed handle |
| Grants `upload` / `image.allowEdit` | **Effective in harness** | Not `APP_NOT_GRANTED`; `upload.inline` + `image.edit` RPC issued. CLI `apps grants anna-photo-enhancer` → “(no grants endpoint data available)” for DRAFT |
| **A1 Enhance (host edit)** | **FAIL (host)** | `upload.inline` **ok** → R2; `image.edit` → `res err transport HTTP 502` |
| **A2 Remove BG (host)** | **FAIL (host)** | Same path: upload ok / `image.edit` **HTTP 502** |
| **A3 Remove object (host)** | **FAIL (host)** | Same with object text “the watermark” |
| Token / CU / R2 notes | Upload billed path exercised; R2 URL returned; image.edit never completed so no edit CU observed in UI |

### RPC evidence (Enhance, representative)

1. `→ upload.inline` with `sample.png` / `purpose: image_input`  
2. `← res ok upload.inline` with `r2_key=exec-uploads/production/.../sample.png`, `download_url` on `*.r2.cloudflarestorage.com`, `expires_in:1800`  
3. `→ image.edit` with enhance prompt + that `image_url`  
4. `← res err transport HTTP 502` (HTML Cloudflare error page)

Homepage `https://anna.partners/` returned **200** while image.edit still **502** — image backend / gateway fault, not full site outage.

## Live edit retry + app hardening — 2026-10-27 ~12:55 UTC

Harness: **reused** existing `anna-app dev` on `http://localhost:5180/` (`llm bridge real`).

| Check | Result |
| --- | --- |
| `upload.inline` | **PASS** |
| `image.edit` default | **FAIL** `transport` HTTP 502 CF HTML |
| `image.edit` + `model: "fal-ai/nano-banana-2"` | **FAIL** same 502 |

### Code shipped (pin + 502 UX)

- `bundle/app.js`: `EDIT_MODEL = "fal-ai/nano-banana-2"`; `formatHostError()` maps 502 / Bad Gateway / `APP_PROVIDER_ERROR` / Cloudflare dumps to a clear platform-unavailable status (no HTML dump).

## Local enhance fallback — 2026-10-27 (same harness :5180)

**Did not spawn a competing harness** — reused pid running since ~12:35 UTC.

| Check | Result | Evidence |
| --- | --- | --- |
| Served `app.js` includes `local_enhance` / `isHostEditUnavailable` | **PASS** | CDP fetch of `/anna-apps/anna-photo-enhancer/dev/app.js` |
| Hero tagline | **PASS** | “Enhance · remove background · remove objects — via Anna” |
| **A1 Enhance → local fallback** | **PASS** | After `data:image/png;base64,…` (len ~2646); status **“Host edit unavailable — used local classical enhance.”**; `statusIsError=false`; Open result enabled |
| Process disabled while running | **PASS** | `startedDisabled=true`; re-enabled after (`runDisabledFinal=false`) |
| **A2 Remove BG** (rembg absent) | **PASS** | Clear “Platform image edit is temporarily unavailable…”; **no HTML dump**; no fake success |
| `host_api.tools` includes local | **PASS** | `bundled:photo-enhancer-local` asserted in tests |

### Code shipped this pass

- `bundle/app.js`: on host edit 502 / provider error → Enhance uses `anna.tools.invoke` (`tool-dev-photo-enhancer-local` / bundled handles) method `local_enhance` with PNG b64 from local File; optional `local_remove_bg` for remove_bg; remove_object stays platform-unavailable.
- `bundle/index.html`: clearer tagline.
- Tests: fallback + `host_api.tools` local + Process disable assertions.
- Docs: PHASE1 / PLAN / this report; forum draft `fixtures/FORUM-IMAGE-EDIT-502.md` (**not posted**).
- **No commit.**

## Live edit retry + Remove BG rembg — 2026-10-27 ~14:37–14:45 UTC

Harness: **reused** `anna-app dev` on `http://localhost:5180/` (`llm bridge real`, pid since ~12:35).

| Check | Result | Evidence |
| --- | --- | --- |
| `upload.inline` | **PASS** | `expires_in: 1800` |
| `image.edit` + nano-banana-2 | **FAIL** | `transport` HTTP 502 CF HTML (still down) |
| `uv pip install "rembg[cpu]"` into local executa `.venv` | **PASS** | rembg 2.0.86 + onnxruntime; `uv add --extra` blocked by harness-locked `.exe` |
| lite model | **`u2netp`** (~4.6MB cached under `~/.rembg`) | first invoke ~29s incl. download |
| Harness `capabilities.remove_bg_rembg` | **true** | tools.invoke |
| **A2 Remove BG → local fallback** | **PASS** | After data URL; status **“Host edit unavailable — used local background removal.”** |
| `anna-app apps push` (draft) | **PASS** | `working draft updated (rev 1)`; status **draft**; no cut / no submit-review |
| Push note | Fixed `user_message_prefix_template` → `[Photo Enhancer] {user_message}` | 400 without placeholder |
| Slug | `app.json` slug aligned to **`anna-photo-enhancer`** (matches id=726) | |

### Code / docs this pass

- `classical_ops.py`: lite rembg → `u2netp`; clearer docstring  
- `bundle/app.js`: stronger remove_bg fallback UX + rembg install hint; canvas always re-encode for local PNG; broader `extractLocalPngB64`  
- `pyproject.toml` / README: rembg install notes (`uv pip install` while harness locks Scripts)  
- `app.json` tagline/description polish; `manifest.json` `{user_message}` placeholder  
- Tests: rembg optional unit; local smoke invokes plugin via `.venv` (avoids harness file lock)  
- **No commit. No App Review.**

## Live edit retry — 2026-10-27 ~21:14–21:17 UTC

Harness: **fresh** `anna-app dev` (prior :5180 dead; stale session had `unknown session_id: sess-1`). `llm bridge real → app_slug=anna-photo-enhancer`. Fixture: `fixtures/sample.png`.

| Check | Result | Evidence |
| --- | --- | --- |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~90d |
| `upload.inline` | **PASS** | R2 `expires_in: 1800` |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` → `transport HTTP 502` CF HTML (`anna.partners \| 502: Bad gateway`) |
| **A2 Remove BG (host)** | **FAIL (host)** | Same 502 |
| **A3 Remove object (host)** | **FAIL (host)** | Same 502 with object text “the watermark” |
| **A1 Enhance → local fallback** | **PASS** | After data URL; status **“Host edit unavailable — used local classical enhance.”**; download enabled |
| **A2 Remove BG → local rembg** | **PASS** | After data URL; status **“Host edit unavailable — used local background removal.”**; engine rembg lite |
| **A3 Remove object (no fake)** | **PASS (clear fail)** | Status: needs platform `image.edit` **or paint mask for local inpaint**; platform unavailable; **no** After preview |
| Local tool repair | **PASS** | Reinstalled `tool-dev-photo-enhancer-local` console script after prior Access-denied venv uninstall |
| `validate --strict` + `npm test` | **PASS** | after Remove Object message change |

### Code this pass

- `bundle/app.js`: Remove Object on host-edit outage → explicit “needs platform image.edit (or a paint mask…)” — **no fake inpaint**.
- Tests assert that copy.
- **No commit. No App Review. No `apps push` this pass** (optional later to refresh draft copy).

## Live edit retry — 2026-10-28 ~19:00–19:03 UTC

Harness: **fresh** `anna-app dev` (prior :5180 was down). `llm bridge real → app_slug=anna-photo-enhancer`. Fixture: `fixtures/sample.png` (SDK upload + UI fallback path).

| Check | Result | Evidence |
| --- | --- | --- |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~89d |
| `upload.inline` | **PASS** | R2 `expires_in: 1800`; key under `app:726/.../sample.png` |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` + nano-banana-2 → `transport HTTP 502` CF HTML (~2.2s) |
| **A2 Remove BG (host)** | **FAIL (host)** | Same 502 (~1.7s) |
| **A3 Remove object (host)** | **FAIL (host)** | Same 502 (~1.5s) with object text “the watermark” |
| **A1 Enhance → local fallback** | **PASS** | After data URL; status **“Host edit unavailable — used local classical enhance.”** |
| **A2 Remove BG → local rembg** | **PASS** | After data URL; status **“Host edit unavailable — used local background removal.”** |
| **A3 Remove object (no fake)** | **PASS (clear fail)** | Status needs platform `image.edit` or paint mask; download disabled |
| Code / validate / push | **N/A** | No code changes this pass; no validate/test/push; **no App Review** |

## Full matrix — 2026-10-28 ~20:11–20:15 UTC

Harness: **fresh** `anna-app dev` (no competing harness; :5180 was free). `llm bridge real → app_slug=anna-photo-enhancer`. Fixture: `fixtures/sample.png` via Playwright UI drive. Automated `validate --strict` + `npm test` **PASS** same session. Offline `--no-llm` UI **skipped** (live UI covered Connect/Ready + modes).

| Check | Result | Evidence |
| --- | --- | --- |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~89d |
| `anna-app doctor` | **NOTE FAIL** | `dev.key` mode 666 vs 0600 (Windows noise); harness started |
| `anna-app validate --strict` | **PASS** | |
| `npm test` | **PASS** | 16 node + 15 executa + classical |
| Connect → Ready | **PASS** | “Ready — drop a photo to begin.” |
| File pick / upload UI | **PASS** | `sample.png · 1 KB ready.` |
| `upload.inline` | **PASS** | R2 `expires_in: 1800`; key under `app:726/.../sample.png` |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` + nano-banana-2 → `transport HTTP 502` CF HTML (~1.6s) |
| **A1 Enhance → local fallback** | **PASS** | After `data:image/png;base64,…` (len 2646); status **“Host edit unavailable — used local classical enhance.”**; Open result enabled |
| **A2 Remove BG (host)** | **FAIL (host)** | Same 502 (~1.6s) |
| **A2 Remove BG → local rembg** | **PASS** | After data URL (len 8122); status **“Host edit unavailable — used local background removal.”**; engine rembg lite |
| **A3 Remove object (host)** | **FAIL (host)** | Same 502 (~2.7s) with “the watermark” |
| **A3 Remove object (no fake)** | **PASS (clear fail)** | Error status: needs platform `image.edit` or paint mask; **download disabled**; no new success path (After stage may retain prior rembg preview) |
| `image.generate` control | **PASS** | `openai/gpt-image-2`; URL returned (~83s) |
| Code / push / App Review | **N/A** | No app code changes; **no commit**; **no App Review** |

**Next wait:** forum #400 host fix targeted ~**2026-10-30**; re-smoke host A1–A3 then.

## Continue — 2026-10-28 ~22:15–22:20 UTC

Harness: **fresh** `anna-app dev` (`:5180` was down). `llm bridge real → app_slug=anna-photo-enhancer`. One Enhance probe only (per continue brief).

| Check | Result | Evidence |
| --- | --- | --- |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~89d |
| `upload.inline` | **PASS** | R2 `expires_in: 1800`; key under `app:726/.../sample.png` |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` + nano-banana-2 → `transport HTTP 502` CF HTML (~1.8s) |
| **A1 Enhance → local fallback** | **PASS** | After data URL; status **“Host edit unavailable — used local classical enhance.”** |
| Remove Object copy | **Updated** | Empty: “Describe what to remove — for example…”; 502: needs Anna's photo-edit + try Enhance/Remove BG |
| `anna-app validate --strict` | **PASS** | after copy change |
| `npm test` | **PASS** | 16 node + 15 executa + classical |
| `anna-app apps push` | **PASS** | working draft **rev 2** |
| Commit / App Review | **N/A** | **no commit**; **no App Review** |

## Re-probe — 2026-10-29 ~08:44–08:47 UTC

Harness: **fresh** `anna-app dev` (`:5180` was down). `llm bridge real → app_slug=anna-photo-enhancer`. Fixture `fixtures/sample.png` via CDP. **Host `image.edit` not recovered.**

| Check | Result | Evidence |
| --- | --- | --- |
| OS `whoami` | **PASS** | `akshay\aksha` |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~88d |
| `upload.inline` | **PASS** | Issued before each edit attempt |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` → `transport HTTP 502` CF HTML |
| **A1 Enhance → local fallback** | **PASS** | After data URL; status *Done — enhanced locally (Anna photo-edit unavailable)…*; download enabled |
| **A2 Remove BG (host)** | **FAIL (host)** | Same 502 |
| **A2 Remove BG → local rembg** | **PASS** | After data URL; path `local_remove_bg` / engine `rembg`; download enabled |
| **A3 Remove object (host)** | **FAIL (host)** | Same 502 with “the watermark” |
| **A3 Remove object (no fake)** | **PASS (clear fail)** | Error: needs Anna photo-edit; try Enhance/Remove BG; **download disabled**; After placeholder |
| Commit / App Review | **N/A** | **no commit**; **no App Review**; STATUS timestamp updated only |

## Re-probe + listing assets — 2026-10-29 ~22:05–22:12 UTC

Harness: **one** fresh `anna-app dev --llm-app-slug reliclear --slug reliclear` (`:5180` was free). `llm bridge real → app_slug=reliclear`. Fixture `fixtures/sample.png` via CDP into App iframe. **Enhance host only** (per brief — skip A2/A3 host if Enhance fails).

| Check | Result | Evidence |
| --- | --- | --- |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~88d |
| Connect → Ready | **PASS** | Reliclear brand; “Ready — upload a photo to begin.” |
| `upload.inline` | **PASS** | R2 `expires_in: 1800`; key under `app:726/.../sample.png` (~22:08:05 UTC) |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` + `model: fal-ai/nano-banana-2` → `transport HTTP 502` CF HTML title `anna.partners \| 502: Bad gateway` (~22:08:05→10, ~5.3s) |
| **A1 Enhance → local fallback** | **PASS** | Status *Done — enhanced locally (Anna photo-edit unavailable)…*; `local_enhance` / `classical_enhance`; download enabled |
| Host A2 / A3 | **SKIPPED** | Enhance host failed — no Remove BG / Remove Object host smoke |
| Listing logo | **CREATED** | `listing/reliclear-logo-256.png` (+ WebP; AI alt variants) |
| Harness screenshot | **SAVED** | `listing/harness-enhance-local-fallback.png` (UI reference; local fallback, not store-final) |
| Console upload docs | **DONE** | `listing/README.md` + README / STATUS |
| `apps push` / cut / review | **N/A** | Draft stays **rev 4**; **no push** (docs/listing only); **no commit**; **no App Review** |

## Post–`1.1.0-beta.201` day re-probe — 2026-10-30 ~13:34–13:37 UTC

User “try now” after scheduled host `1.1.0-beta.201` image.edit fix day. Harness: **one** fresh `anna-app dev --llm-app-slug reliclear --slug reliclear`. `llm bridge real → app_slug=reliclear`. Fixture `fixtures/sample.png` via CDP. **Recovered: NO.**

| Check | Result | Evidence |
| --- | --- | --- |
| OS `whoami` | **PASS** | `akshay\aksha` |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~87d |
| Connect → Ready | **PASS** | Reliclear; “Ready — upload a photo to begin.” |
| `upload.inline` | **PASS** | R2 `expires_in: 1800`; key under `app:726/.../sample.png` (~13:36:05→07 UTC) |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` + `model: fal-ai/nano-banana-2` → `transport HTTP 502: <!DOCTYPE html>` CF HTML title `anna.partners \| 502: Bad gateway` (~13:36:07→27, ~19s) — **502 HTML, not JSON** |
| **A1 Enhance → local fallback** | **PASS** | Status *Done — enhanced locally (Anna photo-edit unavailable)…*; `local_enhance` / `classical_enhance` |
| Host A2 Remove BG | **SKIPPED** (brief) | Enhance host failed; incidental host `image.edit` also **502 HTML** ~13:36:43→47 before local path |
| **A2 Remove BG → local rembg** | **PASS** | Status *Done — background removed locally…*; path `local_remove_bg` / engine `rembg` |
| Host A3 Remove Object | **SKIPPED** | Enhance host failed |
| `apps push` / cut / review | **N/A** | Draft stays **rev 4**; **no push**; **no commit**; **no App Review** |

**Exit criterion (host Upload→Process→Preview→Download via `image.edit`):** **NOT MET.**

## Afternoon re-probe — 2026-10-30 ~14:13–14:15 UTC

User “try again” after morning ~13:36 still-502 (rollout may have landed later). Harness: **one** fresh `anna-app dev --llm-app-slug reliclear --slug reliclear` (`:5180` was down). `llm bridge real → app_slug=reliclear`. Fixture `fixtures/sample.png` via CDP. **Recovered: NO.**

| Check | Result | Evidence |
| --- | --- | --- |
| OS `whoami` | **PASS** | `akshay\aksha` |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~87d |
| Connect → Ready | **PASS** | Reliclear; “Ready — upload a photo to begin.” |
| `upload.inline` | **PASS** | R2 `expires_in: 1800`; key under `app:726/.../sample.png` (~14:14:33→35 UTC) |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` + `model: fal-ai/nano-banana-2` → `transport HTTP 502: <!DOCTYPE html>` CF HTML title `anna.partners \| 502: Bad gateway` (~14:14:35→50, ~15s); Ray **`a52b06f55c168884`** — **502 HTML, not JSON** |
| Host A2 Remove BG | **SKIPPED** | Enhance host failed |
| Host A3 Remove Object | **SKIPPED** | Enhance host failed |
| `apps push` / cut / review | **N/A** | Draft stays **rev 4**; **no push**; **no commit**; **no App Review** |

**Exit criterion (host Upload→Process→Preview→Download via `image.edit`):** **NOT MET.**

## Post-fix-day full smoke — 2026-11-02 ~11:37–11:39 UTC

Past scheduled Fri Oct 30 `1.1.0-beta.201` day (Oct 30 probes still 502). Harness: **one** fresh `anna-app dev --llm-app-slug reliclear --slug reliclear` (`:5180` was free). `llm bridge real → app_slug=reliclear`. Fixture `fixtures/sample.png` via CDP. **Full host A1–A3.** **Recovered: NO.**

| Check | Result | Evidence |
| --- | --- | --- |
| OS `whoami` | **PASS** | `akshay\aksha` |
| `anna-app whoami` | **PASS** | `https://anna.partners` · `aps:dev` · ~84d |
| Connect → Ready | **PASS** | Reliclear; “Ready — upload a photo to begin.” |
| `upload.inline` | **PASS** | R2 `expires_in: 1800`; key under `app:726/.../sample.png` (~11:38:01 UTC) |
| **A1 Enhance (host edit)** | **FAIL (host)** | `image.edit` + `model: fal-ai/nano-banana-2` → `transport HTTP 502: <!DOCTYPE html>` CF HTML title `anna.partners \| 502: Bad gateway` (~11:38:01→20, ~19s); Ray **`a542d9aba8021387`** |
| **A1 → local_enhance** | **PASS** | Status *Done — enhanced locally…*; path `local_enhance` / `classical_enhance` |
| **A2 Remove BG (host)** | **FAIL (host)** | same 502 HTML (~11:38:45→47); Ray **`a542dabc5c5856d2`** |
| **A2 → local rembg** | **PASS** | Status *Done — background removed locally…*; `local_remove_bg` |
| **A3 Remove Object (host)** | **FAIL (host)** | same 502 HTML (~11:39:08→15); Ray **`a542db4bb979be32`** |
| **A3 clear fail** | **PASS** | Status needs Anna photo-edit; download disabled; **no** `local_inpaint` / no fake |
| `apps push` / cut / review | **N/A** | Draft stays **rev 4**; docs/`FORUM-IMAGE-EDIT-502.md` refreshed; **no push**; **no commit**; **no App Review** |

**Exit criterion (host Upload→Process→Preview→Download via `image.edit`):** **NOT MET.**

## Notes

- Doctor key-mode warning remains Windows noise.  
- Credentials file loose-permissions warning: `chmod 600` not meaningful on NTFS; ignore or tighten ACL if desired.  
- Host Option A exit criterion (**Upload → Process → Preview → Download via `image.edit`**) is **not met** — **502 HTML not cleared** as of 2026-11-02 ~11:39 UTC (past scheduled `1.1.0-beta.201` day; Oct 30 morning+afternoon also 502). Enhance + Remove BG usable via local classical/rembg. Remove Object host-only until recovery.  
- Next human: **escalate** forum #400 / post refreshed `FORUM-IMAGE-EDIT-502.md` (Oct 30 + Nov 2 Rays); re-smoke when Anna confirms edit recovery; then `apps cut` / review — do **not** `submit-review` until host edit recovers or you accept local-only demos.  
- Option B CUDA generative still out of scope.
