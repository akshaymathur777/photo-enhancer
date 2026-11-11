# Phase 1 — Production Proof Prep

**Date:** 2026-10-28 (prior proof day 2026-10-27)  
**Branch:** `feature/anna-app-slim`  
**Goal:** Prove Option A against the real Anna host: **Upload → Process → Preview → Download**  
**Out of scope this sprint:** Option B expansion, CUDA, generative desktop port

Related rules: [PHASE0-RULES.md](./PHASE0-RULES.md)  
Latest evidence: [fixtures/TEST-REPORT.md](./fixtures/TEST-REPORT.md)

---

## Current engineering status (verified 2026-11-02 ~11:39 UTC; host edit still 502)

| Item | Status |
| --- | --- |
| Option A App present (`app.json`, `manifest.json`, `bundle/`, `executas/photo-enhancer/`) | Yes — listing name **Reliclear**, slug **`reliclear`** |
| `anna-app validate --strict` | **PASS** (CLI 0.1.61; prior — docs-only this session, not re-run) |
| `npm test` / `test:executa` / local classical | **PASS** (prior full matrix; docs-only this session) |
| Manifest `host_api.image` | `["generate", "edit", "resign"]` |
| Manifest `host_api.upload` | `["inline", "negotiate", "confirm"]` |
| Manifest `host_api.tools` | includes `bundled:photo-enhancer` + `bundled:photo-enhancer-local` |
| Offline harness (`anna-app dev --no-llm`) UI matrix | **PASS** (prior) — live UI covered Connect/Ready this session |
| Live harness | **UP** `http://localhost:5180/` — `llm bridge real → app_slug=reliclear` (~11:37 UTC 2026-11-02) |
| `anna-app whoami` | **LOGGED IN** — `https://anna.partners`, scopes `aps:dev`, ~84d |
| Developer handle | **`@akshay`** (Developer Console activated) |
| Nexus draft app | **`reliclear` id=726** [DRAFT] **working rev 4** |
| Live `upload.inline` | **PASS** — exercised ~11:38 UTC 2026-11-02 (`app:726`) |
| Live `image.generate` (control) | **PASS** — `openai/gpt-image-2` (last ~20:15 UTC 2026-10-28) |
| Live `llm.catalog({serviceType:'image_edit'})` | **PASS** earlier — 2 models incl. `fal-ai/nano-banana-2` |
| Live `image.edit` | **FAIL** — host **HTTP 502** CF HTML (**not recovered**; A1–A3 ~11:38–11:39 UTC 2026-11-02; Rays `a542d9aba8021387` / `a542dabc5c5856d2` / `a542db4bb979be32`) |
| Recovered? | **NO** |
| App code pin + 502 UX + local Enhance/Remove BG | **Done** — rembg `u2netp` lite in local executa env |
| Enhance path with host down | **PASS via fallback** — `local_enhance` (~11:38 UTC 2026-11-02) |
| Remove BG path with host down | **PASS via fallback** — `local_remove_bg` / rembg (~11:38 UTC 2026-11-02) |
| Remove Object with host down | **Clear fail** — needs host edit or paint mask; **no fake inpaint** (~11:39 UTC 2026-11-02) |
| Listing assets | Logo + harness screenshot + Console upload docs under `listing/` — Console upload pending |
| `anna-app apps push` | Last **PASS** → draft **rev 4**; **not** re-pushed this session; **not** cut / **not** submit-review |
| Exit criterion Upload→Process→Preview→Download (host edit) | **NOT MET** — host `image.edit` still 502 HTML |
| Exit criterion (Enhance/Remove BG usable while host down) | **MET via local** — see TEST-REPORT |

**Root cause (best evidence):** Anna platform **`image.edit` upstream / gateway failure**, not app code, grants, or missing BYOK. Same session: `upload.inline` OK; only `image.edit` returns Cloudflare HTML 502. See [502 diagnosis](#502-diagnosis-2026-10-27). App falls back to `local_enhance` / `local_remove_bg`.

---

## Why login was required (completed)

Live Option A uses host-mediated `upload.*` + `image.edit`. That needs:

1. A Developer PAT from device-flow login — **done** (`RTBF-3788`)  
2. Developer access + handle — **done** (`@akshay`)  
3. Per-app / per-user **grants** (separate from manifest ACL)

### Two-layer gates

| Layer | What | Photo Enhancer |
| --- | --- | --- |
| Manifest ACL | `ui.host_api.image` includes `edit`; `upload` includes `inline`/`negotiate`/`confirm` | Declared |
| User grants | `image_grant.enabled` **and** `image_grant.allowEdit`; `upload_grant.enabled` | Harness issued upload + edit without `APP_NOT_GRANTED`. CLI grants endpoint returned no data for DRAFT. If grant errors appear later: Install app → Permissions modal → enable upload + image allowEdit |

Docs: [host-api-image](https://anna.partners/developers/reference/host-api-image.md), [App UI Host API](https://anna.partners/developers/apps/app-ui-host-api.md). Login host: **`https://anna.partners`**.

Returned image URLs are ~30‑min R2 presigns — use preview/download promptly or `image.resign` / durable upload refs.

---

## Day 2 checklist

### 0. Prerequisites (browser)

- [x] Account at [https://anna.partners](https://anna.partners)  
- [x] Developer access activated (Developer Console — verify email + accept Developer ToS + handle)  
- [ ] Optional: join Builder Discord [https://discord.gg/7ftTdsdMMz](https://discord.gg/7ftTdsdMMz)

### 1. Login

```bash
anna-app login --host https://anna.partners
```

- [x] Device flow approved (code **RTBF-3788**)  
- [x] PAT at `~/.config/anna/credentials.json`  
- [x] `anna-app whoami` shows current host  

Logout later: `anna-app logout --host https://anna.partners`

### 2. Optional doctor noise (Windows)

```bash
anna-app doctor
```

Key mode 666 vs 0600 — note and continue on Windows NTFS.

### 3. Real harness (LLM / host APIs on)

```bash
cd anna-photo-enhancer
anna-app validate --strict
anna-app account set-handle <handle>   # once, if required
anna-app dev
```

- [x] Started without `--no-llm`; dashboard `http://localhost:5180/`  
- First register may require developer handle (seen this session)

### 4. Enable grants

When Process fails with `APP_NOT_GRANTED` / permission errors:

- [ ] Enable **upload** (`upload_grant.enabled`)  
- [ ] Enable **image** + **allowEdit** (`image_grant.allowEdit = true`)  
- [ ] Confirm with `anna-app apps grants anna-photo-enhancer` after install/publish  

**This session:** not the blocker — upload succeeded; edit reached host and failed with **502**.

### 5. Test matrix — live (Option A)

| # | Mode | Input | Expect | Status |
| --- | --- | --- | --- | --- |
| A1 | Enhance | `fixtures/sample.png` | Upload; `image.edit` preview; download | **Host FAIL 502 HTML** (~11:38 2026-11-02; Ray `a542d9aba8021387`; local PASS) |
| A2 | Remove BG | same | Meaningful cutout | **Host FAIL 502 HTML** (~11:38 2026-11-02; Ray `a542dabc5c5856d2`; local rembg PASS) |
| A3 | Remove object | same + “the watermark” | Object cleaned | **Host FAIL 502 HTML** (~11:39 2026-11-02; Ray `a542db4bb979be32`; clear-fail PASS) |
| A4 | Large file | ≥4–8 MB | Negotiate path | **NOT RUN** (blocked on edit 502) |
| A5 | Expired URL | stale URL | Clear error | **NOT RUN** |
| A6 | Grant off | disable allowEdit | Clear grant error | **NOT RUN** |

### 6. Capture evidence

- [x] Offline thorough results in TEST-REPORT  
- [x] Live (not `--no-llm`) results appended — upload OK, image.edit 502  
- [x] R2 notes: upload returned `expires_in:1800` presign  
- [ ] Token/CU for successful edit — N/A until host recovers  

### Exit criterion

A real photograph successfully completes:

**Upload → Process (Enhance or Remove BG or Remove Object) → Preview → Download** inside the Anna harness with host APIs.

**Status: PARTIAL — exit criterion NOT MET (recovered NO).** Login + upload proven; host **`image.edit` still HTTP 502 CF HTML** after full A1–A3 smoke ~11:38–11:39 UTC **2026-11-02** (Rays `a542d9aba8021387` / `a542dabc5c5856d2` / `a542db4bb979be32`; also Oct 30 Ray `a52b06f55c168884`) — **past** scheduled `1.1.0-beta.201` day (`image.generate` last OK ~20:15 2026-10-28). **Enhance** + **Remove BG** complete via local fallbacks. Remove Object: clear unavailable message. Draft **rev 4**. Escalate forum (`FORUM-IMAGE-EDIT-502.md`); re-smoke host A1–A3 when image edit recovers; then consider `apps push` / `apps cut` prep — **do not submit App Review** until then (or explicit local-only accept).

### Differential diagnosis attempt (2026-11-11 cloud agent)

| Item | Result |
| --- | --- |
| CLI | **0.1.64** |
| Manifest `host_capabilities` | Added; `anna-app validate` PASS — dispatcher **0.33.0** warns these + `ui.host_api.*` are **retired/ignored** (hygiene only; **not** CF 502 ACL) |
| Live C1/C2/E0–E3/A1 | **NOT RUN** — cloud agent auth skipped (device approve + `ANNA_APP_PAT`) |
| Apps push | **NOT RUN** (same) |
| Decision | **No App Review.** Keep local Enhance/Remove BG. Re-run matrix locally via `fixtures/run-diag-cdp.mjs` when logged in; forum-bump #466 only with fresh Rays + contrast. Do not change `bundle/app.js` edit shape without a successful probe. |

---

## App changes (2026-10-27 retry)

Shipped whether or not edit recovers (edit still 502 after retry):

| Change | Where |
| --- | --- |
| Pin `model: "fal-ai/nano-banana-2"` on `anna.image.edit` | `bundle/app.js` (`EDIT_MODEL`) |
| Same model on reverse-RPC `image/edit` params | `executas/photo-enhancer/photo_enhancer_plugin.py` |
| Friendly status for 502 / Bad Gateway / `APP_PROVIDER_ERROR` / Cloudflare HTML | `formatHostError()` in `bundle/app.js` — “Platform image edit is temporarily unavailable (upload succeeded). Please try again later.” |
| Tests assert pin + UX mapping | `tests/prompts.test.mjs`, `tests/run_executa_suite.py` |
| `validate --strict` + `npm test` | **PASS** after changes |

### Live edit retry (~12:55 UTC, same harness)

| Call | Result |
| --- | --- |
| `upload.inline` | **PASS** (`expires_in: 1800`) |
| `image.edit` default (no model) | **FAIL** `transport` HTTP 502 CF HTML |
| `image.edit` + `model: "fal-ai/nano-banana-2"` | **FAIL** `transport` HTTP 502 CF HTML |
| Full A1–A3 UI smoke | **Skipped** — edit still down |

---

## Local enhance fallback (2026-10-27)

Shipped to keep Enhance usable during platform outage (not a CUDA/Option B expansion — reuses existing `photo-enhancer-local`):

| Change | Where |
| --- | --- |
| On `image.edit` 502 / Bad Gateway / `APP_PROVIDER_ERROR`, Enhance → `tools.invoke` `local_enhance` | `bundle/app.js` |
| PNG re-encode from local `File` (`fileToPngBase64`) | `bundle/app.js` |
| After preview from `data:image/png;base64,…`; status “Host edit unavailable — used local classical enhance.” | UI |
| Remove BG: optional `local_remove_bg` if rembg present; else clear platform message (no fake success) | `bundle/app.js` |
| Remove object: platform-unavailable only (no mask → no fake inpaint) | `bundle/app.js` |
| Process disabled while running; re-enabled after | `bundle/app.js` |
| Hero tagline clarified | `bundle/index.html` |
| Forum draft for user to post | `fixtures/FORUM-IMAGE-EDIT-502.md` |

### CDP verify (same live harness :5180)

| Check | Result |
| --- | --- |
| Enhance + sample.png | **PASS** — After data URL; status local classical enhance |
| Remove BG (rembg absent, earlier) | **PASS** — clear platform-unavailable; no HTML dump |
| Remove BG (rembg installed, ~14:45) | **PASS** — After data URL; status local background removal |
| `validate --strict` + `npm test` | **PASS** |
| `apps push` draft | **PASS** — rev 1 |

### Remove BG rembg install (2026-10-27 ~14:38)

| Step | Result |
| --- | --- |
| `uv add "rembg[cpu]" --optional rembg` | **Blocked** — harness locks `Scripts/tool-dev-photo-enhancer-local.exe` (Access denied) |
| `uv pip install "rembg[cpu]>=2.0"` into `.venv` | **PASS** — rembg 2.0.86 + onnxruntime |
| lite model `u2netp` first download | **PASS** — ~4.6MB to `~/.rembg` |
| Harness `local_remove_bg` | **PASS** after canvas re-encode of File (raw truncated PNG fixtures fail PIL) |

---

## 502 diagnosis (2026-10-27 ~12:28–12:36 UTC)

### 1. Captured `image.edit` error (harness RPC)

| Field | Value |
| --- | --- |
| RPC | `→ req image.edit` then `← res err transport HTTP 502` |
| Latency | ~1.3–2.2 s (too fast for CF idle timeout; origin rejected/crashed) |
| Body | Cloudflare HTML page titled `anna.partners \| 502: Bad gateway` |
| CF triage | Browser **Working** · Cloudflare Dublin **Working** · Host **anna.partners Error** |
| Example Ray IDs | `a511b305fd47be46` (12:28:14 UTC), `a511beafe91f8a58` (nano-banana `model=`), `a511bebcafc98a58` (nano-banana `modelPreferences`) |
| Request shape (valid) | `{ prompt, image_url: <fresh R2 download_url>, n: 1 }` — after successful `upload.inline` |
| Not seen | `APP_NOT_GRANTED`, `APP_INVALID_REQUEST`, JSON `APP_PROVIDER_ERROR` |

Example Enhance req (truncated): `image.edit` with enhance prompt + R2 `image_url` + `n:1` → transport HTTP 502 HTML (not JSON).

### 2. Health / docs / forum

| Check | Result |
| --- | --- |
| `GET https://anna.partners/api/v1/health` | **200** `{"status":"healthy","service":"matrix-nexus","redis":"connected"}` while edits 502 |
| Docs [host-api-image](https://anna.partners/developers/reference/host-api-image.md) | Provider/upstream failures can surface as `APP_PROVIDER_ERROR` with HTTP 502 underneath; raw CF HTML means gateway never returned the facade JSON |
| Forum [staging image 502 #183](https://forum.anna.partners/t/intermittent-http-502-error-when-generating-an-avatar-image-on-anna-staging/183) | Same class: CF HTML 502 on image Host API; host origin unavailable |
| Forum [image.edit provider bug #400](https://forum.anna.partners/t/image-edit-fails-with-provider-errors-readerror-and-fal-ai-image-to-image-404/400) | Anna confirmed **GPT Image 2 edit route** pointed at a bad fal path (`/image-to-image` 404 / `ReadError`). Fix in host `1.1.0-beta.201`, rollout **2026-10-30**. Official interim: `model: "fal-ai/nano-banana-2"` |
| Forum [text 502 / image OK #114](https://forum.anna.partners/t/text-sampling-502s-on-anna-partners-platform-side-text-model-outage-not-an-app-bug/114) | Pattern for isolating modality-specific upstream outages (we have the inverse: generate OK / edit 502) |

### 3. Model preference / Specialized Models / BYOK

| Question | Finding |
| --- | --- |
| Need user Specialized Models setting? | **Not required for auth.** `llm.catalog` shows platform default `openai/gpt-image-2` (`isUserPreferred: false` for both models). Generate succeeded on that default. |
| Need BYOK? | **No.** Host owns provider keys for App `image.*`; generate worked without BYOK. |
| Would switching preference unblock? | **Unlikely today.** Live tries with `model: "fal-ai/nano-banana-2"` and `modelPreferences.hints` also returned CF HTML 502 (Rays above). After Oct 30 GPT-Image-2 edit fix (or when nano-banana edit path recovers), preference/hint may matter again. |

### 4–5. Controls from app runtime (same harness session)

| Call | Result |
| --- | --- |
| `anna.llm.catalog({ serviceType: 'image_edit' })` | **OK** — models: `openai/gpt-image-2` (platform default), `fal-ai/nano-banana-2` |
| `anna.image.generate({ prompt, n: 1 })` | **OK** — `model: openai/gpt-image-2`, 1 R2 image URL |
| `anna.upload.inline` | **OK** |
| `anna.image.edit` (default) | **502** CF HTML |
| `anna.image.edit` + `model: "fal-ai/nano-banana-2"` | **502** CF HTML (workaround from #400 did **not** unblock in this window) |

Direct REST with PAT (not app session): `POST /api/v1/copilot/app/image/edit` without `image_url` → **422** validation (route reachable); generate without app token → **401** `APP_NOT_GRANTED` / invalid audience. So generic nexus is up; failure is specific to authenticated edit execution path.

### 6. Root cause hypothesis + exact next fix

**Best evidence root cause:** Platform-side **`image.edit` provider/gateway path is down or crashing** on `anna.partners` (Cloudflare sees invalid/unavailable origin response). Auth, grants, upload, catalog, and **`image.generate` are healthy**. Consistent with known GPT Image 2 edit routing bug (#400) **plus** a broader edit-path failure today (nano-banana workaround also 502).

| Fix locus | Action | Unblocks Process today? |
| --- | --- | --- |
| **Platform (primary)** | Escalate to Anna with Rays + timestamps + generate-OK/edit-502 contrast; wait for edit gateway recovery and/or host `1.1.0-beta.201` (2026-10-30) | Only when host recovers |
| **App code (secondary)** | Pin `fal-ai/nano-banana-2` on edit + clear 502 UX + Enhance→`local_enhance` fallback | **Done** — Enhance usable today; host edit still 502 |
| **Anna dashboard** | Grants / Specialized Models / BYOK | **No** — not the blocker; grants already allow edit |

**Can we unblock Process today?** **Enhance: yes** (local classical). **Remove BG: yes** (rembg `u2netp` in local executa). **Remove Object via host: no** until platform recovers (UI has no mask → no local fake; clear message only). Draft pushed (rev 1). **2026-10-27 ~21:16 retry:** 502 **not** cleared. **Next human:** install/test draft in Developer Console; re-smoke host A1–A3 when edit recovers; optionally `apps push` for Remove Object copy + post `fixtures/FORUM-IMAGE-EDIT-502.md`; do not `submit-review` yet.

---

## After proof (not Day 2)

Only then: polish, `apps push` / review submission, growth for Qualified MAU. Do not expand CUDA/Option B until live Option A is proven or users demand local processing.
