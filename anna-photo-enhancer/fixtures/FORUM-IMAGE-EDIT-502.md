# Forum reply — POSTED

**Posted:** 2026-11-05 ~02:00 UTC by @Akshay on [#466/2](https://forum.anna.partners/t/image-edit-returns-http-502-cloudflare-bad-gateway-for-every-call-from-nookflip/466/2).  
**Staff reply:** none yet (as of 2026-11-05 ~15:16 UTC).  
Optional cross-link from [#400](https://forum.anna.partners/t/image-edit-fails-with-provider-errors-readerror-and-fal-ai-image-to-image-404/400) (different bug: JSON `APP_PROVIDER_ERROR` / fal 404 — fixed in `1.1.0-beta.201` per [#482](https://forum.anna.partners/t/anna-1-1-0-beta-201-beta-210-anna-ai-cli-0-1-62-0-1-63-browser-permissions-for-apps-llm-thinking-control-10x-async-job-quota-fullscreen-windows-gpt-image-2-edit-fixed/482); does **not** clear CF 502).

---

**Title (if new topic):** `image.edit` still CF 502 after Oct 30 `1.1.0-beta.201` (nano-banana-2)

**Body:**

Hi — same class as [#466](https://forum.anna.partners/t/image-edit-returns-http-502-cloudflare-bad-gateway-for-every-call-from-nookflip/466) (Nookflip). Distinct from [#400](https://forum.anna.partners/t/image-edit-fails-with-provider-errors-readerror-and-fal-ai-image-to-image-404/400)’s JSON fal 404/`ReadError` (catalog fix in [#482](https://forum.anna.partners/t/anna-1-1-0-beta-201-beta-210-anna-ai-cli-0-1-62-0-1-63-browser-permissions-for-apps-llm-thinking-control-10x-async-job-quota-fullscreen-windows-gpt-image-2-edit-fixed/482)).

**Generate OK / edit 502.** Same App sessions: `upload.inline` + R2 URL OK; earlier `anna.image.generate` OK. `anna.image.edit` with pinned `fal-ai/nano-banana-2` still returns **Cloudflare 502 HTML** (Host Error / `transport HTTP 502`), not JSON — including **after** Fri **2026-10-30** `1.1.0-beta.201`, **Mon 2026-11-02**, and again **Wed 2026-11-04**. Hunter’s interim nano-banana pin does **not** succeed here.

App: Marketplace draft **Reliclear** `reliclear` id=726 (rev 4), `@akshay`. Harness: `anna-app dev --llm-app-slug reliclear` → `llm bridge real`.

**Rays (UTC):**
- **2026-10-27:** `a511b305fd47be46`, `a511beafe91f8a58`, `a511bebcafc98a58`
- **2026-10-30** (~14:14, ~15 s): `a52b06f55c168884`
- **2026-11-02** (~11:38–11:39): Enhance `a542d9aba8021387` · Remove BG `a542dabc5c5856d2` · Remove Object `a542db4bb979be32`
- **2026-11-02** (~15:16): Enhance `a54419a2ae6f1c61` (CF stamp `15:16:34 UTC`)
- **2026-11-04** (~17:06): Enhance `a5553541fed85338` (CF stamp `17:06:16 UTC`)
- **2026-11-05** (~15:16): Enhance `a55cd1cb89076a60` (CF stamp `15:16:27 UTC`) — still 502 after forum post

**Ask:** Status of `image.edit` on `anna.partners` past the Oct 30 fix? Happy to retest when edit stops returning CF HTML. Thanks!
