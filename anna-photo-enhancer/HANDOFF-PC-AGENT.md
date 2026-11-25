# PC agent handoff — image.edit CF 502 diagnosis

**Branch:** `cursor/edit-502-diagnosis-c99a`  
**PR:** https://github.com/akshaymathur777/photo-enhancer/pull/5  
**Plan:** `/opt/cursor/artifacts/plans/edit_502_diagnosis_029b98e2.plan.md` (or repo copy if present) — do **not** edit the plan file.

## Why PC

Cloud VM cannot complete Anna login (Turnstile + skipped device/PAT). Live Host API matrix must run on a machine already logged into `https://anna.partners`.

## Already done on this branch

- CLI notes → 0.1.64
- `manifest.json` top-level `host_capabilities` (validate OK; dispatcher 0.33.0 **ignores** these + `ui.host_api.*`)
- `fixtures/diag-edit-502.js` — C1/C2/E0–E3/A1 probes
- `fixtures/run-diag-cdp.mjs` — CDP runner against harness `:5180`
- `STATUS.md` §6b + `PHASE1-PRODUCTION-PROOF.md` differential notes
- Unit tests pass with CLI under `~/.local`

## Your job (complete the plan)

1. `git fetch && git checkout cursor/edit-502-diagnosis-c99a && git pull`
2. `anna-app whoami` — must be logged into anna.partners; if not: `anna-app login --host https://anna.partners`
3. Mark/finish todos: manifest already validated → `anna-app apps push` (draft only) if clean
4. One harness: `anna-app dev --llm-app-slug reliclear --slug reliclear`
5. Run matrix: `node fixtures/run-diag-cdp.mjs` → `.anna/diag-edit-502-result.json`
6. Update `STATUS.md` + `PHASE1-PRODUCTION-PROOF.md` with the matrix table
7. Decision rules:
   - Any edit shape OK → switch `bundle/app.js` to that shape; re-smoke A1–A3
   - Only generate+reference OK → optional interim; forum bump with contrast
   - All edit CF 502 → forum #466 bump with fresh Rays; **do not App Review**
8. Commit + push; update PR #5

## Paste prompt for PC Cursor agent

```
Continue Differential diagnosis: image.edit CF 502 on branch cursor/edit-502-diagnosis-c99a (PR #5).

Cloud agent already added host_capabilities, fixtures/diag-edit-502.js, fixtures/run-diag-cdp.mjs, and STATUS/PHASE1 notes. Live matrix was NOT run (no Anna auth in cloud).

Do NOT edit the plan file. Complete remaining work:
1) anna-app whoami / login if needed
2) anna-app validate; apps push draft if clean
3) anna-app dev --llm-app-slug reliclear --slug reliclear
4) node fixtures/run-diag-cdp.mjs (C1/C2/E0-E3/A1)
5) Update STATUS.md + PHASE1-PRODUCTION-PROOF.md; adopt working edit shape in bundle/app.js if any; else forum-bump path — no App Review until host edit works
6) Commit and push

See anna-photo-enhancer/HANDOFF-PC-AGENT.md
```
