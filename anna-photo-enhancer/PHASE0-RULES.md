# Phase 0 — Verified Founding Builder Rules

**Verified:** 2026-10-27 (local sprint start)  
**Primary source:** [Official Founding Builder Program announcement](https://forum.anna.partners/t/turn-your-ai-agents-apps-into-recurring-monthly-grants-join-the-anna-ai-os-founding-builder-program-up-to-80k-month-pool/205) (forum post by @dora; original 2026-08-03; Section 7 portfolio update + Sept results noted 2026-10-04)  
**Supporting:** [BUIDL / MAU clarification #351](https://forum.anna.partners/t/founding-builder-buidl-deadline-while-app-review-is-pending-and-mau-counting-for-bundle-only-apps/351), [Developer Terms](https://anna.partners/developers/reference/developer-terms.md), [Platform Terms](https://anna.partners/legal/terms), [Revenue-share changelog](https://forum.anna.partners/t/anna-1-1-0-beta-97-beta-122-async-tool-jobs-doc-parsing-byok-app-analytics-revenue-share-is-here/212)

anna.partners homepage is a product shell; **program rules live on the forum announcement**, not a separate static “rules PDF.” Re-check the thread before optimizing for a deadline.

---

## Confirmed

### Qualification window / deadline

| Fact | Value | Status |
| --- | --- | --- |
| Qualification window | **2026-09-01 → 2026-11-30** | **Still current** as of 2026-10-04 staff posts (Sept results + portfolio update both restate Nov 30) |
| Early build/register | From 2026-07-29 | Confirmed in announcement §2 |
| After Nov 30 | Window to *join* Founding cohort closes; already-qualified Apps keep Founding status (subject to Program Rules) and ongoing monthly evaluation | FAQ in announcement |

**Do not treat Nov 30 as assumed folklore — it was re-confirmed in October 2026 staff updates.** Re-verify if Anna posts a Season 2 / window-close notice (budget warning threshold: ~$70k allocated of $80k/mo).

### Qualification requirements

To become a Founding Builder, the App must:

1. Be **approved and published** on the Anna Marketplace  
2. Provide **genuine functional utility** (no placeholders / simple clones)  
3. Reach **≥ 200 Qualified App MAU (Tier E)** in **any single calendar month** inside the window  
4. Comply with App Review, maintenance, and fair-play policies  

Pending App Review does **not** by itself invalidate a DoraHacks BUIDL submission timing (per #351); Marketplace publish + MAU during the window are still required to *qualify*.

### Qualified App MAU

- **Definition:** unique **signed-in** Anna users who complete ≥ 1 **Qualified App Run** in a calendar month (UTC month bounds: 00:00:00 → 23:59:59 last day).  
- **Qualified App Run:** initiated by a valid signed-in user; **successfully** executes the App’s **declared primary function** and returns a **meaningful, non-trivial** result (not “Done” / heartbeat).  
- **Counts:** Free credits, paid credits, and **BYOK** all count 100%.  
- **Does not count:** install/open/view only; failed/cancelled/incomplete; bots/device farms/coordinated self-runs; **developer / team / internal QA** accounts.  
- **Bundle-only Apps** can still qualify if the run is verifiable server-side (e.g. successful host LLM that produces the product result). `storage.set` alone is **not** enough (#351).

### Grant tiers (cash, USD)

Highest tier only per month (non-cumulative). September 2026 had a one-time Launch Boost; **Oct/Nov first qualification → Standard Monthly** only.

| Tier | Monthly MAU | Sept Launch Boost | Standard / mo |
| --- | --- | --- | --- |
| S Galactic Sovereign | 20,000+ | $6,000 | $5,000 |
| A Growth Titans | 10,000+ | $4,000 | $2,500 |
| B Starship Commanders | 4,000+ | $1,600 | $1,000 |
| C Orbit Pioneers | 1,000+ | $400 | $200 |
| D Stratosphere | 500+ | $200 | $100 |
| E Genesis Spark | 200+ | $80 | $50 |

### Payout mechanics (grants)

- Measurement: server-side logs aggregated per calendar month; audited at month-end.  
- **Paid within ~30 days** after the measurement month ends (cash via wire / PayPal / Stripe / agreed channel).  
- Status is permanent once earned (unless disqualified); **monthly pay pauses** if that month’s MAU / availability / maintenance fail, then resumes when requirements recover.  
- Budget: **$80k/mo** pool; grandfathering protects existing qualified Apps; not FCFS within a month.  
- **Portfolio rule (from Oct 2026 cycle):** all Apps under one Builder → one portfolio MAU (deduped users); **≤ 1 grant per Builder per month**. Multi-app does not multiply grants.  

### 70% token / usage revenue share

- **Separate** from Founding Builder cash grants (grants + revenue share can both apply; revenue share does not reduce grants).  
- Builders earn **70% of eligible usage profit** when users run Apps via Anna’s **paid credit** system.  
- Earnings: daily rollup (**T+1**); Developer Console → Earnings; payout requests from **$50** balance.  
- Sources: announcement FAQ + platform changelog beta.120–121.

### App Review requirements

Must be a **complete, independently usable App** (interactive UI + working backend) — not raw scripts, bare MCP, or demos.

Focus areas:

1. Complete experience (UI, backend, I/O, loading/errors)  
2. **AI-native core** (primary value from AI reasoning/workflows)  
3. Functional completeness / real utility  
4. Distinctiveness & reliability  
5. Clear UX  

Typical review: **3–5 business days**. Monthly **meaningful maintenance** required to stay grant-eligible.

### IP / no-exclusivity (what we can actually cite)

| Claim | Evidence | Confidence |
| --- | --- | --- |
| You must own / have rights to what you publish | [Developer Terms](https://anna.partners/developers/reference/developer-terms.md) §2 | High |
| You are solely responsible for published Apps | Developer Terms §7 | High |
| Platform owns Anna Service IP; user Inputs retained; Outputs assigned to user | [Platform Terms](https://anna.partners/legal/terms) §2.3 / §5 | High (general ToS; not App-listing-specific) |
| Founding grants ≠ employment / partnership | Program notice on announcement | High |
| **No exclusivity** requiring Anna-only distribution of Photo Enhancer | **No affirmative exclusivity clause found** in Developer Terms, program FAQ, or announcement | **Inferred / still soft** — safe product stance remains: keep desktop IP independent; Anna is a channel |

---

## Still unclear / watch

1. **Legal exclusivity** — absence of a ban ≠ a written “publish elsewhere freely” warranty; if counsel-critical, ask on the rules thread or support.  
2. **Exact “eligible usage profit” formula** for the 70% (gross vs net, BYOK carve-outs) — high-level only in public posts.  
3. **Whether host `image.edit` success is always scored as a Qualified App Run** for photo apps — rules are primary-function + server-side verification, not a fixed Host API whitelist. Design for real successful edits; don’t game installs.  
4. **Season 2 / window-close** if pool hits warning thresholds.  
5. **Local CLI host path:** docs use `anna-app login --host https://anna.partners` (beginner guide); older threads mention `nexus.anna.partners` — prefer **https://anna.partners** unless staff say otherwise.

---

## Local CLI state (2026-10-27)

| Check | Result |
| --- | --- |
| `anna-app --version` | `0.1.61` |
| `anna-app whoami` | **no accounts** — not logged in |
| `~/.config/anna/credentials.json` | **missing** |
| `anna-app doctor` | `uv` OK; uvx cache cold; nexus not local; **`dev.key` mode 666 vs expected 0600** |
| Key ACL note | On Windows, Unix `chmod 600` / `os.chmod(0o600)` still report mode `666`. Treat as **known Windows doctor noise** unless harness refuses to start; prior `--no-llm` harness already worked. |

**Interactive login required (user must run):**

```bash
anna-app login --host https://anna.partners
```

Device-flow: CLI opens browser → approve while signed into Anna → PAT saved (~90d) to `~/.config/anna/credentials.json`. Confirm with `anna-app whoami`.  
Optional: Developer Console self-serve activation (email verify + Developer ToS) if not already a developer.

Do **not** invent or commit credentials.
