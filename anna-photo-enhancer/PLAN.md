# Photo Enhancer → Reliclear (Anna) → Founding Builder

## Execution Plan v2

### 1. Objective

Turn the existing open-source **Photo Enhancer** project into a polished, reviewable **Anna Marketplace App** listed as **Reliclear** (slug `reliclear`, id=726), while keeping the original desktop product, repository and IP independent.

The project has two objectives:

### Product objective

Ship a genuinely useful Anna App that can acquire real users and qualify for the Founding Builder Program.

### Career objective

Use the project to demonstrate end-to-end AI product engineering:

* AI/vision systems
* local inference
* AI application architecture
* production deployment
* model/inference tradeoffs
* GPU/performance optimization
* real-user feedback
* measurable product impact

The goal is therefore **not simply to obtain an Anna grant**. The goal is to turn Photo Enhancer into a credible public AI engineering case study.

---

# 2. Success Criteria

Success should be measured in stages.

| Level          | Outcome                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| Minimum        | Anna App successfully published                                         |
| Good           | 200+ Qualified MAU                                                      |
| Strong         | 1,000+ Qualified MAU                                                    |
| Excellent      | 4,000+ Qualified MAU + meaningful retention                             |
| Career success | Public technical case study + measurable usage + strong interview story |

Grant qualification rules and dates must be confirmed against the current Official Program Rules before optimizing for MAU.

Do **not** build features purely to manufacture MAU.

The primary product KPI is:

> **Successful photo transformations per active user.**

Secondary metrics:

* activation rate
* successful processing rate
* processing latency
* feature usage
* repeat usage
* retention
* errors/failures

---

# 3. Guiding Principles

### Principle 1 — Ship before perfect

The first objective is:

**working → validated → reviewed → published**

not:

**feature-complete → heavily optimized → eventually published**

### Principle 2 — Anna is a distribution channel

Photo Enhancer remains the core product.

Anna is an additional runtime/distribution channel.

The desktop version remains independent.

### Principle 3 — Do not rewrite the desktop application

Reuse existing logic where practical.

Keep Anna-specific code isolated under:

`anna-photo-enhancer/`

### Principle 4 — Option A first

The first release should use Anna's platform/runtime capabilities wherever practical.

Do not block launch on:

* CUDA packaging
* SDXL
* PowerPaint
* YOLO/SAM
* multi-GB model bundles
* offline binary packaging
* complex local inference

### Principle 5 — Build technical depth after validation

Once real users exist, use the project to investigate:

* model quality
* inference latency
* GPU utilization
* memory usage
* model selection
* agent/tool reliability

---

# 4. Architecture

## Repository

```text
photo-enhancer/
│
├── existing desktop application
│
└── anna-photo-enhancer/
    ├── README.md
    ├── PLAN.md
    ├── app.json
    ├── manifest.json
    │
    ├── bundle/
    │
    ├── executas/
    │   └── photo-enhancer/
    │
    ├── fixtures/
    │   └── TEST-REPORT.md
    │
    └── tests/
```

Git branch:

`feature/anna-app-slim`

The existing `main` branch remains the desktop product.

---

# 5. V1 Product Scope

The first Anna release should be deliberately small.

## Core workflow

```text
Open Photo Enhancer
        ↓
Upload image
        ↓
Choose operation
        ↓
Process
        ↓
Before / After
        ↓
Download
```

## V1 operations

### 1. Enhance

Improve the appearance of the image.

### 2. Remove background

Automatically remove the background.

### 3. Remove object

Remove an unwanted object where supported.

### 4. Optional AI assistant

The assistant can help users discover and invoke the appropriate operation.

The assistant is **not** the primary product.

The product is the image transformation.

---

# 6. Option A — Launch Architecture

Option A is the **only mandatory implementation for V1**.

### App UI

Responsible for:

* upload
* operation selection
* processing state
* before/after comparison
* errors
* download

### Host APIs

Use Anna's supported image/upload capabilities.

Avoid putting API keys or unnecessary credentials in the client bundle.

### Executa

Expose a small set of useful tools:

```text
enhance_photo
remove_background
remove_object
list_modes
ping
```

The assistant should direct users toward the App UI when a local upload is required.

If an image URL is already available, the appropriate tool can operate on it.

---

# 7. Phase 0 — Verify Before Engineering

### Time: 1–2 hours

Before spending significant development time:

* confirm current Founding Builder rules
* confirm Qualified MAU definition
* confirm current qualification window
* confirm grant tiers
* confirm payout mechanics
* confirm 70% token revenue-share mechanics
* confirm current App Review requirements
* confirm IP/no-exclusivity terms

This is especially important because the earlier plan referenced a historical November 30 qualification window.

Do not assume that date remains current.

**Phase 0 status (2026-10-27):** Verified against the official forum Program Rules. Written up in [`PHASE0-RULES.md`](./PHASE0-RULES.md). **Nov 30 2026 window is still current** (restated in Oct 2026 staff updates). Local CLI is **not logged in** — user must run `anna-app login --host https://anna.partners`.

### Account setup

* Activate Anna developer account
* Activate developer profile
* Join Builder Discord
* Review Developer Hub
* Review Build on Anna 101
* Review Build on Anna 102

---

# 8. Phase 1 — Production Proof

### Goal

Prove that the existing Anna implementation works against the real Anna environment.

Current engineering status already shows:

* scaffold completed
* Option A UI completed
* Option A Executa completed
* validation passing
* tests passing

But live host upload/image editing has not yet been tested.

Therefore the immediate priority is **not more code**.

It is production integration.

**Phase 1 prep (2026-10-27 / re-probed 2026-10-28 ~19:02 UTC):** Option A present; `anna-app validate --strict` PASS. Live smoke: login + upload OK; host `image.edit` still **502** (not cleared). App pins nano-banana-2, clear 502 UX; **Enhance → `local_enhance`** and **Remove BG → `local_remove_bg` (rembg/u2netp)**; Remove Object clear host/mask message only. Working draft pushed (now **Reliclear** / slug `reliclear`, id=726). Next wait ~**2026-10-30** platform fix. Checklist: [`PHASE1-PRODUCTION-PROOF.md`](./PHASE1-PRODUCTION-PROOF.md).

### Tasks

1. `anna-app login`
2. Configure required grants
3. Run `anna-app dev`
4. Test real image upload
5. Test Enhance
6. Test Remove Background
7. Test Remove Object
8. Measure token usage
9. Test real R2/presigned URLs
10. Fix production-only issues

### Exit criterion

A real photograph can successfully go:

**Upload → Process → Preview → Download**

inside Anna.

---

# 9. Phase 2 — Product Polish

Once production functionality works, improve the user experience.

### UI

Keep it extremely simple.

Recommended first screen:

> **Enhance your photo with AI**

Then:

**Upload Photo**

After upload:

```text
Enhance
Remove Background
Remove Object
```

After processing:

```text
Before       After
```

Then:

**Download**

### Error handling

Explicitly handle:

* unsupported image
* oversized image
* corrupt image
* processing failure
* timeout
* missing permission/grant
* unavailable backend
* expired URL

### Performance

Measure:

* upload latency
* processing latency
* total time-to-result

Target:

> **User gets a useful result quickly without needing to understand the underlying AI system.**

---

# 10. Phase 3 — Validation & Review

Before submitting:

### Automated

```text
anna-app validate --strict
npm test
npm run test:executa
```

### Manual

Test:

* JPG
* PNG
* WEBP
* portrait
* landscape
* small image
* large image
* failed operation
* repeated operations

### Production

Test the actual uploaded draft.

Do not assume that:

> local harness = production

The existing plan correctly identifies this as a risk.

### Listing

Prepare:

* logo
* screenshots
* concise description
* feature list
* privacy explanation
* appropriate category
* clear positioning

Then:

```text
apps push
     ↓
Draft install
     ↓
Final smoke test
     ↓
App Review
```

---

# 11. Phase 4 — Launch

Once approved:

### Launch priorities

1. Verify public installation
2. Verify core workflow
3. Monitor failures
4. Monitor usage
5. Collect feedback
6. Fix critical issues quickly

Do not immediately add five new features.

The first question is:

> **Do people actually use it?**

---

# 12. Phase 5 — Product Analytics

After launch, establish a simple funnel:

```text
App opened
    ↓
Upload started
    ↓
Upload completed
    ↓
Transformation started
    ↓
Transformation succeeded
    ↓
Result viewed
    ↓
Image downloaded
    ↓
User returns
```

Track whatever Anna's platform makes available and is appropriate under its privacy model.

### Primary metrics

**Activation**

% of users who successfully upload an image.

**Completion**

% of uploaded images successfully processed.

**Success**

% of processed images downloaded/accepted.

**Retention**

Users who return and perform another transformation.

**Qualified MAU**

Track separately for the Founding Builder Program.

---

# 13. Phase 6 — User-Driven Iteration

Only after launch should we decide whether Option B is necessary.

## Option B becomes conditional

Build it only if users demonstrate a need for:

* local processing
* offline capability
* additional image operations
* lower latency
* functionality unavailable through Anna host APIs

Potential Option B capabilities:

* PIL enhancement
* OpenCV inpainting
* optional rembg
* local CPU processing

The existing plan already identifies these as the classical local path.

### Important

Do **not** spend significant time packaging Option B before users validate the need.

---

# 14. Generative AI / CUDA — Deferred

The existing desktop application's heavier stack remains outside V1.

Deferred:

* SDXL
* PowerPaint
* YOLO/SAM
* CUDA-heavy pipelines
* multi-GB model packaging
* complex local binary distribution

The desktop application remains the place for the full generative stack until there is a compelling Anna packaging/runtime story.

This avoids turning the Anna launch into an infrastructure project.

---

# 15. Technical Experiment Track

After launch, the project becomes a technical laboratory.

Run controlled experiments.

## Experiment A — Image quality

Compare models/settings on a fixed evaluation set.

Measure:

* perceptual quality
* artifact rate
* user preference
* latency

## Experiment B — Inference performance

Measure:

* image resolution
* VRAM
* latency
* throughput

## Experiment C — Model tradeoffs

Example:

```text
Model A
Quality: 8.1
Latency: 3.2s
VRAM: 5.8GB

Model B
Quality: 8.6
Latency: 5.7s
VRAM: 8.4GB
```

Then make an explicit engineering decision.

## Experiment D — Agent reliability

Measure:

* correct tool selection
* invalid tool calls
* recovery from errors
* successful completion

This gives the project genuine **AI systems engineering depth**.

---

# 16. Career Track

This should run in parallel with development.

The project should ultimately produce four portfolio assets.

## Asset 1 — GitHub

Clean repository with:

* architecture
* setup
* tests
* screenshots
* benchmarks
* design decisions

## Asset 2 — Technical case study

Title:

> **From Open-Source Desktop App to Production AI Application: Building Photo Enhancer on Anna**

Cover:

* architecture
* constraints
* integration
* tradeoffs
* failures
* performance
* user feedback
* results

## Asset 3 — Performance study

Example:

> **Optimizing AI Image Enhancement on a Consumer GPU**

Cover:

* VRAM
* latency
* model selection
* resolution
* throughput

This connects directly to the ML infrastructure/inference direction.

## Asset 4 — Product impact

Eventually:

> **Photo Enhancer reached X MAU and processed Y images**

Only publish numbers that are actually measured.

---

# 17. Career Narrative

The project should reinforce the following story:

```text
ML Engineer II
      ↓
ML Platform
      ↓
GPU / Kubernetes / Model Infrastructure
      ↓
Photo Enhancer
      ↓
Computer Vision
      ↓
Local AI / Ollama
      ↓
Production AI Application
      ↓
Real Users + Measurements
      ↓
Inference / AI Systems Engineering
      ↓
London Frontier-AI Roles
```

This is much more valuable than presenting Photo Enhancer as simply a hobby application.

---

# 18. Weekly Time Budget

Do not allow this project to consume the main career-study plan.

Target:

**6–8 hours/week**

### Before launch

| Activity       |   Time |
| -------------- | -----: |
| Anna/platform  |   1.5h |
| Implementation |     3h |
| Testing        |   1.5h |
| Documentation  |     1h |
| **Total**      | **7h** |

### After launch

| Activity             |   Time |
| -------------------- | -----: |
| Product improvements |     2h |
| Experiments          |     2h |
| User/usage analysis  |     1h |
| Documentation        |     1h |
| Career portfolio     |     1h |
| **Total**            | **7h** |

---

# 19. Immediate 7-Day Sprint

Do **not** work through the entire six-week roadmap immediately.

Your next sprint is only this:

### Day 1

* Activate Anna developer account
* Join Discord
* Verify current program rules
* Verify grant deadline/window

### Day 2

* `anna-app login`
* Configure grants
* Run production environment

### Day 3

Test:

* Enhance
* Remove Background
* Remove Object

with real images.

### Day 4

Fix production issues.

Focus on:

* R2 URLs
* permissions
* CSP
* image upload
* tool arguments
* errors

### Day 5

Polish the UI.

### Day 6

Run full test suite.

### Day 7

Prepare:

* screenshots
* app description
* README
* listing metadata

### End of Sprint Goal

**A complete Anna MVP that works against the real production environment.**

Nothing more.

---

# 20. Decision Gates

At the end of every phase, make a decision.

### Gate 1

**Does Option A work?**

No → fix it.

Yes → continue.

### Gate 2

**Can users complete the core workflow?**

No → improve UX.

Yes → submit.

### Gate 3

**Does Anna approve the App?**

No → address review feedback.

Yes → launch.

### Gate 4

**Are people actually using it?**

No → investigate product/positioning.

Yes → optimize.

### Gate 5

**Do users need local Option B?**

No → don't build it.

Yes → build it.

### Gate 6

**Is there meaningful traction?**

Yes → invest more heavily.

No → keep it lightweight and move your time back toward the main career roadmap.

---

# 21. Final Architecture Decision

The final intended architecture is:

```text
                 PHOTO ENHANCER
                       │
              ┌────────┴────────┐
              │                 │
          Desktop             Anna
          Product             Edition
              │                 │
       Full AI stack       Slim V1
       CUDA / models       Host APIs
       GPU inference       Anna Runtime
              │                 │
              └────────┬────────┘
                       │
                Shared Product
                 / UX / Brand
```

Anna does **not** replace the desktop product.

The desktop application remains the technically complete product.

Anna becomes the **distribution, experimentation and user-traction channel**.

---

# 22. Final Priority Order

If time becomes limited, work in exactly this order:

**1. Verify current program rules**

**2. Get production Anna integration working**

**3. Ship Option A**

**4. Pass App Review**

**5. Get real users**

**6. Measure usage**

**7. Improve the highest-impact problems**

**8. Run technical experiments**

**9. Add Option B only if justified**

**10. Produce the technical case study**

Do **not** reverse this order.

The biggest mistake would be spending weeks building sophisticated local/CUDA functionality before discovering whether anyone actually wants the Anna version.

---

## Bottom line

The existing engineering work is already in a surprisingly good state: the scaffold, Option A UI/Executa, Option B prototype, validation and test suites are documented as completed.

So the next milestone is **not "build Photo Enhancer."**

It is:

> **Get the existing Option A implementation working against Anna's real production environment and get it submitted.**

Once that's done, the project can evolve based on actual users rather than assumptions.
