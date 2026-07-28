# Pipeline — Lose the Faces, Keep the Lesson

How we de-identify faces in classroom videos while keeping expression, gaze, and affect. This is the story of the design: what we first assumed, the decisions we faced, and the plan we landed on.

---

## Part 1 — The first assumption

We started with a naive 5-step pipeline. Each step was its own network:

```
Video → [1 Detect] → [2 Estimate expression] → [3 Generate face] → [4 Blend in] → [5 Stabilize over time] → Out
```

| Step | What it does | Network |
|------|--------------|---------|
| 1 | Find every face, track it across frames | YOLOv8n |
| 2 | Read expression / pose / gaze | MediaPipe or 3DMM |
| 3 | Generate a synthetic replacement face | GAN or diffusion |
| 4 | Blend the new face into the frame | Poisson / learned |
| 5 | Keep identity stable, no flicker | Gradient injection |

Clean and modular. But five networks means five places to fail — and some of those boundaries quietly destroy the very thing we care about.

---

## Part 2 — The decisions

Reviewing each step (and checking the latest literature) surfaced five real decisions.

### Decision 1 — Do we even need step 2 (expression estimation)?

We assumed we'd read expression/pose/gaze explicitly, then feed it to the generator. But the best generators don't need it. ReferenceNet (Kung 2025) takes a *driving image* and preserves pose/gaze/expression **best-in-table with no estimator at all** — the conditioning is implicit.

Why this matters: an explicit estimator is a *failure point*. MediaPipe/3DMM break on occlusion and on children — and when they break, GANonymization silently falls back to an "average face," wiping out the exact affect we promised to keep.

**→ Merge step 2 into the generator.** Keep the estimators only as *measuring tools* (they compute our utility metrics), not as pipeline stages. Gaze turns out to be emergent — no gaze net needed.

### Decision 2 — Do we need a separate blending step (step 4)?

We assumed generate-then-blend. But CIAGAN, LDFA, and ReferenceNet all generate the face **in place** — inpainting is built in. Classical blending (alpha/Poisson) preserves pixels but can still break *perceived* affect via skin-tone or lighting seams at the boundary.

**→ Merge step 4 into the generator.** End-to-end inpainting. Classical blending stays only as a fallback.

### Decision 3 — GAN or diffusion?

- **Diffusion wins on utility**: best pose/gaze/expression preservation, plus a single knob $d$ that sweeps the privacy–utility trade-off at inference time (free thesis figure), plus seed-locking for a consistent per-student identity.
- **Decisive for us:** GANs are trained on adult faces (FFHQ) and fail on infants. Large-scale diffusion (LAION) generalizes to children. **Our subjects are children.**
- **Cost:** diffusion is heavy (200 steps/frame). We accept this and mitigate with DDIM/distillation.

**→ Diffusion.** GAN (CIAGAN) stays as a fast baseline.

### Decision 4 — How do we keep identity stable over time (step 5)?

Our first idea — "reuse the previous frame's face" — is a feedback loop: artifacts compound and the face drifts over a long lecture. The fix is to anchor every frame to a **fixed synthetic identity per track** (a seed-locked latent) and pull toward *that*, not toward the previous frame.

**→ Keep step 5, but as a thin stage, not a new network.** Gradient injection (reusing the generator's U-Net) toward the per-track canonical latent. Fallback: BLANKET's simpler "generate the identity once, swap it into every frame."

### Decision 5 — Is our generator list even up to date?

Checking 2025–2026 work, it wasn't. Two additions matter most:
- **BLANKET** (ICDL 2025) — built specifically for **infant/child faces**, temporally consistent, open code. This directly answers our biggest risk.
- **Reverse Personalization** (WACV 2026) — Kung's own successor to ReferenceNet; training-free and lets us control age/sex/race of the surrogate (demographic similarity for free).

**→ Rebuild the candidate set.** BLANKET becomes the new primary lean.

---

## Part 3 — The final plan

Five steps became **three**. The two merges remove the two silent quality-killers (expression-estimation failure, blending seams) without training anything new.

```
Input Video (classroom lecture)
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STEP 1 — DETECT + TRACK                             │
│ YOLOv9-M (or YOLOv8n) + ByteTrack                   │
│ High recall — a missed face = a privacy leak        │
│ Out: [frame, track_id, box, confidence]            │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STEP 2 — GENERATE + BLEND  (conditioning built in) │
│ Candidates: BLANKET / Reverse Personalization /    │
│             ReferenceNet / CIAGAN                   │
│ • Implicit conditioning (driving image)            │
│ • End-to-end inpainting (no separate blender)      │
│ • Seed-lock per track → consistent identity        │
│ • Demographic-similar surrogate (age/sex/race)     │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│ STEP 3 — STABILIZE OVER TIME  (thin stage)         │
│ Anchor each frame to the track's fixed identity    │
│ (gradient injection toward canonical latent)       │
│ Fallback: generate identity once, swap per frame   │
└────────────────────────────────────────────────────┘
    │
    ▼
Output Video (de-identified, expression-preserved) + audio
```

### Why this is right

- **Privacy** — high-recall detection (no face missed) + diffusion anonymization knob + a fixed pseudonym per student.
- **Utility** — implicit conditioning preserves expression/gaze/affect best; end-to-end blending avoids seams.
- **Fairness** — BLANKET targets child faces; we evaluate stratified by age/ethnicity.
- **Feasibility** — per-frame + smoothing (video-native diffusion doesn't exist yet); DDIM/distillation for speed.
- **Modularity where it counts** — detection, generation, temporal stay separately benchmarkable.

### What we still must prove (the gating question)

**Does the generator preserve child faces?** Everything hinges on this. Plan: a fairness smoke test (Weeks 8–9) running BLANKET vs ReferenceNet vs Reverse Personalization vs CIAGAN on child faces, then a pre-registered down-select (Week 14) scored on privacy / utility / fairness / compute / temporal stability.

---

## Links

- Problem: [[10-problem]] | Method: [[30-method]] | Evaluation: [[40-evaluation]]
- Decisions: [[70-decisions]] | Open questions: [[60-open-questions]]
- Generator choice detail: [[gan-vs-diffusion]]

#pipeline #architecture #phase1/detection #phase2/generation #phase2/inpainting
