# Decision: GAN vs Diffusion (Primary Generator)

The pivotal Phase 2 decision. Gates all downstream work.

## Contenders

| | **GANonymization** (GAN) | **ReferenceNet** (Diffusion) |
|---|---|---|
| Paper | [[rw-hellmann-ganonymization]] | [[rw-kung-referencenet]] |
| Carrier | Dense landmarks | Latent + ReferenceNet branches |
| Emotion preservation | **Strong** (expression-first) | Good (coherent pose/gaze/expression) |
| Identity control | Attack-threshold anonymization | **Single knob**, tunable |
| Fairness | Not analyzed | **Drops for underrepresented groups** |
| Temporal | Image-only | Image-only |
| Complexity | Landmark dependency | Heavy (SD2.1 + ReferenceNet) |

## Decision criteria

1. **Emotion preservation** (utility) — pedagogical affect analysis is the whole point
2. **Identity suppression** (privacy) — meet attack thresholds
3. **Fairness** — stratified performance across demographics
4. **Compute** — runnable on standard hardware
5. **Temporal extensibility** — can we bolt on frame-to-frame consistency?

## Current leaning (updated 2026-07-28 after agent verification)

**Leaning diffusion (ReferenceNet)** — but the child-face fairness question is now the gating concern.

Verified findings:
- **ReferenceNet** ([[rw-kung-referencenet]]): best-in-table pose/gaze/expression preservation at d=1.2; competitive re-ID at d=1.4; the knob $d$ = ready-made inference-time privacy–utility curve; **seed diversity → consistent per-student pseudonym** (helps temporal). BUT: fairness failures specifically on **infants & ethnic minorities** — critical since our subjects are children. Code public (`hanweikung/face_anon_simple`).
- **GANonymization** ([[rw-hellmann-ganonymization]]): beats DeepPrivacy2 on most emotions but only **comparable to CIAGAN** (CIAGAN better on Fear/Happy/Sadness); anonymizes *less* aggressively (0.71 vs 0.81/0.93 embedding distance); detection failure → "average face" fallback loses expression; CelebA bias (preserves makeup/lipstick).
- **CIAGAN** emerges as a strong GAN baseline — treat as equal to GANonymization, not lesser.

**Gating question:** does ReferenceNet's child-face degradation confirm on our data? If yes → fine-tune on child faces or fall back to GAN.

## Plan

1. ✅ Verify both papers (done — see individual notes)
2. Prototype ReferenceNet (`hanweikung/face_anon_simple`) on a small classroom clip; sweep $d \in \{1.0, 1.2, 1.4, 1.6\}$; fix seed per track ID
3. Build privacy–utility curve (re-ID vs expression/gaze retention vs $d$) → thesis figure
4. Head-to-head vs GANonymization + CIAGAN on same frames
5. Stratify by age/ethnicity; decide, log in [[70-decisions]]

## Links

- Families: [[gan]], [[diffusion]]
- Evaluation: [[40-evaluation]]
- Open questions: [[60-open-questions]]

#phase2/generation #decision
