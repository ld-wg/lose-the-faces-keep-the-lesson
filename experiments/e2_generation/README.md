# E2 — Generators in their original form

**Question.** What is the privacy–utility trade-off of each generator family,
compared with plain censorship? (Hypotheses (a) and (b); objectives 2–4.)

## Design

- **Arms**, all as published, without our ideas:
  - blur and mosaic: censorship, the utility floor;
  - CIAGAN;
  - GANonymization;
  - BLANKET (no identity pre-pass, native swap).
- **Fixed:** Phase 1 with SCRFD-10GF (unless E1 says otherwise); the same
  detections for every arm of a video.
- **Inputs:** `demo1`, `demo2`, `demo3`.

**Metrics** (see `experiments/README.md`):
- **coverage**, with passthrough causes;
- **privacy:** FaceNet (held-out) and ArcFace rank-1 and rank-5, verification
  at FAR 1%, cosine to the real face, Privacy Gain. Each in two views:
  anonymized faces only, and all faces with passthrough counted as a leak;
- **utility:** emotion agreement, expression error, head-pose error, gaze
  error, age error and gender agreement, re-detection. Each is read against
  the E0 noise floor;
- **temporal:** within-track consistency, flicker;
- **cost:** fps and peak GPU memory;
- **robustness:** the same metrics stratified by face size and absolute yaw.

Per video and pooled, with 95% bootstrap intervals over tracks.

## Outputs (`results/`)

| File | Content |
|---|---|
| `latex/tab-e2-main.tex` | coverage, FaceNet / ArcFace rank-1, emotion agreement, expression error, consistency (pooled) |
| `latex/tab-e2-privacy.tex`, `tab-e2-utility.tex` | the full privacy and utility metric sets |
| `latex/tab-e2-strata.tex` | privacy and utility by face size and yaw |
| `latex/fig-e2-tradeoff.pdf` (+ `.csv`) | privacy (FaceNet rank-1) vs utility (emotion agreement), one point per arm and video |
| `latex/results.tex` | `\result{e2/<video|pooled>/<arm>/<metric>}` |
| `<video>/<arm>/output.mp4` | the anonymized videos (never exported to the paper) |

## Status

Runnable (2026-10-01), not yet run on serra1.

Privacy is reported in three views (DECISIONS.md 40):
- generated faces only;
- all faces, with passthrough as a leak;
- the final video after the fail-safe.

Utility is read against the E0 noise floor, which also appears as a row of
the utility table (*real, next frame*). **Cost:** ~8 h on serra1, most of it
SDXL identities for BLANKET.

