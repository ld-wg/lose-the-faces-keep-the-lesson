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
  at FAR 1%, cosine to the real face, Privacy Gain. Each in three views:
  anonymized faces only, all faces with passthrough counted as a leak, and the
  final video after the fail-safe;
- **utility:** emotion (Cohen's κ), expression error, head-pose error, age
  error and gender agreement, re-detection (gaze is a limitation). Each is read against
  the E0 noise floor;
- **temporal:** within-track consistency, flicker;
- **cost:** fps (peak memory is not measured);
- **robustness:** FaceNet rank-1 stratified by face size, absolute yaw and
  detection confidence.

Per video and pooled, with 95% bootstrap intervals over tracks.

## Outputs (`results/`)

| File | Content |
|---|---|
| `latex/tab-e2-main.tex` | coverage, FaceNet rank-1 in three views, Privacy Gain and verification on the final video (pooled) |
| `latex/tab-e2-privacy.tex`, `tab-e2-utility.tex` | the full privacy and utility metric sets |
| `latex/tab-e2-strata-face-px.tex`, `-abs-yaw.tex`, `-det-conf.tex` | FaceNet rank-1 on generated faces by face size, \|yaw\| and detection confidence |
| `latex/tab-e2-cost.tex` | frames per second per arm |
| `latex/fig-e2-tradeoff.pdf` (+ `.csv`) | privacy (FaceNet rank-1, final video) vs emotion κ, one point per arm, pooled |
| `latex/results.tex` | `\result{e2/<video|pooled>/<arm>/<metric>}` |
| `<video>/<arm>/output.mp4` | the anonymized videos (never exported to the paper) |

## Status

Run on serra1 2026-10-06 (three restarts on the shared machine, finished with
`--resume`); exported to `paper/generated/e2_generation/`. BLANKET's main
process ran on CPU, so its fps here is pessimistic (E5, DECISIONS.md 47).

Privacy is reported in three views (DECISIONS.md 40):
- generated faces only;
- all faces, with passthrough as a leak;
- the final video after the fail-safe.

Utility is read against the E0 noise floor, which also appears as a row of
the utility table (*real, next frame*). **Cost:** ~8 h on serra1, most of it
SDXL identities for BLANKET.

