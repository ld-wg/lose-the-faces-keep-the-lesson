# E5 — Cost of the final pipeline

**Question.** How much cheaper can the final pipeline get, and what does each
speed-up cost the viewer: privacy, utility and motion?

Profiling on 2026-10-20 (60 frames of demo1) set the starting point:
- The campaign ran BLANKET's main process on CPU. Its head segmentation then
  took 412 ms per face; on the GPU it takes 24 ms.
- Per face, GFPGAN takes 78 ms, the swap server's landmarking 37 ms and the
  swap itself 26 ms.
- PNG transport and the RPC are about 10 ms together.

## Design

**The base pipeline.** Every arm is the paper's final pipeline:
- BLANKET with Phase 1 detection;
- C2 `quality_mean`;
- P2 β 1.2;
- the privacy gate at the video's ArcFace FAR 1% threshold;
- fail-closed composition.

**Shared inputs.** Arms reuse E3's frozen SDXL identities
(`cache/e3_contribution/<video>/blanket`), so all of them swap the same
synthetic faces and are compared on the same faces.

| Arm | Factor | Setting |
|---|---|---|
| `ref` | — | every frame, GFPGAN on, main process on the GPU |
| `ref-cpu` | device | main process on CPU, as in the campaign (demo1 and demo2 only) |
| `enh-off` | enhancer | no GFPGAN |
| `enh-96` | enhancer | GFPGAN only on faces of at least 96 px |
| `every2`, `every3` | fixed sampling | a track's face is generated on one of every 2 (≈15 fps) or 3 (≈10 fps) frames |
| `adapt-0.05`, `adapt-0.08` | adaptive sampling | regenerate when the real face changed since the last generated frame by at least τ (106-landmark error, inter-ocular units), or after 6 frames |
| `fast` | combined | `enh-96` + `adapt-0.05` |

**Sampled frames.** When a frame is not generated, the fail-safe shows the
track's last generated face, resized into the current box. Its reuse window
is 10 frames.

**Choosing τ for the adaptive arms.** The noise floor of the expression error
between consecutive real frames is 0.032. A τ of 0.05 therefore regenerates on
real changes, not on landmark jitter.

## Metrics

**Cost, from the run manifests and the BLANKET profile.**
- Share of detector faces generated.
- ms per detector face: the frame loop only, without the pre-pass or server
  start-up.
- Stage ms: segmentation, GFPGAN, swap, landmarking.
- Projected GPU hours per hour of class, at 30 fps and 20 faces per frame.

**What the viewer sees.** These come from the evaluator's `--video-view`,
which runs on the composed `output.mp4`, so a reused face is scored as it
looks:
- privacy: FaceNet (held-out) rank-1 and verification;
- utility: expression, pose, emotion κ and gender, against the real face;
- **motion error:** between consecutive frames of a track, |change of the
  shown face's 106 landmarks − change of the real face's|. A frozen face and a
  jittering one both score high.

**Statistics.** Paired differences against `ref` on the same faces, with
track-bootstrap intervals.

## Outputs (`results/latex/`)

| File | Content |
|---|---|
| `tab-e5-cost.tex` | cost per arm: generated share, ms per face, stage ms, GPU hours per class hour |
| `tab-e5-video.tex` | privacy, utility and motion on the composed video |
| `tab-e5-paired.tex` | Δ against `ref` (FaceNet rank-1, expression, motion), interval excludes zero |
| `fig-e5-tradeoff.pdf` (+ `.csv`) | GPU hours per class hour against motion error and against FaceNet rank-1 |
| `results.tex` | `\result{e5/<video|pooled>/<arm>/<metric>}`, plus `ms-per-face`, `class-hours`, `speedup`, `stage-*-ms`, `delta-*` |

## Status

Written 2026-10-20. Needs E3's identity cache, and the bridge at 16cf02d or
later (per-call `enhance`).
