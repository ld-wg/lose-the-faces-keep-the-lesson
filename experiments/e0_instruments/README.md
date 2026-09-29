# E0 — Instrument validation

**Question.** Do the recognizers and the utility probes measure what the paper
claims they measure, on this footage?

Every other experiment reads its privacy numbers through two face recognizers
and its utility numbers through landmark, pose, emotion and attribute probes.
E0 checks those instruments on real faces before any anonymized face is
measured with them. Its tables go to the paper's methodology appendix.

## Design

For each input video, Phase 1 detections (SCRFD-10GF, the same settings as
E2/E3), then three measurements:

1. **Recognizer validity.** For FaceNet (held-out evaluator) and ArcFace
   (guidance space) there are two kinds of pairs:
   - *genuine:* real frames of one track, at least 5 frames apart;
   - *impostor:* real frames of two tracks visible in the same frame, so they
     are certainly different people.

   The threshold is set at FAR 1%. Reported: threshold, TAR (true-accept rate)
   at that threshold, genuine and impostor means, pair counts. A recognizer
   with low TAR here cannot support a privacy claim.
2. **Swap-stage mechanism** (once, on the first video's real crops):
   - FaceFusion's recognizer vs buffalo_l `w600k_r50` on identical aligned
     crops (are the P2 push target and the swap in the same space?);
   - how far inswapper's `emap` is from orthogonal (does BLANKET's native mix
     happen in one space?).

   The crops are deleted as soon as the check ends.
3. **Utility noise floors** — *pending, needs the utility probes.* The same
   metric between consecutive real frames of a track, for emotion, expression,
   pose, gaze and attributes. It gives what the probe fluctuates by itself;
   E2/E3 drift is read against it.

## Outputs (`results/`)

| File | Content |
|---|---|
| `recognizers.md` / `latex/tab-e0-recognizers.tex` | threshold, TAR @ FAR 1%, genuine / impostor means, per video and recognizer |
| `mechanism.md` / `latex/tab-e0-mechanism.tex` | recognizer equivalence and `emap` orthogonality |
| `latex/fig-e0-scores.pdf` (+ `.csv`) | genuine vs impostor similarity distributions, per recognizer, with the thresholds |
| `latex/results.tex` | `\result{e0/<video>/<recognizer>.tar}`, `…threshold`, `e0/mechanism/…` |

## Run

```bash
uv run experiments/e0_instruments/run.py                 # videos from config.toml
uv run experiments/e0_instruments/run.py --video demo2   # one video
```

The mechanism check runs in the BLANKET bridge's swap environment
(`$BLANKET_REPO/.venv-swap`). Without it, that step is skipped and marked as
such in `provenance.json`. It takes about 5 min per video for detection and
calibration.

## Already measured (exploratory, 2026-09)

- **TAR @ FAR 1%:** FaceNet 0.69 (demo2) and 0.73 (demo3); ArcFace 0.80 and
  0.74.
- **Recognizer equivalence:** cosine 1.0000 on 300 crops.
- **`emap`:** ‖EᵀE − I‖/√512 = 51.9, and cos(e, e·E) ≈ 0.00.
