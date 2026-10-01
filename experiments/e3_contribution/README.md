# E3 — With and without our ideas

**Question.** Do the thesis contributions improve privacy without costing
utility?
- **C2** estimates each person's real identity from the whole track.
- **P2** pushes the swapped face away from that identity in the swap
  embedding.

## Design

Four parts, all on `demo1`, `demo2` and `demo3` with SCRFD-10GF detections.
Arms within a part share faces and, for BLANKET, the same frozen SDXL
identities, so they differ only in the factor under test and are compared
pairwise.

| Part | Question | Arms | Metrics |
|---|---|---|---|
| **E3a** · C2 | Does the whole-track estimate represent the person better than one frame? | aggregation modes `first`, `best`, `mean`, `quality_mean`, `ema_adaptive` (α_f sweep) | cosine to the unseen half of the track (plain and quality-weighted); convergence vs frames seen; Spearman quality × agreement (pooled, within track); cross-track floor |
| **E3b** · seed candidates and Phase 1 detection | Do quality-ranked seed crops, and swapping the face Phase 1 found (plus the lenient identity detector), fix BLANKET's coverage? | BLANKET original; with candidates; with candidates and `detection = phase1` (DECISIONS.md 36) | coverage by cause; privacy and utility as controls |
| **E3c** · P2 strength | How far does the push cut re-identification, and at what utility cost? | `native`, `none`, `track` with β ∈ {0.2, 0.35, 0.5, 0.8, 1.2, 1.6} | E2's metric set, plus the transfer gap (ArcFace − FaceNet rank-1) |
| **E3d** · P2 target | Does the aggregated identity make a better push target than one frame? | at β 1.2: targets `quality_mean`, `mean`, `best`, `first`, `ema_adaptive` | as E3c |

GANonymization is not included: it has no identity input, so the ideas do not
apply to it.

## Outputs (`results/`)

| File | Content |
|---|---|
| `latex/tab-e3-aggregation.tex` | E3a: cosine to unseen frames per mode, per video and pooled |
| `latex/tab-e3-coverage.tex` | E3b: coverage by cause, original vs candidates vs Phase 1 detection |
| `latex/tab-e3-p2-strength.tex` | E3c: privacy and utility per arm, with paired intervals vs `native` |
| `latex/tab-e3-p2-target.tex` | E3d |
| `latex/fig-e3-p2-curve.pdf` (+ `.csv`) | FaceNet and ArcFace rank-1 vs β, with expression and emotion as guards |
| `latex/fig-e3-convergence.pdf` (+ `.csv`) | E3a convergence |
| `latex/results.tex` | `\result{e3c/<video|pooled>/<arm>/<metric>}`, … |

## Status

Runnable (2026-10-01), not yet run on serra1. Parts:
- **e3a:** C2;
- **e3b:** coverage;
- **e3c, e3d:** P2 strength and target;
- **e3e:** the fail-closed policies (hide low-confidence detections, privacy
  gate);
- **e3f:** P3, CIAGAN's identity-code push;
- **e3g:** P1, SDXL guidance. Run it only after the bridge's
  `tools/p1_spike.py` says go.

`E3_PARTS` selects a subset.

**Validation:** the exploratory FaceNet rank-1 (demo2 0.597 / 0.284, demo3
0.754 / 0.306 for `native` / β 1.2) is reproduced only with the pre-#11 Phase 1
(`--no-fill --conf 0.5`) and `detection = upstream`. The paper's runs use the
current pipeline, so their numbers differ by design.

