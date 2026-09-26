# src/eval — provenance, licensing, calibration log

Evaluation tooling for the thesis contribution (contribution plan Steps 1–2,
`research/next-steps/contribution-implementation-plan.md`). Nothing here
writes a face embedding to disk; every output is a scalar or a count.

## Vendored: FaceNet InceptionResnetV1

- Source: `timesler/facenet-pytorch`, commit
  `787da06156087cd6b616fe6608213722bddc30cd`, `models/inception_resnet_v1.py`.
- License: MIT (`_vendor/LICENSE_facenet_pytorch`).
- Changes: weight-download code and the `pretrained`/`device` constructor
  arguments removed; the network and `forward()` are unchanged (see the
  file header).
- Why vendored: `facenet-pytorch` 2.6.0 (latest on PyPI, checked
  2026-09-25) pins `torch<2.3,>=2.2` and `torchvision<0.18`; this project
  runs `torch==2.6.0+cu124` on serra1.

## Weights: `weights/facenet_vggface2.pt`

- URL: `https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt`
- sha256: `281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b`
  (downloaded on serra1 2026-09-25 with `python -m src.eval.heldout --download`).
- Loads with `strict=True` after dropping the `logits.*` classifier head.
- **Flag, not resolved here:** the model is a port of David Sandberg's
  `facenet` checkpoint `20180402-114759`, trained on VGGFace2. The facenet
  code is MIT, but the terms that apply to weights trained on VGGFace2,
  and the dataset's current distribution status, were not verified. Use
  for research evaluation only until checked, the same standard as the
  CelebA flags in `phase2_generate/models/*/NOTICE.md`.

## Why two recognizers

Guidance (later steps) uses ArcFace `w600k_r50`. Measuring privacy with the
same model a generator was optimized against would overstate it, so
`evaluate.py` reports FaceNet (different architecture, data and loss) next
to ArcFace. A gap between the two scores means the push only fooled the
guidance model.

## Calibration choices

- **FaceNet alignment:** 5-point similarity onto the ArcFace template scaled
  to 160×160, 10% margin (`heldout.FACENET_MARGIN`). FaceNet was trained on
  looser MTCNN crops, and the margin is the knob. It is validated by the
  true-accept rate below, not assumed.
- **In-domain threshold:** genuine pairs are real frames of one track at least
  5 frames apart. Impostor pairs are real frames of two tracks visible in
  the same frame, so they are certainly different people; this keeps a
  person who re-enters as a new track out of the impostor set. The
  threshold is set at FAR 1%.
- **Re-detection:** SCRFD-10GF at 320×320 on each anonymized crop. The
  detection with the best IoU (≥ 0.3) against the expected box supplies the
  landmarks. Otherwise Phase 1's landmarks are used and the observation
  counts as not re-detected.

## Calibration log

### 2026-09-25: identity pre-pass validated on `video-demo-2.mov` (Step 1)

`python -m src.eval.identity_report` on Phase 1 run `demo2`: 244 frames,
24 tracks, 2928 observations, none missing landmarks, about 45 s on
serra1.

- **Pose:** yaw ranges from −42.5° to +61.8° (median 13.6°). The unit is
  degrees and the range is plausible.
- **Spread:** median cosine of a frame to the rest of its track is 0.85
  (p5 0.65). Only track 7 has outliers, 3 frames (203, 205, 207). No
  tracker ID switch was flagged.
- **Which aggregate represents the person best on unseen frames.** Each
  track was split into even and odd frames; the estimate comes from one
  half and is scored by mean cosine to the other:

  | mode | mean cos | quality-weighted cos |
  |---|---|---|
  | `first` (what BLANKET does today) | 0.679 | 0.679 |
  | `best` | 0.708 | 0.723 |
  | `mean` | 0.845 | 0.846 |
  | `quality_mean` | 0.842 | **0.848** |
  | `ema_adaptive`, α_f = 0.95 (Deep OC-SORT) | 0.763 | 0.765 |
  | `ema_adaptive`, α_f = 0.8 (best of the sweep) | 0.792 | 0.793 |

  Aggregating the whole track beats any single frame by about +0.17 and
  beats the causal EMA. This supports plan decisions D1/D2. Deep OC-SORT's
  α_f = 0.95 is tuned for long tracking sequences and stays anchored to
  the first frame on 4–242-frame tracks.
- **Quality weighting helps less than planned.** Pooled over all frames,
  the combined quality score correlates with agreement (Spearman 0.46,
  above any single term: det 0.32, size 0.32, pose 0.18). **Within a
  track the correlation is near zero** (0.05). Quality separates good
  tracks from bad ones but barely ranks frames inside one track, which is
  why `quality_mean` ≈ `mean`. The raw embedding norm is the best
  within-track predictor, and it is still weak (0.15).
- **Cross-track.** Co-occurring tracks (certainly different people) reach
  cosine 0.67 (mean 0.21, p95 0.51). Tracks 10 and 25 (frames 2–74 and
  221–243) have cosine 0.93, almost certainly one person re-entering. Any
  re-link threshold on this footage must sit above ~0.7.
