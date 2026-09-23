# NOTICE — vendored third-party code

`vendor/` contains a verbatim copy of
[**dvl-tum/ciagan**](https://github.com/dvl-tum/ciagan)'s generator
architecture (Maximov, Elezi, Leal-Taixe, *CIAGAN: Conditional Identity
Anonymization Generative Adversarial Networks*, CVPR 2020), needed to load
the released checkpoint and run inference. Unlike `../yolo_facev2_s/vendor`,
this **is** imported at every inference call, not just for a one-time
conversion — see "Design note: native PyTorch, no ONNX conversion" below.

## Provenance

- **Source repo:** https://github.com/dvl-tum/ciagan
- **Pinned commit:** `38d0eac9d03d3970607a59b43f8041dfac05887d` (the repo's
  `master` HEAD at fetch time, via `GET /repos/dvl-tum/ciagan/commits`)
- **Fetch date:** 2026-09-18
- **Fetched via:** `raw.githubusercontent.com` directly (not reconstructed
  from memory or approximated) — `source/arch/arch_unet_flex.py` and
  `LICENSE` copied verbatim into `vendor/`.
- **License: MIT.** Confirmed via `GET /repos/dvl-tum/ciagan` →
  `license.spdx_id == "MIT"`, and the repo's own `LICENSE` file (copyright
  2020 Dynamic Vision and Learning Group), copied verbatim to
  `vendor/LICENSE` per the MIT terms' own requirement to include the
  notice with any copy.

## What's vendored, and why

Only `arch_unet_flex.py`'s `Generator` class is instantiated by
`backend.py`. The same file also defines `Discriminator` (training-only,
unused here) and the `ResidualBlock*`/`ConvLayer` building blocks — kept in
the file rather than split out, since it's copied verbatim.

`backend.py` does **not** vendor or import `source/process_data.py`,
`source/util_data.py`, `source/test.py`, or `source/util_func.py` — those
implement the *offline* CelebA dataset-preparation and training-time data
loading pipeline (dlib landmark extraction into a saved `clr/lndm/msk`
folder structure, then a second PIL/torchvision resize-and-crop stage read
by a `Dataset`). `backend.py` reimplements the *equivalent per-image
geometry* those two stages produce (same inter-eye-distance-based crop, same
68-point line-art/mask construction, same channel-order quirk — see its
own module docstring for the exact correspondence) directly against an
in-memory crop from this project's own pipeline, rather than round-tripping
through disk in the original two-stage shape. This is a reimplementation of
documented, MIT-licensed logic, not vendoring — verified line-by-line
against the original files at the pinned commit above before writing it.

## Design note: native PyTorch, no ONNX conversion

Unlike the Phase 1 detector backends (all ONNX + `onnxruntime`), this
backend keeps `vendor/`'s `Generator` loaded via plain PyTorch at inference
time. There's no verified ONNX exporter for its `forward(x, onehot=...)`
signature — a conditional one-hot embedding branch, `InstanceNorm2d`,
`spectral_norm`, and `ReflectionPad2d` all need checking against an actual
export before trusting one, and writing/validating that exporter is out of
scope for this round. `torch`/`dlib` are an optional extra
(`phase2-ciagan` in `pyproject.toml`), not a base dependency, matching how
Phase 1's `yolo-facev2-convert` extra is scoped.

## Checkpoint status — obtained and verified working (2026-09-22)

The official pretrained checkpoint (1200 CelebA identities, ≥30 images
each) is linked from the upstream README as a Google Drive file:
`https://drive.google.com/file/d/1j5iT-SvvbC-JRy7qvY-eEP4sLzvoh8Ut/view` —
third-party hosting, no published checksum upstream, no versioned GitHub
Release. Downloaded on serra1 and verified end to end:

- **SHA256:** `e2f296e286472c55da91bc04528e85bedf9a3babfa8e6b9b4e186c79c9013358`
  (89.8 MB) — no upstream hash exists to check this against; record of the
  copy this project actually tested against, for future drift detection.
- `Generator().load_state_dict(sd, strict=True)` — all 248 keys matched
  exactly, no `strict=False` needed.
- Real inference on real crops from `video-demo.mov` produces plausible,
  clearly-anonymized adult faces (different, recognizable synthetic
  identity per seed), composited with a visible but not degenerate seam at
  the mask boundary — a genuine first-round result, not a placeholder.
- Full run against a 239-frame track set: 1112/1112 face observations
  generated (0 passthrough), same `track_id` verified visually consistent
  across frames ~176 apart. ~2.3 fps end to end — dlib's landmark stage
  (CPU-only, see "Landmark strategy" in `backend.py`'s docstring) is the
  bottleneck, not the GAN forward pass itself.
- **GPU note:** `torch`'s default PyPI wheel bundles a CUDA 13.0 runtime
  that serra1's driver (555.42, CUDA 12.5 max) can't run — same failure
  class as the `onnxruntime-gpu` issue documented in `docs/serra1-ssh.md`.
  Fixed in `pyproject.toml` via `[tool.uv.sources]` redirecting `torch` to
  PyTorch's own cu124 wheel index on Linux, capped `<2.7` (the newest
  version cu124 publishes). Verified: `torch==2.6.0+cu124` initializes CUDA
  and runs real on-device tensor ops on serra1.

## Bug found + fixed (2026-09-22): oversized, disconnected-looking faces

The first full-video render (composited back into the source frames) looked
wrong — generated faces were oversized and floated disconnected from the
body. Root cause, confirmed by re-deriving the geometry rather than
guessing: `_face_transform()`'s crop radius (`h_r = dx*5`) bakes in CelebA's
own aligned-photo convention — how much of a *portrait* frame a head
occupies at a given inter-eye pixel distance `dx`. On this project's actual
classroom footage, the same real `dx` corresponds to a head that's simply
bigger in-frame than CelebA's convention assumes, and the formula has no way
to know that.

**First hypothesis, tested and falsified:** assumed the bug was in `run.py`
feeding too-tight a crop (Phase 1's old 32px fixed-pad debug crop). Tested
by sweeping the crop's *context padding* alone (`context_ratio` 0.4-1.3)
while keeping the transform formula untouched — the pasted-back face size
barely changed at all across that whole range (verified visually, see the
calibration images from this session). This makes sense in hindsight: `dx`
is measured directly off real pixel positions in the source frame, entirely
independent of how much surrounding canvas the crop includes — enlarging
the crop only adds untouched background around a mask region whose
*absolute* size was already fixed by `h_r`/`w_r`. Padding was never the
lever.

**Real fix:** added a `portrait_scale` multiplier directly on `h_r`/`w_r`
in `_face_transform()` (see its docstring). Swept `portrait_scale` in
{0.35, 0.5, 0.65, 0.8, 1.0} against real frames from `video-demo.mov`,
pasted each back into the full frame, and visually compared face-to-body
proportion (not isolated crops resized to a common height, which hides the
real effect). **0.5** looked closest to a natural head size relative to the
visible shoulders — smaller values started looking like undersized,
blocky patches; 1.0 (upstream's own formula, unmodified) reproduces the
original oversized bug. `models/__init__.py`'s `DEFAULT_CONTEXT_RATIO`
was also reduced from 0.75 to 0.6 in the same pass (smaller `portrait_scale`
means a smaller region is actually needed, so the crop no longer has to be
as generous) — the two values were calibrated together, not independently.

**Known remaining limitation, not addressed by this fix:** the mask
boundary is still a visibly hard edge (CIAGAN's own jaw+eyebrow polygon,
no soft blending) — real but a separate, harder problem from the size bug
just fixed. Not attempted this round.

### Follow-up (2026-09-22): Poisson blending — real but modest improvement

Replaced the hard-mask paste (`output[mask] = generated[mask]`) with
`cv2.seamlessClone` (gradient-domain compositing, `_poisson_composite()` —
see its docstring) to address the hard-edge limitation above. Verified:
deterministic (same input twice → identical output), pixels far from the
mask stay untouched, falls back to the hard-mask paste if `seamlessClone`
raises (a real risk for a mask near the crop's edge).

**Honest finding from a direct hard-mask-vs-Poisson comparison on real
frames:** the seam itself does blend more smoothly, but most of the
"psychedelic"/noisy look reported after this fix comes from the generated
content itself (visible chromatic-fringing/ripple artifacts inside the
face region), not primarily the paste boundary — Poisson blending doesn't
meaningfully fix that. This turned out to have a real, fixable cause — see
the next entry.

### Follow-up (2026-09-22): `context_ratio` also controls noise, not just framing

The user's hypothesis, tested directly rather than assumed: does a bigger
context crop itself make the "psychedelic" noise worse? Swept
`context_ratio` from 0.0 (crop = exactly the detected box) to 0.6 (the
prior default) at a fixed `portrait_scale=0.5`, composited each into the
full frame. **Confirmed** — noise visibly increases with `context_ratio`,
smoothly across the whole range tested (0.0/0.08/0.12/0.15/0.18/0.22/0.3/0.6).

This resolves an apparent contradiction with `_face_transform()`'s own
math, which derives the generated region's size (`h_r`/`w_r`) from the
detected inter-eye distance `dx` alone — in principle independent of how
much surrounding crop context there is. Empirically, though, `dx` itself
comes out differently depending on how much context surrounds the face
when dlib runs: the frontal HOG detector needs *some* margin to fire
reliably, and behaves differently again once there's a lot of extra
background relative to the face. A larger detected/implied region means
the network's fixed 128×128 output gets stretched over more physical
pixels when warped back — magnifying its own per-pixel artifacts instead
of the natural down-sampling a smaller target region provides. Net effect:
`context_ratio` and `portrait_scale` are **not independent controls**,
whatever the transform's algebra alone suggests.

**New default: `context_ratio=0.15`** (down from 0.6), `portrait_scale`
unchanged at 0.5 — visibly less noisy than 0.6 while keeping heads
reasonably (not tiny-postage-stamp) sized. Still not noise-free — this is
a real reduction, not a full fix, and likely close to this checkpoint's
practical floor for this footage. Worth investigating further (possibly
resampling-kernel artifacts from the 128-canvas round trip, or a genuine
limitation of this checkpoint's output quality on out-of-distribution
footage) before concluding the ceiling has been reached.

### Follow-up (2026-09-22): rotation correction added; mask coverage identified as a separate, deeper issue

Two more complaints from real-video viewing: generated faces "don't line
up properly," and the result only ever covers the "face proper," never
hair/forehead/ears/full head. Researched what established literature does
here before guessing (see [[70-decisions]]'s entry for full citations):

- **Rotation:** confirmed against InsightFace/ArcFace's own alignment code
  (`face_align.py`) and the FFHQ/StyleGAN dataset-prep script that a
  similarity transform (rotation + scale + translation) from eye landmarks
  is standard practice, not an edge case — resting on Umeyama (1991).
  `_face_transform()` never did this (matching upstream's own
  `process_data.py`, which also never rotates). Added a rotation term
  derived from the eye-to-eye angle, applied before the anisotropic scale;
  since the same matrix inverts for paste-back, the generated content now
  re-rotates to match the real head's tilt when composited. Also switched
  from upstream's x-only inter-eye distance to the true Euclidean
  eye-to-eye distance — the x-only version was itself only valid for a
  near-upright face, the same gap rotation correction closes. Verified:
  determinism and the outside-mask-untouched invariant both still hold.
- **Mask coverage — not fixed here, tracked as a separate, deeper issue.**
  Checked what other established face-swap/anonymization methods actually
  replace: FSGAN and SimSwap cover face **+ hair** (segmentation-based
  masks, not landmarks); **GANonymization uses full-head segmentation** (a
  dedicated U-Net, not a landmark polygon) — already on this project's own
  candidate shortlist. CIAGAN and LDFA are the outliers in the literature,
  both restricted to "face proper." This is a real limitation of CIAGAN's
  own training convention, not something crop/transform tuning can reach —
  flagged for a future candidate (GANonymization or similar) rather than
  patched here.

### Follow-up (2026-09-22): `portrait_scale` re-calibrated after the `context_ratio` fix — the two were never re-tuned together

After reducing `context_ratio` to fix noise (see above), faces looked too
small — expected but not yet corrected: `portrait_scale=0.5` was
calibrated against the *old* `context_ratio=0.6`, and the two interact
(see above) rather than independently controlling size and noise. Swept
`portrait_scale` in `{0.5, 0.65, 0.8, 1.0, 1.2}` at the new
`context_ratio=0.15`, composited into full frames: size increases smoothly
with `portrait_scale` as expected, but — importantly — **noise does not
reappear until past ~1.0-1.2** at this tighter crop, unlike the original
bug where noise was already severe at `portrait_scale=1.0` combined with
`context_ratio=0.6`. This confirms the noise was never really about
`portrait_scale` alone — it's the *combination* with a loose crop that
hurts.

**New default: `portrait_scale=1.0`** (up from 0.5) — properly sized and
still visibly cleaner than the original bug, at `context_ratio=0.15`.
Notably, 1.0 is upstream's own unmodified formula — once the crop-context
confound is controlled for, it turns out to need no correction at all for
this footage. `1.2` already showed visible noise creeping back in the
sweep — don't push past ~1.0 without re-checking.

**Also changed as part of this fix:** `run.py` no longer reads pixels from
Phase 1's saved `Face.crop_path` at all — it cuts its own crop directly
from the source video around `Face.box`, sized via `context_ratio`. This
was necessary regardless of the scale fix (Phase 1's old crop convention
was never going to match whatever portrait framing this backend actually
needs) and, as a side effect, drops the `--save-crops` requirement on the
Phase 1 run entirely.

## `shape_predictor_68_face_landmarks.dat` — dlib's 68-point model

Needed for the 68-point landmark extraction this backend's preprocessing
depends on (see `backend.py`). The dvl-tum/ciagan repo vendors a copy
directly at `source/shape_predictor_68_face_landmarks.dat`, but its license
terms were **not independently verified in this session** — dlib's own
distribution (via `dlib.net`, trained on the iBUG 300-W dataset) is the
canonical source and is recommended instead; confirm its license before
redistributing this repository publicly with the file included. Verified
working from that source: `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`,
decompressed with `bunzip2`.

## Verified facts from reading the pinned source directly (not assumed)

- Generator input: `(B, 6, 128, 128)` = `concat(landmark_line_art, face * (1 - mask))`, plus a one-hot vector `(B, num_classes)` passed as `onehot=` to `forward()`. `num_classes=1200` matches the released checkpoint's training identity count.
- Composition is in-place: `clamp(generated * mask + original * (1 - mask), 0, 1)`.
- `forward()` has no dropout or sampling — deterministic given the same input tensor and one-hot vector.
- Landmark scheme is dlib/iBUG 68-point, not InsightFace's 5-point — see `backend.py`'s docstring for how this project resolves that gap.
