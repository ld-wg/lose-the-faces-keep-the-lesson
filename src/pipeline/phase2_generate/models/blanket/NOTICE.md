# NOTICE — external dependency, NOT vendored

Unlike `../ciagan/vendor/` and `../ganonymization/vendor/`, there is **no
`vendor/` directory here** and no line of BLANKET's own source is copied
into this repository, in any form, at any commit. This is a deliberate
compliance choice (see "Compliance flag 1" below), not an oversight.

**What this is instead**: `backend.py` in this directory talks, over a Unix
socket, to two long-lived external server processes
(`identity_server.py`/`swap_server.py`) that live in a sibling repository,
**`blanket-anonymizer-bridge`** (public, GPL-3.0,
https://github.com/ld-wg/blanket-anonymizer-bridge — see that repo's own
README/NOTICE.md for the full provenance chain). That sibling repo is the
*only* place BLANKET's own code is ever imported.

## Provenance — BLANKET

- **Source repo:** https://github.com/ctu-vras/blanket-infant-face-anonym
- **Paper:** Hadera, Čech, Purkrabek, Hoffmann, *BLANKET: Anonymizing Faces
  in Infant Video Recordings*, IEEE ICDL 2025, pp. 1–8. arXiv: 2512.15542.
- **License: GPL-3.0.**
- **Fetch date (source read for this NOTICE):** 2026-09-23/24, via `gh api`
  and `raw.githubusercontent.com` directly against the repo's `master`
  branch — not reconstructed from this project's own earlier research note,
  which turned out to be wrong on several points (see "Verified facts"
  below).
- **Pinned commit used by `blanket-anonymizer-bridge`'s own git submodule:**
  `84245d76dc7dc1cc59c5b356f1530346d422b922` (`master` HEAD at fetch time,
  2026-09-24).
- **Vendors its own copy of FaceFusion** (`external/facefusion/` inside
  BLANKET's own repo, not a git submodule — a full, unmodified copy),
  under FaceFusion's own **OpenRAIL-AS** license (Copyright (c) 2025 Henry
  Ruhs). This project depends on that vendored copy transitively, through
  BLANKET's own repo, without modifying it.

## Compliance flag 1 — GPL-3.0, process isolation (READ BEFORE MERGING)

This project's own repository has no `LICENSE` file and is already public
(merged PRs #1–5 on GitHub). Importing BLANKET's GPL-3.0 source directly
into this process (vendoring it the way `ciagan`/`ganonymization` vendor
their MIT-licensed architectures) would very likely place this entire
repository under GPL-3.0's copyleft obligation the moment it's distributed
— which already happens continuously via `git clone`/`pull`.

This backend is architected specifically to avoid that: BLANKET's code
runs as two separate, unmodified external processes, in `blanket-anonymizer-bridge`'s own two venvs, communicating with this project only through OS-level
mechanisms (a Unix socket + temp files) — never through a Python `import`.
`blanket-anonymizer-bridge` itself is licensed GPL-3.0 (it *does* import
BLANKET directly, so that's the correct license for it), kept as its own
repository specifically so that license boundary is legible and auditable,
not buried in an unversioned folder on a shared server.

**This is a widely-used engineering practice for keeping a GPL dependency's
copyleft from propagating into a differently-licensed or unlicensed
codebase, but it is not a legal certainty** — the FSF's own guidance on
process isolation depends on how tightly coupled the communication is.
**This NOTICE is not legal advice. Do not treat this architecture as a
resolved compliance question — get real sign-off before any use beyond a
personal/academic research context**, and revisit if this project's own
scope changes (e.g. commercial use, wider distribution).

## Compliance flag 2 — `inswapper_128` provenance

BLANKET's own `FaceFusionDirectAnonymizer` defaults to FaceFusion's
`inswapper_128`/`inswapper_128_fp16` face-swap model. This checkpoint has a
well-known, unresolved distribution/provenance controversy: the original
author restricted official redistribution after misuse concerns
(non-consensual deepfakes), and the training data provenance was never
publicly disclosed. Community mirrors (including FaceFusion's own hosting)
continue to redistribute it, but its licensing/distribution status remains
disputed — unlike every other checkpoint this project depends on, which has
a clean, single, attributable source. This is an architectural choice
inherited from BLANKET itself, not introduced by this backend, but flagged
in full because this project is choosing to depend on it. Needs its own
compliance/ethics review before any use beyond personal/academic research.

## Compliance flag 3 — FaceFusion's OpenRAIL-AS license

The vendored FaceFusion copy inside BLANKET's own repo (`external/facefusion/`)
is licensed under OpenRAIL-AS (a "responsible AI license" with real
use-restriction clauses, not a plain permissive license). Read its actual
clauses (`external/facefusion/LICENSE.md` in BLANKET's repo) before relying
on this backend for anything beyond the anonymization/privacy-protection
research purpose this project exists for — don't just cite the license name
in passing.

## Design notes

- **Two isolated services, not one process, and not BLANKET's own
  `VideoPipeline`/`run_video.py` CLI.** `VideoPipeline` (BLANKET's own
  top-level entry point) owns a whole-video loop with its own internal
  multi-face IoU-based tracking (`match_faces_across_frames`) — it expects
  to process an entire video end-to-end, not one already-tracked crop at a
  time, and duplicates tracking work this project's own Phase 1 already
  did. Instead, `identity_server.py`/`swap_server.py` call BLANKET's two
  real building blocks directly:
  - `blanket.anonymization.pipelines.image_pipeline.generate_synthetic_identity()`
    (the `IdentityGenerator` role) — runs once per `Identity.seed`, cached
    in `backend.py`.
  - `blanket.anonymization.methods.facefusion.FaceFusionDirectAnonymizer`
    (the `FaceSwapper` role) — instantiated once per identity image,
    calling only its `face_swapper`/`face_enhancer` processors, never
    `deep_swapper` (the one FaceFusion processor that imports `deepface`/
    TensorFlow — confirmed unused by this call path via a direct source
    read of `blanket/anonymization/methods/facefusion.py`).
- **Splitting into two venvs, not one, isn't just about the two different
  dependency worlds (torch/diffusers vs. onnxruntime/FaceFusion) — it's
  also about weight.** BLANKET's own resolved environment
  (`requirements_exact.txt`, generated via `uv pip compile pyproject.toml`)
  pulls in TensorFlow/Keras/TensorBoard (transitively via `deepface`) plus
  Flask/Gradio/pytest — none of which the `face_swapper`/`face_enhancer`
  path this project actually calls ever imports. A single shared venv would
  carry all of that dead weight; the two-venv split, with a hand-picked
  dependency subset per role (see `blanket-anonymizer-bridge`'s own
  `requirements-identity.txt`/`requirements-swap.txt`), avoids it. Real cost
  of this choice: those two hand-picked subsets are this project's own
  maintenance burden, not upstream's `pip install -e .` — they can drift if
  BLANKET changes its own imports. Mitigation: `blanket-anonymizer-bridge`'s
  own NOTICE.md documents exactly which files/imports each subset covers.
- **`detections` parameter is dead code — cannot inject this project's own
  Phase 1 detections into the per-frame swap.**
  `FaceFusionDirectAnonymizer.anonymize(image, detections)`'s `detections`
  argument is never read in the method body (verified directly against the
  real source) — per-frame face redetection always happens internally via
  FaceFusion's own `yolo_face`. Substituting this project's own detections
  there would require patching BLANKET's source, increasing GPL derivative
  exposure — deliberately not done.
- **`synthetic_face_path` IS a clean substitution point.** Both
  `VideoPipeline(identity_image_path=...)` and
  `FaceFusionDirectAnonymizer(synthetic_face_path=...)` take a plain file
  path, regardless of how that image was produced — this project's own
  context-padded crop feeds `generate_synthetic_identity()`'s own internal
  (real) YOLO+SPIGA detection, same as it already feeds `ciagan`/
  `ganonymization`.
- **Real upstream bug worked around, not patched.**
  `stable_diffusion_parameters.yaml`'s `seed: 1` is hardcoded, and
  `StableDiffusionAnonymizer.generate()` takes no per-call seed argument —
  `identity_server.py` must set `anonymizer.seed = <Identity.seed>` as a
  public attribute before calling `.generate()`, or every track in a video
  gets the same diffusion RNG regardless of this project's own per-track
  seed. This is an external attribute assignment (legitimate use of a
  public API surface), not a source modification.
- **`--refine-mask` is default-ON for blanket, unlike ciagan's opt-in.**
  BLANKET's own compositing mask (convex hull of SPIGA landmarks / the
  inswapper footprint) never covers hair or silhouette — its own
  documented weak point (2.5/5 perceived de-identification, "silhouette/
  hairline retained"). Reusing this project's already-calibrated full-head
  segmentation model (`../_segmentation.py`) for the final composite is a
  real, testable attempt to close that gap — an original contribution of
  this integration, not something BLANKET's own paper describes or tests.
  Needs real-video calibration before trusting the visual result either
  way (see the empty calibration log below).
- **`swap_server.py` points at BLANKET's own real, unmodified
  `blanket/configs/module_parameters/facefusion_parameters.yaml` — no
  override config authored by this project.** Read directly (2026-09-24,
  after an earlier draft of this plan had speculatively suggested
  overriding `iou_filter`/`face_selector_mode` without having read this
  file yet): `max_faces: 1` already caps output to a single swapped face
  per crop (the earlier speculative concern about a second face in a
  padded crop getting swapped is a non-issue), `iou_filter: true` with a
  real tuned threshold is the authors' own default and is left as-is (this
  project's own per-track caching of one `FaceFusionDirectAnonymizer`
  instance per identity makes it a meaningful, not redundant, continuity
  check across that track's own crops over time), and
  `execution_providers: [coreml, cuda, cpu]` already prioritizes GPU
  correctly — no hand-tuning needed or done.
- **Python version**: BLANKET's own `pyproject.toml`/`setup.py` both
  declare `python_requires=">=3.9"`, but its vendored FaceFusion hard-checks
  `sys.version_info < (3, 10)` at runtime (`facefusion/core.py`) — the two
  files inside BLANKET's own repo disagree with what it actually needs.
  Both external venvs in `blanket-anonymizer-bridge` must be created with
  Python ≥3.10 explicitly (serra1's system Python is 3.12, used for this).

## Verified facts from reading the pinned source directly (not assumed)

These correct this project's own earlier research note
(`research/papers/diffusion/rw-hadera-blanket.md`, which now has a
"released code differs" section, 2026-09-24) and both design explorations
that preceded a real source read:

- Checkpoint is **SDXL inpainting**
  (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) + an **SDXL refiner**
  (`stabilityai/stable-diffusion-xl-refiner-1.0`), resolution **896×896** —
  not "Realistic Vision" / SD1.5 / 512.
- **Two ControlNets simultaneously**: `xinsir/controlnet-openpose-sdxl-1.0`
  + `diffusers/controlnet-canny-sdxl-1.0`, weight 1.0 each — not Canny
  alone.
- Real prompt/negative-prompt (from
  `blanket/configs/module_parameters/stable_diffusion_parameters.yaml`) are
  long and specific — not the short "a face of a baby" this project's
  earlier research note quoted.
- Inpainting mask: convex hull of SPIGA landmarks + Gaussian blur — this
  part of the earlier research note was correct.
- BLANKET's own pipeline already color-blends its diffusion output with
  `cv2.seamlessClone` (Poisson) — the same mechanism
  `../_compositing.py::poisson_composite` already uses.
- BLANKET's own README states: "This code is experimental and not yet
  production-ready... Full reliability will be achieved once the 'missing
  detections' problem is solved."
- The YAML's `scheduler: DPMSolverMultistepScheduler` is never read (no
  Python file under `blanket/` mentions `scheduler`); the pipeline uses the
  checkpoint's own `EulerDiscreteScheduler` (epsilon prediction). Found
  2026-09-24.
- BLANKET sets FaceFusion's `face_swapper_weight` to `100` against a
  0.0–1.0 range; it clamps to an embedding mix of
  `1.35 · synthetic − 0.35 · real face in the current frame`, i.e. a fixed,
  single-frame push away from the real person. This is the baseline the
  contribution plan's P2 step replaces with a track-level push. Found
  2026-09-24; full detail, including an open question about the two terms
  living in different embedding spaces, in the bridge's NOTICE.md.

## Checkpoint status

All weights (SDXL inpainting + refiner + 2 ControlNets, SPIGA landmarks,
`inswapper_128`, `gfpgan_1.4`, `yolov11l-face`) downloaded and verified
loadable on serra1, 2026-09-24 — resolved automatically via HuggingFace
Hub cache (`~/.cache/huggingface`) inside `blanket-anonymizer-bridge`'s two
venvs, not manually pinned/hashed by this project (unlike `ciagan`/
`ganonymization`'s single local checkpoint files) since BLANKET's own code
resolves them all itself. No separate sha256 tracking done here — the
pinned BLANKET submodule commit plus the HF model IDs named in its own
`stable_diffusion_parameters.yaml`/`facefusion_parameters.yaml` are the
provenance trail instead.

## Calibration log

### 2026-09-24: seven real upstream/environment bugs found and fixed via direct integration testing, standalone smoke tests pass end-to-end

Building `blanket-anonymizer-bridge` and wiring it up surfaced seven real,
empirically-confirmed bugs (full detail and exact fixes in that repo's own
NOTICE.md, summarized here): (1) unpinned `torch`/`torchvision` resolving
CUDA-13 wheels serra1's driver can't run; (2) `transformers` transitively
pulling an unpinned, also-CUDA-13 `torchaudio` via a vision-only import
path; (3) `setuptools>=65.0.0` resolving a too-new version that dropped
`pkg_resources` entirely (SPIGA needs it); (4)
`StableDiffusionAnonymizer.__init__`'s own default `config_path` resolving
one directory level too shallow; (5) `detector_factory.py` resolving its
own config files via bare relative paths, assuming BLANKET's own repo root
as the process's working directory; (6)+(7) `enable_vae_slicing()` missing
on multiple SDXL pipeline classes in the installed `diffusers` version (the
base ControlNet-inpaint pipeline, then the refiner pipeline). None of these
required modifying BLANKET's own source — all fixed externally (explicit
arguments, class-level monkeypatches, or matching an implicit CWD
assumption), consistent with this backend's whole process-isolation
design. After these fixes, both standalone smoke tests passed for real:
`identity_server.py` produced a genuine SDXL-generated identity image from
a real crop in ~215s, `swap_server.py` produced a genuine FaceFusion-swapped
image in ~46s (first call, including one-time FaceFusion model setup).

**Real timing finding, own bug #8**: this project's own `Backend`'s
default `server_timeout` (180s) was too short for a **cold** first
`generate()` call inside a fresh `run.py` process — full SDXL+2
ControlNets+refiner pipeline component loading (not just inference) plus
generation plus refinement exceeded 180s on serra1's shared/contended GPU
(confirmed via a real client-side `TimeoutError` at ~43% through the base
diffusion steps, after the pipeline had already spent >100s just loading
components). Raised the default to 900s in both `backend.py` and `run.py`'s
`--blanket-server-timeout`.

**Own bug #9, found by the real end-to-end run**: the refiner stage
crashes with a shape-broadcast `ValueError` when `output_size` uses the
crop's own odd/small dimensions instead of BLANKET's native 896×896 (see
`blanket-anonymizer-bridge`'s NOTICE.md for the full writeup and fix —
generate at 896×896, resize back down ourselves).

**Real finding, not a bug**: one identity's SDXL-generated face had an
extreme down/side head angle (ControlNet faithfully reproducing the
original crop's own pose) that FaceFusion's own detector couldn't find —
`swap_server.py` now treats this as a legitimate per-identity skip (see
that repo's NOTICE.md). Interestingly, on the successful full run below,
this *same* seed/track (5, seed 1383523105) generated a usable face
instead — SDXL generation is not perfectly reproducible run-to-run even
with a fixed seed, likely due to CUDA's own non-determinism under
`enable_sequential_cpu_offload()`/attention-slicing on a shared, contended
GPU. Worth remembering: a fixed seed guarantees *an attempt at* the same
identity, not a bit-identical one across separate runs.

### 2026-09-24: first real end-to-end run, `video-demo.mov`, frames 0–19, 5 tracks

`python -m src.pipeline.phase2_generate.run --model blanket --phase1-dir
<demo-gpu Phase1 output> --out runs/phase2-blanket-smoke --limit 20`
(`--refine-mask` default-on), then `compose_video.py`. Real result, not
`--random-init`:

- **49 "ok", 36 "skipped_no_landmarks"** (out of 85 (frame, track)
  observations in this window) — 0 `skipped_no_identity`/`skipped_no_frame`.
- Of 5 tracks: **3 fully or mostly generated** (tracks 1, 3, 5 — 18, 18,
  18 total observations, track 3 split 13 generated/5 passthrough before
  its first successful identity generation, tracks 1 and 5 fully
  generated after their first frame succeeded), **2 tracks 100%
  passthrough** (tracks 2 and 4 — 13 and 18 observations respectively,
  never got a usable face/landmarks at all in this window — a real
  detection-coverage gap on this footage, not investigated further this
  round; may be small/distant/angled faces, consistent with BLANKET's own
  known sensitivity to non-frontal/non-infant faces documented above).
- **Qualitative spot-check** (pulled a generated crop + its original via
  `scp`, viewed directly): the synthetic face is clearly a different
  person from the original — the core anonymization function works — but
  visibly skews toward a much younger/infant-like face than the original
  subject (expected, given BLANKET's own infant-only training/tuning,
  already flagged above) and shows a visible seam/blending artifact at
  the hair/silhouette boundary despite `--refine-mask`'s full-head
  segmentation compositing being active. Whether `--refine-mask` measurably
  helps here (vs. off) has NOT been A/B tested yet — flagged as the
  natural next calibration step, same discipline as `ganonymization`'s own
  `context_ratio`/confidence sweeps.
- `context_ratio=0.8` (borrowed from `ganonymization`, see
  `models/__init__.py`) has NOT been independently swept for `blanket` —
  still an unvalidated starting point, not a calibrated value.
- Full run took several minutes per unique identity (~2–3 min for
  generation+refinement, tens of seconds for the first swap per identity,
  fast thereafter) on a GPU shared with another user's job — real-video
  full-length runs will need this factored into time/compute planning,
  not treated as free.

### 2026-09-24: investigated the two 100%-passthrough tracks — not a box-generation bug

Regenerated the exact `crop_box(context_ratio=0.8)` crops `run.py` itself
produced for tracks 2 and 4 (the two with zero successful generations) and
viewed them directly — same framing/centering as the tracks that DID
succeed, no off-by-one or mis-centering found in `crop_box()`/
`_context_crop()`. Both crops are genuinely poor source material: track 2
is heavily pixelated/mosaic-like (likely low native resolution at that
distance from camera, upscaled), track 4 is significantly motion-blurred.
This matches the already-documented pattern from `ganonymization`'s own
calibration (detector strictness varies a lot with real-world image
quality, not just framing) — **not a bug to fix in this project's own
crop code**, a real detector-robustness limit of BLANKET's own YOLO+SPIGA
stack on this specific footage's harder faces. Re-sweeping
`context_ratio` might still be worth trying independently (not yet done),
but the visual evidence here doesn't point to framing as the cause.

### 2026-09-26: seed candidates + Phase 1 boxes (contribution Step 3) — coverage barely moves, the bottleneck moves

Run: `--identity-prepass --blanket-identity-cache runs/blanket-identities-demo2`,
3 attempts per track. Each attempt is one SDXL generation from one of the
track's best-quality crops, with Phase 1's box instead of BLANKET's YOLO,
followed by `check_identity`. Took about 2 h 20 min on serra1 with the GPUs
shared.

**Result: 1153/2928 anonymized (39.4%), vs 1119 (38.2%) for the baseline.**
13 of 24 tracks got a usable identity. The other 11 failed all 3 attempts
with `identity_unusable`, never `identity_no_face`.

| cause | baseline (2026-09-24) | this run |
|---|---|---|
| identity never generated (BLANKET's YOLO saw no face) | 598 | 0 |
| identity generated but FaceFusion finds no face in it | 943 | 1484 |
| per-frame swap misses (`swap_no_face` / `swap_iou_rejected`) | 268 | 289 / 2 |

The Phase 1 box removes the "never generated" cause completely. Those
tracks now reach SDXL, and their identities are unusable instead. Three
hypotheses for why were tested on the failed images themselves, without
any new SDXL run:

- **Resolution — disconfirmed.** Identities are resized back to crop size
  (sometimes ~110 px), so this was the first suspect. But upscaling the
  failed images ×2 or ×4 does not make FaceFusion detect a face at its 0.5
  threshold. One failed image is 645×406, and usable ones are as small as
  153×136.
- **Head pose / quality ranking — disconfirmed.** Pitch and yaw
  (buffalo_l 3D-68) at the candidate frames do not separate the groups.
  Usable tracks 1–3 have candidates at pitch −19° to −24° (heads down);
  unusable tracks 6, 10, 11, 14, 17 and 18 are near 0°. Candidate quality
  does not separate them either: track 7 fails at 0.66, track 9 works at
  0.17.
- **Detector confidence — the lead to follow.** At face_detector_score
  0.3 instead of 0.5, about half of the failed images yield one face. The
  generated faces seem to be atypical enough to score low with FaceFusion's
  `yolo_face`. Visual inspection of two failed images shows faces covered
  by a hand or bowed over a desk, reproduced faithfully by the openpose and
  canny ControlNets. The planned `--blanket-swap-face-detector-score`
  sweep is the next step. Failures are cached and their images dropped
  from the cache directory, so that sweep needs a fresh cache (≈2 h of
  SDXL) or a re-check over the bridge's `output/identities/`.

**GPU contention finding.** Another user's training job holds 20 of 24 GB on
both of serra1's GPUs. The first two attempts at this run went out of
memory with SDXL, FaceFusion and this process's head-segmentation model on
one GPU; the swap server alone takes about 3.1 GB. The working
configuration is `--blanket-identity-gpu 1 --blanket-swap-gpu 0 --ctx-id -1`
(pre-pass and segmentation on CPU).

### 2026-09-24: `video-demo-2.mov` full run completes — 1119/2928 (38%), failure modes attributed

Third attempt (after the seamlessClone and IoU-filter fixes below)
completed the full 244-frame, 24-track video with no crash: **1119 "ok" /
1809 passthrough out of 2928 face-observations (38.2%)**. For comparison on
the identical Phase 1 output: `ciagan` 2928/2928 (its dlib fallback treats
the whole crop as the face ROI, so it never skips), `ganonymization`
1226/2928 after its own two calibration rounds. Only 3/24 tracks fully
generated, 5 partial, 16 never generated.

The 1809 losses decompose exactly into three causes (per-track counts from
`run_manifest.json` cross-checked against the server log):

| Cause | Tracks | Observations lost |
|---|---|---|
| Identity generated, but the generated image had no face FaceFusion could detect (track cached as unusable) | 5, mostly long (242, 242, 239, 199, 21 frames) | 943 (52%) |
| Identity never generated — BLANKET's own YOLO/SPIGA found no face in *any* of the track's crops, even though Phase 1's SCRFD did | 11, mostly short | 598 (33%) |
| Per-frame misses inside otherwise-usable tracks (FaceFusion's own per-frame redetection, IoU rejections, frames before the first successful identity) | 5 | 268 (15%) |

SDXL ran 13 times (13 tracks got an identity attempt); 5 of those were
unusable. Non-determinism across runs was observed again: the same seed
(track 3) was unusable in the previous attempt and usable in this one.

**What this points to** (recorded as the prerequisite step of the
contribution plan, `research/next-steps/contribution-implementation-plan.md`):
the two largest causes are both "the identity was built from whichever
single crop happened to come first." Choosing the seed crop by an
aggregated per-track quality score, regenerating from the next-best crop
instead of permanently caching a failure, and reusing Phase 1's own
detection box inside `identity_server.py` (its detection step is this
project's own wrapper code, not BLANKET's source, so this needs no upstream
patch) target 52% + 33% of the losses directly. Output video:
`runs/phase2-blanket-demo2-output.mp4` (local, gitignored).

### 2026-09-24: `video-demo-2.mov` full run, second crash — IoU-filter rejection uncaught

After the seamlessClone fix (below), the same full run got much further
(past frame 100) before hitting a second, different crash:
`FaceFusionDirectAnonymizer.anonymize()`'s own `iou_filter` rejected every
detected face for one identity against its `previous_bboxes`, raising
`RuntimeError("IoU filter rejected all faces - use previous frame")` — a
legitimate outcome in BLANKET's own real pipeline (their own
`VideoPipeline.run()` catches this and reuses the last frame), but
`swap_server.py` only caught "No faces detected" before this fix. See
`blanket-anonymizer-bridge`'s own NOTICE.md — now caught as a transient
(uncached) per-frame passthrough. Re-running again with this fix.

### 2026-09-24: `video-demo-2.mov` full run found and fixed a real seamlessClone edge-case crash

Running the full 244-frame, 24-track `video-demo-2.mov` (denser scene,
more people near frame edges than `video-demo.mov`) crashed the whole
pipeline partway through with a hard `cv2.error` inside
`cv2.seamlessClone` — its own ROI-in-bounds requirement fails when a
face's own convex-hull mask sits close enough to the crop's edge (itself
close to the source frame's edge, since `crop_box()` clamps there) that
the mask's extent pokes past the crop. See `blanket-anonymizer-bridge`'s
own NOTICE.md for the full writeup — fixed by catching `cv2.error` and
falling back to a plain hard-mask paste, the same fallback pattern this
project's own `models/_compositing.py::poisson_composite()` already uses.
Re-running the full video with this fix.

### 2026-09-24: dropped "baby face" from the default prompt

See `blanket-anonymizer-bridge`'s own NOTICE.md for the full writeup —
BLANKET's own real prompt says "a baby face" explicitly, which is why
generated identities skewed infant-like in the first run regardless of
subject age (the checkpoint itself is a generic SDXL release, not
infant-trained — this is a prompt-text bias, not a model one). Default
prompt now age-neutral; BLANKET's original wording is still directly
reachable via `identity_server.py --prompt` for comparison.
