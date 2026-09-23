# NOTICE — vendored third-party code

`vendor/` contains a verbatim copy of
[**hcmlab/GANonymization**](https://github.com/hcmlab/GANonymization)'s
pix2pix generator architecture (Hellmann, Mertes, Benouis, Hustinx, Hsieh,
Conati, Krawitz, André, *GANonymization: A GAN-Based Face Anonymization
Framework for Preserving Emotional Expressions*, ACM Trans. Multimedia
Comput. Commun. Appl. 21(1), Article 6, Dec 2024/Jan 2025.
DOI: [10.1145/3641107](https://doi.org/10.1145/3641107),
arXiv: [2305.02143](https://arxiv.org/abs/2305.02143)), needed to load the
released checkpoint and run inference. A second vendored model,
`HeadSegmentationModel` (used for this backend's compositing mask — see
"Design note" below), lives in the **shared**
`../_vendor/head_segmentation_model.py`, not here, because
`../ciagan/backend.py` opt-in-reuses it too (`--refine-mask`, see
`../ciagan/NOTICE.md`).

## Provenance — GANonymization

- **Source repo:** https://github.com/hcmlab/GANonymization
- **Pinned commit:** `7dbe45e3b172670ff59be982c662749165ab5ced` (the repo's
  `main` HEAD at fetch time, via `GET /repos/hcmlab/GANonymization/commits`)
- **Fetch date:** 2026-09-23
- **Fetched via:** `raw.githubusercontent.com` directly — `lib/models/pix2pix.py`
  and `LICENSE` copied verbatim, GeneratorUNet/UNetDown/UNetUp extracted.
- **License: MIT.** Copyright (c) 2023 Chair of Human-Centered Artificial
  Intelligence, University of Augsburg — copied verbatim to
  `vendor/LICENSE_ganonymization` per the MIT terms' own requirement to
  include the notice with any copy.

## Provenance — head-segmentation (shared, `../_vendor/`)

- **Source repo:** https://github.com/wiktorlazarski/head-segmentation
- **Pinned commit:** `c1b4b9b4f12f168f13d603747b81e8f264bcf501` (tag
  `v1.3.0`, released 2023-09-07 — a real tagged release, not a floating
  `main` HEAD, verified via `GET /repos/wiktorlazarski/head-segmentation/tags`)
- **Fetch date:** 2026-09-23
- **Fetched via:** `raw.githubusercontent.com` directly —
  `head_segmentation/model.py` and `LICENSE` copied verbatim (`model.py`
  kept whole — only ~54 lines, one class). Also read (not vendored, since
  this project reimplements the wrapper itself — see "Design note" below):
  `segmentation_pipeline.py`, `image_processing.py`, `constants.py`.
- **License: custom, split code/model.** Verbatim, `vendor/LICENSE_head_segmentation`
  (this project's shared `_vendor/` directory):
  > "CODE: You can do what you want we don't care but if something doesn't
  > work for you don't blame us. MODEL COMMERCIAL USAGE: The model was
  > trained with the CelebA dataset. The authors of this dataset do not
  > allow for commercial usage. Therefore, we (authors of this repo) only
  > allow for commercial usage of source code [...] we do not gr[a]nd a
  > commercial license for the model usage which we are providing by
  > default with the repo."

## Compliance flag — read before any commercial/production use

Both checkpoints this backend uses are CelebA-trained:

- head-segmentation's own LICENSE (quoted in full above) explicitly
  withholds a commercial license for its **model weights** specifically
  (the code itself is unrestricted).
- GANonymization's pix2pix checkpoints are *also* trained on CelebA (paper
  Sec. 3.4: "we process the CelebA dataset... for training") and so
  practically inherit the same restriction, even though hcmlab's own MIT
  license text for the *code* doesn't mention it.

This project is academic research (a paper comparing face-anonymization
methods), so this is very likely fine as-is — **flag for compliance review
before any commercial/production use of either checkpoint.**

## What's vendored, and why

**GANonymization** (`vendor/pix2pix_generator.py`): only `GeneratorUNet`,
`UNetDown`, `UNetUp` from `lib/models/pix2pix.py` — the only classes
`backend.py` instantiates. NOT vendored from that same file: `Discriminator`
and `weights_init_normal` (training-only, unused for inference), and
`Pix2Pix(pytorch_lightning.LightningModule)` (the training wrapper —
`backend.py` loads a bare `GeneratorUNet`'s `state_dict` directly instead,
stripping the `generator.`-prefixed keys itself; see "Checkpoint status"
below for how that's verified, not assumed).

**head-segmentation** (`../_vendor/head_segmentation_model.py`): only
`HeadSegmentationModel(smp.Unet)` (~30 lines) from `head_segmentation/model.py`.
**Deliberately not** added as an installed dependency (`head-segmentation`
has no PyPI release — git-only). Its own `setup.py` installs its whole
unpinned `requirements.txt` (`streamlit`, `wandb`, `hydra-core`,
`pytorch-lightning`, `albumentations`, `pandas`, `seaborn`, `matplotlib`,
`gdown`, `loguru`, ...) for functionality the real inference path never
imports — verified by reading `segmentation_pipeline.py`/`image_processing.py`/
`model.py` directly, which together only need `cv2`, `numpy`, `torch`,
`torchvision`, `PIL`, and `segmentation_models_pytorch` (a normal,
wheel-published PyPI package). `albumentations` would also pull
`opencv-python-headless`, the same class of import-name collision already
fixed for `insightface`/`onnxruntime`. `../_segmentation.py` reimplements
the ~12 lines of pre/post-processing glue (`PreprocessingPipeline` +
`HumanHeadSegmentationPipeline.predict()`'s pieces) directly against the
vendored model instead of depending on the upstream wrapper — see that
file's own docstring.

**Reversible fallback**, if a future maintainer prefers tracking upstream
`head-segmentation` directly instead: a pinned git dependency
(`c1b4b9b4f12f168f13d603747b81e8f264bcf501`) plus a
`[[tool.uv.dependency-metadata]]` override stripping its `requirements.txt`
down to what's actually imported — the same mechanism already used for
`insightface` in `pyproject.toml`.

## Design note: full-head segmentation, repurposed for compositing, not background removal

Upstream's own inference pipeline (`main.py::anonymize_image`) **never
calls segmentation at inference** — confirmed by reading `main.py`
directly: the real path is `FaceCrop(align) -> ZeroPaddingResize(512) ->
FacialLandmarks478 -> Pix2PixTransformer`. Segmentation is used only in
`main.py::preprocess`, the *training data prep* path, to remove background
before the pix2pix generator ever sees an image (the paper states this is
because pix2pix produced visually better results without background
variation — Sec. 3.2). Upstream also never composites its output back into
a source frame at all: `anonymize_image` writes a standalone synthesized
face image, nothing more.

This backend uses the same segmentation model for the **opposite**
purpose: not training-time background removal, but this project's own
inference-time compositing mask — full-head coverage is the entire reason
GANonymization was prioritized over CIAGAN (see
`../../../../../research/next-steps/70-decisions.md`'s 2026-09-22 entry
and `../ciagan/NOTICE.md`'s "mask coverage" limitation). This is a genuine
repurposing of upstream's component, not a documented upstream use case —
flagged here explicitly, and its accuracy for this different purpose needs
the same real-video calibration CIAGAN's own compositing went through (see
"Calibration log" below), not assumed correct by analogy.

## Design note: no identity-conditioning mechanism

Unlike CIAGAN's fixed 1200-class one-hot identity vector, pix2pix here is
a deterministic landmark-dots -> face map with no seed/noise input
anywhere in its forward pass (verified against the vendored
`GeneratorUNet.forward()` — a plain U-Net, no conditioning branch at all).
This project's `Identity.seed`-driven "same track always maps to the same
synthetic identity" guarantee, which CIAGAN provides via its one-hot
vector, is **architecturally impossible** to replicate here. `identity_class()`
returns `seed` unchanged purely to satisfy the Backend contract for
manifest logging — it is not a real class index and should not be read as
"the synthetic identity" the way CIAGAN's is.

## Checkpoint status

**Pix2pix generator** — two public checkpoints, Augsburg's own file server
(no upstream checksum, no versioned GitHub Release for either):

- `GANonymization_25.ckpt` ("25 epochs — publication version"):
  https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/ (687,127,579 bytes,
  verified reachable via `curl -I`)
- `GANonymization_50.ckpt` ("50 epochs — demo version"):
  https://mediastore.rz.uni-augsburg.de/get/Sfle_etB1D/ (686,311,019 bytes)

**This project defaults to the 25-epoch checkpoint** (`DEFAULT_WEIGHTS_FILENAME["ganonymization"]`
= `ganonymization_pix2pix_25.ckpt`): its numbers are the ones reported in
the peer-reviewed paper's evaluation tables (Sec. 4), so results from this
project stay directly comparable to the published benchmark. The 50-epoch
checkpoint appears tuned for the project's own public demo, not for
reproducing the paper's reported numbers.

**Head-segmentation checkpoint**: hosted on Google Drive
(`constants.py`'s `HEAD_SEGMENTATION_MODEL_URL`,
`https://drive.google.com/uc?id=1RxXX4g3zMk2CwDtsq-gGleq7aSGoUUHf`),
upstream auto-downloads it via `gdown` — this project deliberately does
**not** add `gdown` as a dependency (see "What's vendored, and why" above,
and this project's existing norm: CIAGAN requires a pre-placed file +
raises `FileNotFoundError` with instructions). Download manually, e.g.
`uvx gdown '<url>' -O weights/head_segmentation.ckpt` (a one-off shell
command, not a project dependency — Google Drive's large-file confirmation
redirect makes plain `curl`/`wget` unreliable here).

**Verified on serra1, 2026-09-23** (downloaded via `curl`/`gdown` respectively):

- `weights/ganonymization_pix2pix_25.ckpt`: 687,127,579 bytes, SHA256
  `eba49bd525033b55022a91d6b4398ab088b51339fd6895db7afdd648d776eec9`.
  A real PyTorch Lightning checkpoint (`state_dict`/`hyper_parameters`/
  `optimizer_states`/...). `hyper_parameters` records `n_epochs: 50`
  (the *configured training budget*, not this specific save), but
  `ckpt["epoch"] == 24` confirms this file genuinely is the 25th-epoch
  save (0-indexed) — i.e. really is the "25 epochs" checkpoint the
  filename claims, not a naming mismatch. `state_dict`'s 17
  `generator.*`-prefixed keys all loaded into the vendored `GeneratorUNet`
  with `strict=True` — 17 is correct, not suspiciously low: `InstanceNorm2d`
  defaults to `affine=False` (no learnable params), so each of the 15
  down/up blocks contributes exactly one Conv2d/ConvTranspose2d weight
  tensor (bias=False), plus the final block's Conv2d weight+bias = 17.
- `weights/head_segmentation.ckpt`: 359,589,211 bytes, SHA256
  `f40446ae67288b2721c942543cf24e7439fd116535b64e3d06778f580baf5b4a`.
  Also a Lightning checkpoint. `hyper_parameters`: `encoder_name=resnet34`,
  `encoder_depth=5`, `nn_image_input_resolution=512` — read dynamically
  from the checkpoint by `_segmentation.py::load_head_segmentation()`, not
  hardcoded. `state_dict` has 279 keys; 278 are real
  `neural_net.*`-prefixed model keys (all matched, 0 missing) plus one
  unrelated `criterion.weight` key (the training loss's class-weight
  buffer, not part of the model) — correctly dropped by `strict=False`.
  `load_head_segmentation()` now logs this exact breakdown and raises if
  any *missing* key ever appears (which would mean a partially
  random-initialized model silently passed through).

**First real-checkpoint run** (10 frames, `runs/demo`, both checkpoints
above): produced genuinely anonymized, non-random composited output —
visually confirmed a face-shaped synthetic region pasted into what looks
like a fuller head region than `ciagan`'s jaw-only mask (small sample,
not a rigorous visual comparison). A seam is visible at the mask boundary.
**Not yet evaluated**, and explicitly deferred pending further review:
whether `_rotation_matrix()`'s sign convention is actually correct on a
real tilted head, whether `context_ratio=0.6` needs recalibration, and
seam/noise quality — the real calibration pass this NOTICE's "Calibration
log" section below expects.

## Verified facts from reading the pinned source directly (not assumed)

- **Landmark canvas is dots, not connected lines.**
  `lib/transform/facial_landmarks_478_transformer.py` calls
  `mediapipe.solutions.drawing_utils.draw_landmarks(image, landmark_list,
  landmark_drawing_spec=DrawingSpec(color=WHITE_COLOR, thickness=1,
  circle_radius=0))` — `connections=` is never passed (defaults to `None`).
  Reading mediapipe's own `draw_landmarks` body: `if connections:` gates
  the *entire* line-drawing block, so with `connections=None`, no lines
  are ever drawn — only a small dot per landmark. This directly
  contradicts informal "landmark mesh" descriptions found elsewhere
  (including this project's own earlier research notes,
  `research/papers/gan/rw-hellmann-ganonymization.md`) — corrected here
  against the real source, not the secondary description.
- **Real inference ordering**: `main.py::anonymize_image` calls
  `FaceCrop(align=True) -> ZeroPaddingResize(img_size) ->
  FacialLandmarks478() -> Pix2PixTransformer(...)`, in that order —
  landmarks are extracted from the *final* letterboxed canvas, not
  detected once and rescaled. This backend's `_try_generate()` mirrors
  that ordering (a second, independent FaceMesh pass on the
  rotated+letterboxed canvas), not a rescale of an earlier pass's points.
- **RetinaFace's own `align=True` does real eye-based rotation correction**
  (`retinaface/commons/postprocess.py::alignment_procedure`, cosine-rule
  eye-angle rotation) — confirmed directly, not inferred. This project
  skips RetinaFace (see module docstring in `backend.py`), so this
  backend's own `_rotation_matrix()` exists specifically to not start out
  missing something upstream's own pipeline has.
- **`Pix2Pix` (the training wrapper, not vendored) is a real
  `pytorch_lightning.LightningModule`** wrapping `self.generator =
  GeneratorUNet()` — confirmed from `lib/models/pix2pix.py`'s own
  `__init__`. This is the basis for `backend.py`'s `generator.`-prefix
  stripping when loading the checkpoint's `state_dict` — verified against
  the class definition, but **not yet verified against the real 655MB
  checkpoint file itself** (a real risk this project's own `backend.py`
  handles by raising a clear error naming the actual top-level key
  prefixes found, rather than silently loading nothing or crashing
  opaquely, if this assumption turns out wrong for the real file).
- **`HeadSegmentationModel.load_from_checkpoint`'s own `torch.load(...)`
  call has no explicit `weights_only` argument** — PyTorch 2.6 flipped
  that default to `True`, which cannot unpickle a full Lightning
  checkpoint's `hyper_parameters` dict (this project's `torch<2.7` pin
  makes 2.6.x reachable). `../_segmentation.py::load_head_segmentation()`
  reimplements the same loading logic with `weights_only=False` explicit
  instead of calling the vendored method — see that file's docstring.
  (The same landmine applies to the GANonymization pix2pix checkpoint,
  also a Lightning checkpoint — `backend.py`'s own loader passes
  `weights_only=False` there too.)
- **head-segmentation's own inference expects RGB, not BGR**
  (`image_processing.py::PreprocessingPipeline.preprocess_image` hands the
  array straight to `PIL.Image.fromarray()`), and normalizes with
  ImageNet mean/std (`0.485/0.456/0.406`, `0.229/0.224/0.225`) — **not**
  the `0.5/0.5/0.5` this project's pix2pix generator uses. The two
  vendored models are not interchangeable in either respect; see
  `../_segmentation.py`'s docstring.
- **Output label convention**: `constants.py`'s `LABEL2INDEX = {"background":
  0, "head": 1}` — `../_segmentation.py::predict_head_mask()` returns
  `argmax(...) == 1` as the head mask, verified against this, not guessed.

## Calibration log

### 2026-09-23: checkpoints obtained, first real run confirms the pipeline works — full calibration paused for review

Both checkpoints downloaded and verified on serra1 (see "Checkpoint
status" above — sizes, SHA256s, and key-match counts all confirmed
against the real files, not assumed). A first real run (10 frames of
`runs/demo`, both checkpoints, `--ctx-id 0`) produced genuine, plausible
anonymized output — not random-init garbage, a real face-shaped
composited region with a visible seam, appearing to cover more of the
head than `ciagan`'s jaw-only mask on casual visual inspection.

**Paused here, deliberately, before a full calibration pass** — the
project owner asked to review this checkpoint before continuing into the
open-ended visual-tuning work CIAGAN's own four rounds required. Still
open, not yet evaluated:

- Whether `_rotation_matrix()`'s sign convention (derived algebraically in
  its own docstring, not yet checked against a real tilted-head frame) is
  actually correct — `--align-rotation`/`--no-align-rotation` should be
  swept side-by-side on a real tilted head before trusting the derivation.
- Whether `context_ratio=0.6` (an untuned starting guess, see
  `models/DEFAULT_CONTEXT_RATIO`'s comment) needs adjustment.
- Seam/noise quality at full-video scale, and whether the mask genuinely
  achieves full-head coverage (hair/forehead/ears) or only looks that way
  in the small sample reviewed so far.
- How often the rotation-then-retry-unrotated fallback in `generate()`
  actually triggers on real footage.

Not a documented root cause or a tuned default yet — a status snapshot to
resume from, in the same spirit as `ciagan/NOTICE.md`'s own dated entries
(each one is a real, tested finding, not a guess), just mid-process here
rather than complete.
