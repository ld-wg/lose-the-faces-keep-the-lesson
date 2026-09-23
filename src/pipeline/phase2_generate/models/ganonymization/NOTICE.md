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

*(SHA256 of both downloaded files, and `strict=True`/`strict=False`
key-match results against the real files, to be recorded here after the
real-checkpoint calibration run on serra1 — not yet performed as of this
NOTICE's initial commit.)*

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

*(To be filled in during the real-checkpoint calibration run on serra1 —
see the project plan's Verification section. Expect several real,
documented rounds of bugs before this is mergeable, the same bar CIAGAN's
own four-round calibration was held to — not written as a single
confident pass. In particular: whether `align_rotation` measurably helps
or hurts on real tilted-head frames, whether `context_ratio=0.6` needs
recalibration, whether the `generator.`-prefix and label-index assumptions
above hold against the real downloaded files, and how often the
rotation-then-refail fallback in `generate()` actually triggers.)*
