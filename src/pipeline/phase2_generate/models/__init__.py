"""Registry of Phase 2 generation backends, keyed by `--model` name.

Same lazy-import registry shape as `phase1_detect.models` — see that
module's docstring. Every backend module under `<name>/backend.py` exposes
a class named `Backend` with a `generate(crop, seed) -> Optional[np.ndarray]`
method; constructor signatures vary by backend (unlike Phase 1's detectors,
different generator architectures take genuinely different config).
"""

from __future__ import annotations

import importlib

#: model name -> submodule under this package
_MODULES = {
    "ciagan": "ciagan",
    "ganonymization": "ganonymization",
    "blanket": "blanket",
}

#: model name -> expected weights filename under CONFIG.weights_dir.
#
#  ganonymization: defaults to the **50-epoch** checkpoint, not the
#  25-epoch "publication version" — reversed from this project's initial
#  choice after real-video testing (see models/ganonymization/NOTICE.md's
#  calibration log, 2026-09-23). The 25-epoch checkpoint's numbers match
#  the paper's own evaluation tables, which is a real reason to prefer it
#  for literal reproducibility, but on this project's actual footage it
#  produces visibly noisier/less coherent output than the 50-epoch one —
#  confirmed side-by-side on the same real crop, not assumed. Pass
#  `--weights weights/ganonymization_pix2pix_25.ckpt` explicitly if
#  reproducing the paper's own reported numbers is the goal.
#  blanket: deliberately absent. Unlike ciagan/ganonymization, there is no
#  single checkpoint this project manages under CONFIG.weights_dir — all of
#  BLANKET's own checkpoints (SDXL inpainting + refiner + 2 ControlNets,
#  inswapper_128, GFPGAN) are resolved entirely inside the two external
#  venvs of the sibling blanket-anonymizer-bridge repo, outside this
#  project's control. See models/blanket/NOTICE.md.
DEFAULT_WEIGHTS_FILENAME = {
    "ciagan": "ciagan_generator.pth",
    "ganonymization": "ganonymization_pix2pix_50.ckpt",
}

#: Expected head-segmentation checkpoint filename under CONFIG.weights_dir.
#  A plain string, not a per-model dict: it's the same shared checkpoint
#  and model class (models/_vendor/head_segmentation_model.py) for
#  ganonymization (required — its primary compositing mask), ciagan
#  (opt-in, via --refine-mask — see models/ciagan/NOTICE.md), and blanket
#  (default-on --refine-mask — see models/blanket/NOTICE.md; here it's the
#  only way to address BLANKET's own documented weak-identity-suppression
#  limitation, not just a seam cleanup).
DEFAULT_SEGMENTATION_WEIGHTS_FILENAME = "head_segmentation.ckpt"

#: model name -> expected dlib shape-predictor filename under CONFIG.weights_dir
#  (only backends that need a separate landmark model populate this).
DEFAULT_DLIB_PREDICTOR_FILENAME = {
    "ciagan": "shape_predictor_68_face_landmarks.dat",
}

#: model name -> the generator's native square canvas resolution. Not a
#  free per-run tunable — each checkpoint's architecture is trained at a
#  fixed resolution (ciagan: 128, hard-enforced by its own backend.py;
#  ganonymization: 512, per the paper's Face Extraction step, also
#  hard-enforced). Exists so run.py doesn't hardcode one number's default
#  for every --model.
DEFAULT_IMG_SIZE = {
    "ciagan": 128,
    "ganonymization": 512,
    # blanket: informational only, NOT enforced by this backend (unlike the
    # other two) — verified against BLANKET's own real
    # stable_diffusion_parameters.yaml: SDXL inpainting at 896x896. The
    # actual resize happens entirely inside the external IdentityGenerator
    # process, outside this project's control.
    "blanket": 896,
}

#: model name -> crop padding as a multiple of the detected box's own
#  width/height (see run.py's `_context_crop`). Each backend has its own
#  implicit portrait-framing convention baked into its checkpoint/training
#  data, so this isn't a generic constant — see models/ciagan/NOTICE.md for
#  how ciagan's value was derived (empirically, against real crops, not
#  guessed from anthropometric ratios alone). Lower than you'd expect from
#  padding alone: a bigger context crop measurably increases visible
#  generation noise for this checkpoint, not just the framing. Interacts
#  with ciagan's own `portrait_scale` default (`models/ciagan/backend.py`)
#  — re-check both together if you change either, see NOTICE.md.
DEFAULT_CONTEXT_RATIO = {
    "ciagan": 0.15,
    # Calibrated via a real-video sweep on video-demo-2.mov (see
    # models/ganonymization/NOTICE.md's calibration log) — NOT a value
    # ciagan's own mechanism transfers to: ganonymization letterbox-resizes
    # the WHOLE crop into a fixed 512 canvas (no landmark-radius-derived
    # affine transform), and full-head segmentation needs hair/ears/
    # forehead physically present in the crop to segment at all. The
    # sweep ({0.15, 0.25, 0.35, 0.45, 0.6, 0.8}) was non-monotonic — a
    # real dip at 0.25-0.45 — with coverage peaking at the highest value
    # tested (0.8: 1226 "ok" vs 0.6's 1194, out of 2928 face-observations)
    # and comparable visual quality between them. Values above 0.8 were
    # not tested — flagged as an open question, not chased further this
    # round.
    "ganonymization": 0.8,
    # blanket: starting point only, borrowed from ganonymization's own
    # calibrated value for the same reason (--refine-mask's full-head
    # segmentation needs hair/forehead physically present in the crop) —
    # NOT itself calibrated against real blanket output yet. Re-sweep once
    # a real run exists, see models/blanket/NOTICE.md's calibration log.
    "blanket": 0.8,
}

MODEL_NAMES = tuple(_MODULES)


def load_backend_class(model: str):
    """Import and return the `Backend` class for `model`, on demand."""
    if model not in _MODULES:
        raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_NAMES)}")
    module = importlib.import_module(f".{_MODULES[model]}.backend", package=__name__)
    return module.Backend
