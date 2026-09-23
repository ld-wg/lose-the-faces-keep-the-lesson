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
}

#: model name -> expected weights filename under CONFIG.weights_dir.
DEFAULT_WEIGHTS_FILENAME = {
    "ciagan": "ciagan_generator.pth",
    "ganonymization": "ganonymization_pix2pix_25.ckpt",
}

#: Expected head-segmentation checkpoint filename under CONFIG.weights_dir.
#  A plain string, not a per-model dict: it's the same shared checkpoint
#  and model class (models/_vendor/head_segmentation_model.py) for both
#  ganonymization (required — its primary compositing mask) and ciagan
#  (opt-in, via --refine-mask — see models/ciagan/NOTICE.md).
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
    # NOT yet calibrated against real footage (starting value only).
    # Deliberately looser than ciagan's tuned 0.15: full-head segmentation
    # needs hair/ears/forehead actually present inside the crop to segment
    # at all, unlike ciagan's landmark-only mask. ciagan's own
    # noise-vs-context_ratio finding (above) was tied to dlib's detected
    # inter-eye distance shifting with crop context, feeding a
    # landmark-radius-derived affine transform — a mechanism specific to
    # that geometry. ganonymization instead letterbox-resizes the WHOLE
    # crop into a fixed 512 canvas, so that mechanism may not transfer;
    # re-derive empirically rather than assuming it does or doesn't.
    "ganonymization": 0.6,
}

MODEL_NAMES = tuple(_MODULES)


def load_backend_class(model: str):
    """Import and return the `Backend` class for `model`, on demand."""
    if model not in _MODULES:
        raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_NAMES)}")
    module = importlib.import_module(f".{_MODULES[model]}.backend", package=__name__)
    return module.Backend
