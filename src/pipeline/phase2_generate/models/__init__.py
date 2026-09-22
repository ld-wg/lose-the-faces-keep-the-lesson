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
}

#: model name -> expected weights filename under CONFIG.weights_dir.
DEFAULT_WEIGHTS_FILENAME = {
    "ciagan": "ciagan_generator.pth",
}

#: model name -> expected dlib shape-predictor filename under CONFIG.weights_dir
#  (only backends that need a separate landmark model populate this).
DEFAULT_DLIB_PREDICTOR_FILENAME = {
    "ciagan": "shape_predictor_68_face_landmarks.dat",
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
}

MODEL_NAMES = tuple(_MODULES)


def load_backend_class(model: str):
    """Import and return the `Backend` class for `model`, on demand."""
    if model not in _MODULES:
        raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_NAMES)}")
    module = importlib.import_module(f".{_MODULES[model]}.backend", package=__name__)
    return module.Backend
