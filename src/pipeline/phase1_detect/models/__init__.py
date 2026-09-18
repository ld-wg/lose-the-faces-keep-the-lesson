"""Registry of Phase 1 detector backends, keyed by `--model` name.

Backend modules are imported lazily, on first use of a given model — so
selecting "scrfd-10gf" never imports "yolo_facev2_s"'s torch dependency (or
touches "scrfd_34gf"'s mmdet-only convert.py), and only the optional extra
actually needed has to be installed (see pyproject.toml's
[project.optional-dependencies]).

Every backend module under `<name>/backend.py` exposes a class named
`Backend` with the constructor signature
`Backend(conf_threshold, det_size, ctx_id, weights)` and a
`detect(frame) -> list[Detection]` method (see `..detector.Detection`).
"""

from __future__ import annotations

import importlib

#: model name -> submodule under this package
_MODULES = {
    "scrfd-10gf": "scrfd_10gf",
    "scrfd-34gf": "scrfd_34gf",
    "yolo-facev2-s": "yolo_facev2_s",
}

#: model name -> expected weights filename under CONFIG.weights_dir.
#  scrfd-10gf has none: it resolves its own pretrained pack by name through
#  InsightFace's own model cache, not a local file in this repo's weights dir.
DEFAULT_WEIGHTS_FILENAME = {
    "scrfd-34gf": "scrfd_34g.onnx",
    "yolo-facev2-s": "yolo_facev2s.onnx",
}

MODEL_NAMES = tuple(_MODULES)


def load_backend_class(model: str):
    """Import and return the `Backend` class for `model`, on demand."""
    if model not in _MODULES:
        raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(MODEL_NAMES)}")
    module = importlib.import_module(f".{_MODULES[model]}.backend", package=__name__)
    return module.Backend
