"""Central path/environment configuration for the privacy-preserving-yolo pipeline.

Resolution order (later sources win):
    1. Built-in defaults (paths relative to this repo)
    2. ``config.local.json`` at the repo root (gitignored — your machine-specific overrides)
    3. Environment variables prefixed with ``PPY_`` (highest priority)

Available keys and their env-var overrides:

    ======================  ====================  ==========================================
    JSON key                Env var               Meaning
    ======================  ====================  ==========================================
    ``widerface_root``      ``PPY_WIDERFACE``     Raw WIDER FACE root. Must contain
                                                  ``wider_face_split/``, ``WIDER_train/``,
                                                  ``WIDER_val/`` (extracted).
    ``dataset_dir``         ``PPY_DATASET_DIR``   Where the converted YOLO dataset
                                                  (images/ + labels/) is written.
    ``weights_dir``         ``PPY_WEIGHTS_DIR``   Directory with pretrained weights.
    ``runs_dir``            ``PPY_RUNS_DIR``      Ultralytics training output root.
    ======================  ====================  ==========================================

Usage:
    from src.config import CONFIG          # repo-root on sys.path
    CONFIG.widerface_root                  # pathlib.Path
    CONFIG.require_widerface()             # raises with a helpful message if missing

    # or, from inside src/ (e.g. src/identify/train_face_detector.py):
    from config import CONFIG

CLI:
    python -m src.config            # print resolved config + validation status
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG_FILE = REPO_ROOT / "config.local.json"
EXAMPLE_CONFIG_FILE = REPO_ROOT / "config.example.json"

#: env var name for each config key
_ENV_VARS = {
    "widerface_root": "PPY_WIDERFACE",
    "dataset_dir": "PPY_DATASET_DIR",
    "weights_dir": "PPY_WEIGHTS_DIR",
    "runs_dir": "PPY_RUNS_DIR",
}

#: built-in defaults (used when nothing overrides them)
_DEFAULTS = {
    "widerface_root": REPO_ROOT / "data" / "widerface",
    "dataset_dir": REPO_ROOT / "data" / "wider_face_yolo",
    "weights_dir": REPO_ROOT / "weights",
    "runs_dir": REPO_ROOT / "runs",
}


@dataclass(frozen=True)
class Config:
    """Resolved, immutable path configuration."""

    widerface_root: Path
    dataset_dir: Path
    weights_dir: Path
    runs_dir: Path

    # -- WIDER FACE expected layout -------------------------------------- #
    @property
    def train_annotations(self) -> Path:
        return self.widerface_root / "wider_face_split" / "wider_face_train_bbx_gt.txt"

    @property
    def val_annotations(self) -> Path:
        return self.widerface_root / "wider_face_split" / "wider_face_val_bbx_gt.txt"

    @property
    def train_images(self) -> Path:
        return self.widerface_root / "WIDER_train" / "images"

    @property
    def val_images(self) -> Path:
        return self.widerface_root / "WIDER_val" / "images"

    # -- validation ------------------------------------------------------- #
    def widerface_status(self) -> dict[str, tuple[Path, bool]]:
        """Return {description: (path, exists)} for everything training needs."""
        return {
            "WIDER FACE root": (self.widerface_root, self.widerface_root.is_dir()),
            "train annotations": (self.train_annotations, self.train_annotations.is_file()),
            "val annotations": (self.val_annotations, self.val_annotations.is_file()),
            "train images": (self.train_images, self.train_images.is_dir()),
            "val images": (self.val_images, self.val_images.is_dir()),
        }

    def require_widerface(self) -> None:
        """Raise FileNotFoundError with setup instructions if the dataset is missing."""
        missing = [
            (desc, path)
            for desc, (path, ok) in self.widerface_status().items()
            if not ok
        ]
        if not missing:
            return
        lines = "\n".join(f"  [missing] {desc}: {path}" for desc, path in missing)
        raise FileNotFoundError(
            "WIDER FACE dataset not found or incomplete:\n"
            f"{lines}\n\n"
            "Fix by either:\n"
            f"  1. Editing {LOCAL_CONFIG_FILE.name} "
            f"(copy {EXAMPLE_CONFIG_FILE.name} as a template), or\n"
            f"  2. Setting the {_ENV_VARS['widerface_root']} environment variable.\n"
            "The directory must contain wider_face_split/, WIDER_train/ and WIDER_val/ "
            "(extract the archives first — ~5 GB of free space required)."
        )


def _load() -> Config:
    values: dict[str, Path] = {k: Path(v) for k, v in _DEFAULTS.items()}

    if LOCAL_CONFIG_FILE.is_file():
        try:
            data = json.loads(LOCAL_CONFIG_FILE.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ERROR: {LOCAL_CONFIG_FILE} is not valid JSON: {exc}")
        unknown = sorted(set(data) - set(_DEFAULTS))
        if unknown:
            print(
                f"WARNING: ignoring unknown keys in {LOCAL_CONFIG_FILE.name}: "
                f"{', '.join(unknown)} (valid: {', '.join(sorted(_DEFAULTS))})",
                file=sys.stderr,
            )
        for key in _DEFAULTS:
            if key in data:
                values[key] = Path(data[key]).expanduser()

    for key, env_var in _ENV_VARS.items():
        env_val = os.environ.get(env_var)
        if env_val:
            values[key] = Path(env_val).expanduser()

    return Config(**{k: v.resolve() for k, v in values.items()})


#: Singleton resolved at import time.
CONFIG = _load()


def _main() -> None:
    print(f"Repo root:        {REPO_ROOT}")
    print(f"Local overrides:  {LOCAL_CONFIG_FILE} "
          f"({'found' if LOCAL_CONFIG_FILE.is_file() else 'not found — using defaults/env'})")
    print()
    for key in sorted(_DEFAULTS):
        env_var = _ENV_VARS[key]
        env_set = " [from env]" if os.environ.get(env_var) else ""
        print(f"{key:16} {getattr(CONFIG, key)}{env_set}")
    print()
    ok = True
    for desc, (path, exists) in CONFIG.widerface_status().items():
        mark = "ok" if exists else "missing"
        ok = ok and exists
        print(f"  [{mark}] {desc}: {path}")
    if not ok:
        print("\nDataset incomplete — see: python -m src.config  (or README → Configuration)")
        sys.exit(1)
    print("\nWIDER FACE dataset ready.")


if __name__ == "__main__":
    _main()
