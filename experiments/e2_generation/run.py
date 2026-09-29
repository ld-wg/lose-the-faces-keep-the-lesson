"""E2 — generators in their original form. Design and outputs: README.md in this folder.

    uv run experiments/e2_generation/run.py [--video NAME ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import pending  # noqa: E402

pending(__file__, [
    "src/pipeline/phase2_generate/models/censor/ — blur and mosaic censorship backend",
    "src/eval/ utility probes — emotion (FER), head pose (1k3d68), gaze, age/gender (genderage)",
    "src/eval/evaluate.py — rank-5, Privacy Gain, bootstrap intervals over tracks, face-size / yaw strata",
])
