"""E3 — with and without our ideas (C2, seed candidates, P2). Design and outputs: README.md.

    uv run experiments/e3_contribution/run.py [--video NAME ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import pending  # noqa: E402

pending(__file__, [
    "run.py orchestration of E3a–E3d (all pipeline components already exist)",
    "src/eval/ utility probes and bootstrap / paired statistics shared with E2",
])
