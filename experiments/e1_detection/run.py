"""E1 — face detectors. Design and outputs: README.md in this folder.

    uv run experiments/e1_detection/run.py [--video NAME ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import pending  # noqa: E402

pending(__file__, [
    "src/eval/widerface.py — WIDER FACE val: official AP protocol, P/R/F1 @ conf, recall @ FPPI, latency, memory",
    "src/eval/detection_stats.py — track statistics from detections.jsonl / tracks.json",
])
