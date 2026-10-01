"""Phase 1 statistics on a video without ground truth (E1): how much does a
detector + tracker find, and how fragmented are its tracks?

From `detections.jsonl` (detector boxes only, including tracks too short to
be confirmed) and `run_stats.json`:
    faces_per_frame, tracks, median_track_frames, short_track_share
    (tracks under `short` frames: a fragmentation proxy, since there are no
    ground-truth identities to count ID switches), fps.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..pipeline.contracts import Frame


def track_stats(phase1_dir: Path, short: int = 10) -> dict:
    phase1_dir = Path(phase1_dir)
    per_track, frames, boxes = Counter(), 0, 0
    with (phase1_dir / "detections.jsonl").open() as f:
        for fr in map(Frame.from_json, f):
            frames += 1
            for face in fr.faces:
                if face.source in ("detector", "unconfirmed"):
                    per_track[face.track_id] += 1
                    boxes += 1
    lengths = np.array(list(per_track.values()))
    run = json.loads((phase1_dir / "run_stats.json").read_text()) if (phase1_dir / "run_stats.json").is_file() else {}
    return {
        "frames": frames,
        "faces_per_frame": round(boxes / frames, 2) if frames else None,
        "tracks": int(len(lengths)),
        "confirmed_tracks": run.get("tracks"),
        "median_track_frames": float(np.median(lengths)) if len(lengths) else None,
        "short_track_share": round(float((lengths < short).mean()), 3) if len(lengths) else None,
        "fps": run.get("fps"),
    }
