"""Phase 1 -> Phase 2 data contract.

Shared, dependency-free types for what `phase1_detect` writes to disk and what
`phase2_generate` reads. Living here (not inside either phase package) avoids
one phase importing the other's internals.

Two artifacts, two different sizes, two different lifetimes — that split is
deliberate, not accidental:
  - `detections.jsonl`: one `Frame` per line, per-frame face geometry. Too big
    to hold in memory for a full lecture, so it's a line-delimited stream.
  - `tracks.json`: one `Manifest`, whole-video. Small (one `Identity` per
    student, not per frame) so it's fine as a single JSON object.

`Identity.seed` is the pipeline's "fixed synthetic identity per track"
decision (research/pipeline.md Part 2/4) made concrete. Every version of that
decision before this file existed was prose — nothing actually assigned a
seed. `derive_seed()` is deterministic (not `random.randint`) so re-running
Phase 1 on the same video reproduces the same per-student identity without
persisting any RNG state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2), source-frame pixels
Landmarks = list[tuple[float, float]]    # 5-point: eyes, nose, mouth corners


def derive_seed(video_source: str, track_id: int) -> int:
    """Deterministic per-track seed. Same (video, track_id) -> same seed, always."""
    digest = hashlib.blake2b(f"{video_source}:{track_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF  # non-negative, fits int32


@dataclass
class Face:
    """One detected face, in one frame."""
    track_id: int
    box: Box
    confidence: float
    landmarks: Optional[Landmarks] = None
    crop_path: Optional[str] = None  # relative to the run dir, e.g. "crops/000123_4.jpg"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "box": [round(v, 1) for v in self.box],
            "conf": round(self.confidence, 3),
            "landmarks": [[round(x, 1), round(y, 1)] for x, y in self.landmarks] if self.landmarks else None,
            "crop_path": self.crop_path,
        }

    @staticmethod
    def from_dict(d: dict) -> "Face":
        lm = d.get("landmarks")
        return Face(
            track_id=d["track_id"],
            box=tuple(d["box"]),
            confidence=d["conf"],
            landmarks=[tuple(p) for p in lm] if lm else None,
            crop_path=d.get("crop_path"),
        )


@dataclass
class Frame:
    """One line of `detections.jsonl`."""
    frame_id: int
    faces: list[Face] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"frame_id": self.frame_id, "tracks": [f.to_dict() for f in self.faces]})

    @staticmethod
    def from_json(line: str) -> "Frame":
        d = json.loads(line)
        return Frame(frame_id=d["frame_id"], faces=[Face.from_dict(t) for t in d["tracks"]])


@dataclass
class Identity:
    """One track's whole-video identity — what Phase 2 seed-locks its generator to.

    `representative_crop` is the highest-confidence observation's crop: a
    reasonable default reference image until/unless a better selection rule
    (sharpest, most frontal) is needed.
    """
    track_id: int
    seed: int
    first_frame: int
    last_frame: int
    num_observations: int
    representative_crop: Optional[str] = None
    representative_confidence: Optional[float] = None


@dataclass
class Video:
    source: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class Manifest:
    """`tracks.json` — the whole-video half of the Phase 1 -> Phase 2 contract."""
    video: Video
    identities: list[Identity]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: Path) -> "Manifest":
        d = json.loads(Path(path).read_text())
        return Manifest(video=Video(**d["video"]), identities=[Identity(**t) for t in d["identities"]])
