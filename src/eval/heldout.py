"""Held-out face recognizer — FaceNet InceptionResnetV1 trained on VGGFace2.

Used only to *measure* privacy, never to guide generation (plan D6). The
guidance model is ArcFace `w600k_r50` (ResNet-50, WebFace600K, additive
angular margin loss); this evaluator differs in architecture, training
set and loss, so a push that only fools the guidance model shows up as a
gap between the two scores instead of as a privacy gain.

Weights: facenet-pytorch's released `20180402-114759-vggface2.pt` (a port
of David Sandberg's facenet model), downloaded once with
`python -m src.eval.heldout --download` into `CONFIG.weights_dir`. Model
definition vendored in `_vendor/` — see that file's header for why the
package itself is not a dependency. Licensing notes: `NOTICE.md`.

Input convention (facenet-pytorch's `fixed_image_standardization`): RGB,
`(x − 127.5) / 128`, 160x160. Faces are aligned with the same 5-point
similarity transform as the ArcFace path, onto the ArcFace template scaled
to 160 with a 10% margin — FaceNet was trained on looser MTCNN crops than
ArcFace's tight template, and the margin is the calibration knob. Whether
the evaluator works on this footage is checked, not assumed:
`evaluate.py` reports its true-accept rate on real same-track pairs at the
in-domain threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402

from ..pipeline.identity import align  # noqa: E402

logger = logging.getLogger(__name__)

FACENET_WEIGHTS_URL = (
    "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
)
DEFAULT_WEIGHTS_FILENAME = "facenet_vggface2.pt"
FACENET_SIZE = 160
FACENET_MARGIN = 0.1


def download_weights(dest: Path) -> str:
    """Download the VGGFace2 state_dict to `dest`; returns its sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        logger.info(f"downloading {FACENET_WEIGHTS_URL}")
        urllib.request.urlretrieve(FACENET_WEIGHTS_URL, tmp_path)
        shutil.move(str(tmp_path), dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


class FaceNetEmbedder:
    """512-d L2-normalized FaceNet embeddings of 5-point-aligned faces."""

    def __init__(self, weights: Optional[Path] = None, ctx_id: int = 0, margin: float = FACENET_MARGIN):
        self.weights = Path(weights) if weights else CONFIG.weights_dir / DEFAULT_WEIGHTS_FILENAME
        self.ctx_id = ctx_id
        self.margin = margin
        self._model = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch

        from ..pipeline.phase2_generate.models._segmentation import hash_and_log, resolve_torch_device
        from ._vendor.inception_resnet_v1 import InceptionResnetV1

        if not self.weights.is_file():
            raise FileNotFoundError(
                f"FaceNet weights not found at {self.weights} — run `python -m src.eval.heldout --download`"
            )
        hash_and_log(self.weights, "FaceNet (VGGFace2) evaluator")
        state = torch.load(self.weights, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("logits.")}
        model = InceptionResnetV1()
        model.load_state_dict(state, strict=True)
        self._device = resolve_torch_device(self.ctx_id)
        self._model = model.eval().to(self._device)

    def align(self, image_bgr: np.ndarray, landmarks: Sequence[Sequence[float]]) -> np.ndarray:
        return align.warp(image_bgr, landmarks, FACENET_SIZE, self.margin)

    def embed(self, image_bgr: np.ndarray, landmarks: Sequence[Sequence[float]]) -> np.ndarray:
        return self.embed_aligned([self.align(image_bgr, landmarks)])[0]

    def embed_aligned(self, aligned_bgr: list[np.ndarray]) -> np.ndarray:
        import torch

        self._load()
        batch = np.stack([a[:, :, ::-1] for a in aligned_bgr]).astype(np.float32)  # BGR -> RGB
        x = torch.from_numpy(batch).permute(0, 3, 1, 2)
        x = (x - 127.5) / 128.0
        with torch.no_grad():
            return self._model(x.to(self._device)).cpu().numpy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Held-out FaceNet evaluator — weight download")
    p.add_argument("--download", action="store_true", help="download the VGGFace2 weights")
    p.add_argument("--dest", default=str(CONFIG.weights_dir / DEFAULT_WEIGHTS_FILENAME))
    args = p.parse_args()
    if not args.download:
        p.error("nothing to do (pass --download)")
    sha = download_weights(Path(args.dest))
    print(f"{args.dest}  sha256={sha}")


if __name__ == "__main__":
    main()
