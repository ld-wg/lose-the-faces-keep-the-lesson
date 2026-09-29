"""BLANKET backend — orchestrates BLANKET's own upstream code (Hadera et al.,
ICDL 2025, `ctu-vras/blanket-infant-face-anonym`, GPL-3.0) as two isolated
external processes, never imported into this project's own process.

**Why process isolation, not vendoring (see `NOTICE.md` for the full case)**:
this project's own repository has no `LICENSE` file and is already public.
Importing BLANKET's GPL-3.0 source directly into this process would very
likely place the whole repository under GPL-3.0's copyleft the moment it's
distributed — which already happens continuously via `git clone`/`pull`.
Nothing from BLANKET's own source is copied or imported here; this module
only talks to two long-lived external server processes over a Unix socket,
each running in its own separate environment inside the sibling
`blanket-anonymizer-bridge` repository (GPL-3.0 itself, by design — see that
repo's own LICENSE/README) — that repository is the only place BLANKET's
code is actually imported.

**Two isolated services, not one process** — verified directly against
BLANKET's own real source (`gh api`/`raw.githubusercontent.com` reads of
`ctu-vras/blanket-infant-face-anonym`, 2026-09-23/24), not assumed from the
paper's prose or this project's own earlier (partly wrong) research note:

1. **`IdentityGenerator`** — real function
   `blanket.anonymization.pipelines.image_pipeline.generate_synthetic_identity()`.
   Runs once per `Identity.seed` (cached below), never once per frame. Needs
   torch/diffusers/transformers/accelerate/controlnet-aux/ultralytics/spiga —
   confirmed via BLANKET's own `pyproject.toml` plus a direct read of
   `image_pipeline.py`'s own imports.
2. **`FaceSwapper`** — real class
   `blanket.anonymization.methods.facefusion.FaceFusionDirectAnonymizer`,
   calling only FaceFusion's `face_swapper`/`face_enhancer` modules, never
   `deep_swapper` (the one FaceFusion processor that pulls in `deepface`/
   TensorFlow — confirmed unused by this call path via a direct source
   read, not assumed). Needs onnxruntime(-gpu)/opencv/numpy/pyyaml/tqdm plus
   the vendored `external/facefusion` package.

Splitting these into two separate venvs (rather than one shared environment,
which is all BLANKET's own `pip install -e .` offers) avoids pulling
TensorFlow/Flask/Gradio into a venv that never uses them, and avoids forcing
one dependency resolution to satisfy both the torch/diffusers world and the
onnxruntime/FaceFusion world at once — the same class of conflict this
project already fought once for `torch`/`onnxruntime-gpu` on serra1's CUDA
12.5 ceiling, not a hypothetical one.

**Real, verified facts that correct assumptions made before reading the
source** (see `NOTICE.md` and `research/papers/diffusion/rw-hadera-blanket.md`,
the latter still needs a correction pass):

- Checkpoint is **SDXL inpainting**
  (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) + an SDXL refiner,
  896x896 — not "Realistic Vision"/SD1.5/512.
- **Two ControlNets** (openpose + canny), not one.
- `FaceFusionDirectAnonymizer.anonymize(image, detections)`'s `detections`
  argument is **dead code, never read** in the real source — per-frame face
  redetection always happens internally via FaceFusion's own `yolo_face`.
  This project's own Phase 1 detections cannot be substituted there without
  patching BLANKET's source, which this backend deliberately does not do.
- `synthetic_face_path` (the identity-seed image path) IS a clean,
  unmodified substitution point — this project's own context-padded crop
  feeds it, same as it already feeds `ciagan`/`ganonymization`.
- **Real upstream bug, worked around here, not in their source**: BLANKET's
  own `stable_diffusion_parameters.yaml` hardcodes `seed: 1`, and
  `StableDiffusionAnonymizer.generate()` has no per-call seed argument — the
  external `identity_server.py` (in the bridge repo) must set
  `anonymizer.seed = <this project's Identity.seed>` as a public attribute
  before calling `.generate()`, or every track in a video gets the same
  diffusion RNG regardless of this project's own per-track seed.

No dlib/mediapipe/torch import happens in *this* module — everything heavy
runs inside the two external server processes; this file only does
`subprocess`/`socket`/`json`/`tempfile` plumbing plus this project's own
already-vendored head-segmentation compositing (opt-in, see `refine_mask`).
"""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .._compositing import poisson_composite
from .._segmentation import load_or_random_head_segmentation, predict_head_mask, resolve_torch_device

logger = logging.getLogger(__name__)

_CONNECT_RETRY_INTERVAL = 0.5  # seconds between socket-connect attempts while a server boots


class _ExternalServiceError(RuntimeError):
    """Raised on an explicit `{"error": ...}` RPC reply, a timeout, or a dead process.

    Deliberately not silent — matches this project's existing convention
    (see `ganonymization/backend.py`) of raising a clear, specific error
    instead of returning `None` for anything other than a legitimate "no
    face found" result.
    """


class _RpcClient:
    """Thin Unix-socket JSON-line client for one external server process.

    Protocol (shared with `identity_server.py`/`swap_server.py` in the
    sibling `blanket-anonymizer-bridge` repo, defined and controlled by this
    project, not part of BLANKET's own code): one JSON object per line in
    each direction. `{"op": ..., **kwargs}` -> `{"result": <value-or-null>}`
    or `{"error": "<message>"}`. `null` result means a legitimate "nothing
    found" (e.g. no face detected), not a failure — only `error` raises.
    """

    def __init__(self, socket_path: Path, python: Path, server_script: Path,
                 cwd: Path, extra_args: list[str], timeout: float, label: str):
        self.socket_path = socket_path
        self.python = python
        self.server_script = server_script
        self.cwd = cwd
        self.extra_args = extra_args
        self.timeout = timeout
        self.label = label
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None

    def ensure_started(self) -> None:
        if self._sock is not None:
            return
        if self.socket_path.exists():
            self.socket_path.unlink()
        if not self.python.is_file():
            raise _ExternalServiceError(
                f"{self.label}: python interpreter not found at {self.python} — "
                "expected a venv inside the blanket-anonymizer-bridge checkout "
                "(see models/blanket/NOTICE.md for setup)."
            )
        if not self.server_script.is_file():
            raise _ExternalServiceError(
                f"{self.label}: server script not found at {self.server_script}."
            )
        cmd = [str(self.python), str(self.server_script), "--socket", str(self.socket_path), *self.extra_args]
        logger.info(f"{self.label}: starting external server ({' '.join(cmd)})")
        self._proc = subprocess.Popen(cmd, cwd=str(self.cwd))
        atexit.register(self._shutdown)

        deadline = time.monotonic() + self.timeout
        last_error = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise _ExternalServiceError(
                    f"{self.label}: server process exited early (code {self._proc.returncode}) "
                    "before it ever accepted a connection — check its own stdout/stderr."
                )
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self.socket_path))
                self._sock = sock
                logger.info(f"{self.label}: connected")
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
                last_error = e
                time.sleep(_CONNECT_RETRY_INTERVAL)
        raise _ExternalServiceError(
            f"{self.label}: timed out after {self.timeout}s waiting for the server to accept "
            f"a connection at {self.socket_path} (last error: {last_error})"
        )

    def call(self, op: str, **kwargs) -> Optional[str]:
        self.ensure_started()
        assert self._sock is not None
        request = json.dumps({"op": op, **kwargs}) + "\n"
        try:
            self._sock.sendall(request.encode("utf-8"))
            buf = b""
            self._sock.settimeout(self.timeout)
            while not buf.endswith(b"\n"):
                chunk = self._sock.recv(65536)
                if not chunk:
                    raise _ExternalServiceError(f"{self.label}: server closed the connection mid-reply")
                buf += chunk
        except (OSError, socket.timeout) as e:
            raise _ExternalServiceError(f"{self.label}: RPC '{op}' failed ({e})") from e

        reply = json.loads(buf.decode("utf-8"))
        if "error" in reply:
            raise _ExternalServiceError(f"{self.label}: RPC '{op}' returned an error: {reply['error']}")
        return reply.get("result")

    def _shutdown(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class Backend:
    """BLANKET via two isolated external processes (`IdentityGenerator`, `FaceSwapper`).

    Unlike `ciagan`/`ganonymization`, there is no single local checkpoint
    file this project manages directly — `weights`/`ctx_id` are accepted
    only for `FaceGenerator`'s uniform construction contract and are not
    otherwise used (BLANKET's own checkpoints are resolved entirely inside
    the two external venvs, outside this project's `CONFIG.weights_dir`).
    """

    def __init__(
        self,
        weights: Optional[Path] = None,
        ctx_id: int = 0,
        random_init: bool = False,
        bridge_repo: Optional[Path] = None,
        identity_python: Optional[Path] = None,
        swap_python: Optional[Path] = None,
        identity_socket: Optional[Path] = None,
        swap_socket: Optional[Path] = None,
        # Covers the whole per-call round trip (see _RpcClient.call()), not
        # just server startup — a cold SDXL+2 ControlNets+refiner load plus
        # generation plus refinement exceeded the original 180s default and
        # triggered a real client-side timeout on serra1's shared GPU
        # (verified 2026-09-24, see NOTICE.md's calibration log).
        server_timeout: float = 900.0,
        refine_mask: bool = True,
        segmentation_weights: Optional[Path] = None,
    ):
        del weights  # accepted only for FaceGenerator's uniform construction contract, unused here
        self.ctx_id = ctx_id
        self.random_init = random_init
        self.bridge_repo = Path(bridge_repo) if bridge_repo else None
        self.server_timeout = server_timeout
        self.refine_mask = refine_mask
        self.segmentation_weights = segmentation_weights

        self._scratch_dir = Path(tempfile.mkdtemp(prefix="blanket_ipc_"))
        atexit.register(shutil.rmtree, self._scratch_dir, ignore_errors=True)

        identity_python = Path(identity_python) if identity_python else (
            self.bridge_repo / ".venv-identity" / "bin" / "python" if self.bridge_repo else None
        )
        swap_python = Path(swap_python) if swap_python else (
            self.bridge_repo / ".venv-swap" / "bin" / "python" if self.bridge_repo else None
        )
        identity_socket = Path(identity_socket) if identity_socket else self._scratch_dir / "identity.sock"
        swap_socket = Path(swap_socket) if swap_socket else self._scratch_dir / "swap.sock"

        if not self.random_init:
            if self.bridge_repo is None or identity_python is None or swap_python is None:
                raise FileNotFoundError(
                    "blanket backend needs --bridge-repo (checkout of blanket-anonymizer-bridge, "
                    "with .venv-identity/.venv-swap already set up) unless --random-init is set. "
                    "See models/blanket/NOTICE.md."
                )
            self._identity_client = _RpcClient(
                socket_path=identity_socket, python=identity_python,
                server_script=self.bridge_repo / "identity_server.py", cwd=self.bridge_repo,
                extra_args=[], timeout=self.server_timeout, label="blanket/identity",
            )
            self._swap_client = _RpcClient(
                socket_path=swap_socket, python=swap_python,
                server_script=self.bridge_repo / "swap_server.py", cwd=self.bridge_repo,
                extra_args=[], timeout=self.server_timeout, label="blanket/swap",
            )
        else:
            self._identity_client = None
            self._swap_client = None
            logger.warning(
                "blanket backend running with random_init=True: output is NOT real "
                "anonymization (identity crop is echoed back) — smoke-test only."
            )

        self._identity_cache: dict[int, str] = {}  # seed -> identity image path on serra1's local disk
        self._seg_model = None
        self._seg_resolution = None
        self._device = None

    def identity_class(self, seed: int) -> int:
        """Not a real class index (no fixed identity vocabulary) — `seed` also IS
        the actual diffusion RNG seed passed to BLANKET's own generator (see the
        `seed: 1` workaround in the module docstring), so unlike GANonymization's
        purely-cosmetic `identity_class()`, this one has real effect upstream,
        just no discrete "class space" to report. Returned unchanged for logging.
        """
        return seed

    def _load_segmentation(self) -> None:
        if self._seg_model is not None:
            return
        self._device = resolve_torch_device(self.ctx_id)
        self._seg_model, self._seg_resolution = load_or_random_head_segmentation(
            self.segmentation_weights, self.random_init, self._device,
        )

    def _write_png(self, image: np.ndarray, name: str) -> Path:
        path = self._scratch_dir / name
        cv2.imwrite(str(path), image)
        return path

    def _seed_identity_path(self, crop: np.ndarray, seed: int) -> Optional[str]:
        cached = self._identity_cache.get(seed)
        if cached is not None:
            return cached
        if self.random_init:
            path = str(self._write_png(crop, f"seed_{seed}_random.png"))
            self._identity_cache[seed] = path
            return path

        crop_path = self._write_png(crop, f"seed_{seed}_input.png")
        identity_path = self._identity_client.call("generate", crop_path=str(crop_path), seed=seed)
        if identity_path is None:
            # BLANKET's own YOLO+SPIGA detection found no usable face in this
            # particular frame's crop — legitimate, not an error. Deliberately
            # NOT cached: the next frame of the same track gets a fresh
            # attempt, since a later frame may have a clearer face.
            return None
        self._identity_cache[seed] = identity_path
        return identity_path

    def generate(self, crop: np.ndarray, seed: int) -> Optional[np.ndarray]:
        """Anonymize the face in `crop` (BGR uint8, context-padded box already
        cut by `run.py`). Returns a same-shape/dtype BGR image, or `None` if
        no usable face could be found (caller falls back to passthrough).
        """
        identity_path = self._seed_identity_path(crop, seed)
        if identity_path is None:
            return None

        if self.random_init:
            swapped = crop.copy()
        else:
            crop_path = self._write_png(crop, "frame_in.png")
            swapped_path = self._swap_client.call("swap", identity_path=identity_path, crop_path=str(crop_path))
            if swapped_path is None:
                # FaceFusion's own internal yolo_face redetection found nothing
                # in this crop — legitimate "skip this frame", same contract as
                # ciagan/ganonymization's own no-landmarks case.
                return None
            swapped = cv2.imread(str(swapped_path))
            if swapped is None:
                raise _ExternalServiceError(
                    f"blanket/swap: RPC reported success but {swapped_path} is unreadable"
                )

        if not self.refine_mask:
            return swapped

        # Explicit seam, entirely this project's own code: BLANKET's own
        # convex-hull/inswapper footprint mask never covers hair/silhouette
        # (its documented weak point — 2.5/5 perceived de-identification).
        # Reusing this project's already-calibrated full-head segmentation
        # model for the FINAL composite is a real, testable improvement over
        # upstream's own compositing, not a claim upstream itself makes.
        self._load_segmentation()
        mask = predict_head_mask(
            self._seg_model, self._seg_resolution, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), self._device,
        )
        return poisson_composite(crop, swapped, mask)
