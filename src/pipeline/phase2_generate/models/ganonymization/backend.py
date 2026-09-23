"""GANonymization backend — landmark-conditioned pix2pix re-synthesis with
full-head segmentation for compositing (Hellmann et al., ACM TOMM 2024,
DOI 10.1145/3641107, arXiv:2305.02143).

Architecture verified directly against the paper PDF (Fig. 2, Sec. 3) and
the official repo (`hcmlab/GANonymization`, pinned commit
`7dbe45e3b172670ff59be982c662749165ab5ced`) — not reconstructed from a
secondhand summary. Real inference pipeline, confirmed from
`main.py::anonymize_image` and `lib/transform/*.py`:
`FaceCrop(align) -> ZeroPaddingResize(512) -> FacialLandmarks478 -> Pix2PixTransformer`
— face segmentation is training-only, never called at inference.

This backend deliberately diverges from that pipeline in one place, per a
project-level decision (see `../../../../../research/next-steps/70-decisions.md`
and `NOTICE.md`): it does **not** add RetinaFace/`retina-face`/`deepface`
(which would pull in TensorFlow as a third deep-learning runtime and
reintroduce the CUDA-wheel-conflict class already fought for
`onnxruntime-gpu`/`torch`). Instead it runs MediaPipe FaceMesh directly on
the context-padded crop `run.py` already cuts — the same pattern
`../ciagan/backend.py` already uses for dlib. Face segmentation, skipped by
upstream at inference, is instead used here for its opposite intended
purpose: producing this backend's own paste-back compositing mask (the
entire reason this backend exists — see NOTICE.md's compliance/coverage
notes).

Two non-obvious things reproduced/adapted from upstream, easy to get
quietly wrong:

1. **The landmark canvas is dots, not connected lines.** Verified from
   `lib/transform/facial_landmarks_478_transformer.py`:
   `mediapipe.solutions.drawing_utils.draw_landmarks(..., connections=None)`
   — mediapipe's own `draw_landmarks` gates all line-drawing on
   `connections`, so passing `None` draws only small dots at each of the
   478 points, never a connected mesh. `_landmark_canvas()` below
   reproduces dots (not an exact pixel-for-pixel rendering of mediapipe's
   own circle-drawing routine — flagged for a visual check during real
   calibration, see NOTICE.md).
2. **Landmarks are detected twice, not rescaled once.** Matches the real
   upstream ordering (`FaceCrop -> ZeroPaddingResize -> FacialLandmarks478`
   — landmarks are extracted from the *final* letterboxed 512x512 canvas,
   not detected once on the original crop and algebraically rescaled).
   `generate()` below does the same: an optional first pass on the raw
   crop exists only to estimate a leveling rotation (see below), a second,
   independent pass on the rotated+letterboxed canvas is what actually
   feeds the generator.

**Rotation correction exists but defaults to OFF (`align_rotation=False`).**
Upstream's real `FaceCrop(align=True)` wraps RetinaFace's own
`alignment_procedure`, which *does* do eye-based in-plane rotation
correction before the model ever sees the image (verified against
`retinaface/commons/postprocess.py`), so skipping RetinaFace per this
project's own decision does mean starting out *missing* something
upstream's own pipeline has, in principle. In practice, real-video
calibration found the opposite of a strong prior: rotating in-place within
a fixed-size canvas (`cv2.warpAffine(..., borderMode=cv2.BORDER_REPLICATE)`)
can push real facial content (chin, forehead, ear) outside the original
crop bounds and backfill the gap with replicated edge pixels — confirmed
causing MediaPipe's second detection pass to fail on a real large/close-up
face where the *unrotated* canvas succeeded. Since `generate()`'s automatic
no-rotation fallback already recovers any case where rotation fails, this
means `align_rotation=True` was pure downside (wasted compute, inconsistent
per-frame treatment) with no confirmed upside — see NOTICE.md's calibration
log for the real test. Left in as an opt-in (`--align-rotation`), not
deleted, in case a future fix (e.g. rotating within a padded, larger
canvas) makes it a net positive.

**Checkerboard/blur tendency is architectural, not a bug here.** The
vendored `GeneratorUNet` (`vendor/pix2pix_generator.py`) is upstream's
stock pix2pix U-Net: `UNetUp` stacks 7 `nn.ConvTranspose2d(kernel=4,
stride=2)` upsampling stages, and only the *final* stage uses the
resize-convolution fix known to reduce transposed-conv checkerboarding
(Odena et al., "Deconvolution and Checkerboard Artifacts", Distill 2016).
Kernel divisible by stride reduces but doesn't eliminate the artifact, and
it can compound across stacked layers. Real-video testing on this
project's footage shows a visible grid pattern with the real face still
recognizable beneath it — direct high-resolution inspection of the paper's
own published example figures shows opaque results with no comparable
severity, so this project's small/blurry real-world crops and imperfect
alignment (vs. RetinaFace's precise crop) are likely *amplifying* a milder
inherent tendency, not that this severity is unavoidable. Fixing the
tendency itself would need retraining or post-hoc super-resolution — out
of scope; `blend_mode`/`sharpen_generated` below make compositing more
*robust to* this quality ceiling, they don't remove it.

**No identity-conditioning mechanism exists.** Unlike CIAGAN's 1200-class
one-hot identity vector, pix2pix here is a deterministic
landmark-dots -> face map with no seed/noise input. This project's
`Identity.seed`-driven "same track always maps to the same synthetic
identity" guarantee is architecturally impossible to replicate — see
`identity_class()`.

**Checkpoint format note (both checkpoints, generator and segmentation):**
unlike CIAGAN's plain `state_dict`-only `.pth` file, both of this backend's
checkpoints are full PyTorch Lightning checkpoints (`state_dict` plus a
`hyper_parameters` dict, plus optimizer state for the generator one) —
`torch.load(..., weights_only=True)` (torch's default since 2.6, reachable
under this project's `torch<2.7` pin) cannot unpickle that. Both loaders
below pass `weights_only=False` explicitly. This project controls the
provenance of both files (downloaded once, sha256-recorded in NOTICE.md),
so that trust boundary is the same one CIAGAN's checkpoint already accepts,
not a new relaxation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import numpy as np

from .._compositing import feathered_alpha_composite, poisson_composite
from .._segmentation import load_head_segmentation, predict_head_mask

logger = logging.getLogger(__name__)

#: MediaPipe FaceMesh 478-point scheme — outer eye corners, verified real
#: indices against `mediapipe/python/solutions/face_mesh_connections.py`'s
#: FACEMESH_RIGHT_EYE/FACEMESH_LEFT_EYE connection sets.
_RIGHT_EYE_OUTER = 33
_LEFT_EYE_OUTER = 263

_IMG_INTERP = cv2.INTER_LANCZOS4


class _LetterboxGeometry(NamedTuple):
    """Enough to invert `_letterbox_resize()` exactly."""
    new_w: int
    new_h: int
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


def _letterbox_resize(image: np.ndarray, size: int) -> tuple[np.ndarray, _LetterboxGeometry]:
    """Aspect-preserving resize + centered zero-pad to `size` x `size`.

    Reimplements upstream's `ZeroPaddingResize` (paper Sec. 3.1: resize to
    `size` pixels on the greater axis, then zero-pad the shorter axis on
    both sides to keep the face centered), but ALSO returns the geometry
    needed to invert it — upstream never needed an inverse (it only ever
    saves the 512x512 result to disk); this backend pastes back into the
    source frame, so it does.
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=_IMG_INTERP)

    canvas = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, _LetterboxGeometry(new_w, new_h, pad_x, pad_y, w, h)


def _letterbox_unresize(image: np.ndarray, geom: _LetterboxGeometry, interpolation: int) -> np.ndarray:
    """Invert `_letterbox_resize()`: crop out the padding, resize back to the original (w, h)."""
    cropped = image[geom.pad_y:geom.pad_y + geom.new_h, geom.pad_x:geom.pad_x + geom.new_w]
    return cv2.resize(cropped, (geom.orig_w, geom.orig_h), interpolation=interpolation)


def _landmark_canvas(points: np.ndarray, size: int) -> np.ndarray:
    """Black `size`x`size`x3 canvas with a white ~1px dot at each point.

    NOT connected by lines — see module docstring point 1. Achromatic
    (equal R=G=B), so unlike ciagan's landmark canvas there's no BGR/RGB
    channel-order quirk to reproduce here.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            cv2.circle(canvas, (xi, yi), radius=1, color=(255, 255, 255), thickness=-1)
    return canvas


def _rotation_matrix(points: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """2x3 affine leveling the eye line (`_RIGHT_EYE_OUTER`/`_LEFT_EYE_OUTER`), about `center`.

    `theta = atan2(dy, dx)` is the eye-line's tilt from horizontal in
    image coordinates (y down). Leveling it means rotating the image
    content by the standard-math angle `-theta`; working through
    `cv2.getRotationMatrix2D`'s own convention (`x' = x*cos(angle) +
    y*sin(angle)`, `y' = -x*sin(angle) + y*cos(angle)`, i.e. a
    standard-math rotation by `-angle`) shows that requires passing
    `angle = theta` (in degrees) directly, not `-theta`, to
    `getRotationMatrix2D`. Derived algebraically, not fitted — but
    direction/sign conventions like this are exactly the kind of thing
    this project has gotten backwards before on a first pass (see
    ciagan/NOTICE.md's rotation-correction entry); treat this as a strong
    prior to verify visually against a real tilted-head frame, not an
    unquestionable derivation.
    """
    right, left = points[_RIGHT_EYE_OUTER], points[_LEFT_EYE_OUTER]
    dx, dy = float(left[0] - right[0]), float(left[1] - right[1])
    theta_deg = float(np.degrees(np.arctan2(dy, dx)))
    return cv2.getRotationMatrix2D(center, theta_deg, 1.0)


def _enhance_for_detection(image_rgb: np.ndarray) -> np.ndarray:
    """CLAHE-boosted copy of `image_rgb`, for feeding MediaPipe detection only.

    Used only when `enhance_detection_input=True` (see `Backend`), and only
    for the detection call that feeds the generator — never for
    `predict_head_mask()`'s input, so a detection-quality experiment can't
    get confounded with a segmentation-quality change. Since the dot canvas
    is redrawn from scratch from the detected *points* (not derived from
    pixel values), this contrast boost never reaches the generator's actual
    input distribution — it only affects whether detection succeeds.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def _sharpen(image: np.ndarray, amount: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    """Unsharp-mask sharpening of the generator's raw output.

    Tested as a way to give `poisson_composite()`'s `cv2.seamlessClone` real
    gradients to lock onto when the generated content has weak local
    contrast (see NOTICE.md's anti-bleed-through entry). Named risk, not
    assumed away: sharpening amplifies ANY high-frequency content, including
    this architecture's own checkerboard tendency (see module docstring's
    "Checkerboard tendency" note) — judge visually side-by-side against the
    unsharpened result, never accept solely because it reduces see-through.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(image.dtype)


class Backend:
    """GANonymization via vendored `GeneratorUNet` + `HeadSegmentationModel`.

    Like ciagan, keeps both vendored architectures loaded natively in
    PyTorch at inference time (no ONNX conversion attempted).
    """

    def __init__(
        self,
        weights: Optional[Path] = None,
        ctx_id: int = 0,
        segmentation_weights: Optional[Path] = None,
        img_size: int = 512,
        align_rotation: bool = False,
        random_init: bool = False,
        blend_mode: str = "poisson",
        sharpen_generated: bool = False,
        min_detection_confidence: float = 0.5,
        enhance_detection_input: bool = False,
    ):
        if img_size != 512:
            # The paper's Face Extraction step and this backend's own
            # letterbox/rotation geometry are both derived for 512 — not
            # a free knob the way it isn't for ciagan's 128 either.
            raise NotImplementedError(
                "ganonymization backend only supports img_size=512 (the "
                "released checkpoints and this backend's crop geometry are "
                "both derived specifically for that size)"
            )
        if blend_mode not in ("poisson", "feather"):
            raise ValueError(f"blend_mode must be 'poisson' or 'feather', got {blend_mode!r}")
        self.weights = weights
        self.ctx_id = ctx_id
        self.segmentation_weights = segmentation_weights
        self.img_size = img_size
        self.align_rotation = align_rotation
        self.random_init = random_init
        # Opt-in anti-transparency knobs (see module docstring's
        # "Checkerboard/blur tendency" note and NOTICE.md's calibration
        # log) — default to today's already-shipped behavior (poisson,
        # unsharpened, 0.5 confidence, no contrast boost) until each is
        # validated against real footage.
        self.blend_mode = blend_mode
        self.sharpen_generated = sharpen_generated
        self.min_detection_confidence = min_detection_confidence
        self.enhance_detection_input = enhance_detection_input

        self._model = None  # lazy load
        self._seg_model = None
        self._seg_resolution = None
        self._face_mesh = None
        self._device = None

    def identity_class(self, seed: int) -> int:
        """NOT a real identity-class index — GANonymization has no
        identity-conditioning mechanism at all (see module docstring).
        Returns `seed` unchanged, purely so generation.jsonl/
        run_manifest.json have something to log per the Backend contract.
        """
        return seed

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "mediapipe is required for the ganonymization backend (478-point "
                "FaceMesh landmark extraction). Install with: "
                "uv sync --extra phase2-ganonymization"
            ) from e
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for the ganonymization backend. Install with: "
                "uv sync --extra phase2-ganonymization"
            ) from e

        if self.ctx_id < 0:
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self._device = device

        # Matches GANonymization's own FaceMesh construction
        # (lib/transform/facial_landmarks_478_transformer.py), verified.
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=self.min_detection_confidence,
        )

        from .vendor.pix2pix_generator import GeneratorUNet
        model = GeneratorUNet()

        from .._vendor.head_segmentation_model import HeadSegmentationModel

        if self.random_init:
            logger.warning(
                "ganonymization backend running with random_init=True: output is "
                "NOT real anonymization — smoke-test only, do not use for a real run."
            )
            seg_model = HeadSegmentationModel(
                encoder_name="resnet18", encoder_depth=5,
                pretrained=False, nn_image_input_resolution=self.img_size,
            )
            seg_resolution = self.img_size
        else:
            if self.weights is None or not Path(self.weights).is_file():
                raise FileNotFoundError(
                    f"GANonymization generator checkpoint not found: {self.weights}\n"
                    "See models/ganonymization/NOTICE.md for the checkpoint link "
                    "and its caveats, or pass --random-init for a smoke test "
                    "(not real anonymization)."
                )
            if self.segmentation_weights is None or not Path(self.segmentation_weights).is_file():
                raise FileNotFoundError(
                    f"head-segmentation checkpoint not found: {self.segmentation_weights}\n"
                    "See models/ganonymization/NOTICE.md for how to obtain it, "
                    "or pass --random-init for a smoke test (not real anonymization)."
                )

            weights_path = Path(self.weights)
            sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            logger.info(f"Loading GANonymization generator from {weights_path} (sha256={sha256})")
            # Lightning checkpoint, not a bare state_dict — see module docstring.
            ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
            raw_state_dict = ckpt["state_dict"]
            generator_state_dict = {
                k[len("generator."):]: v for k, v in raw_state_dict.items() if k.startswith("generator.")
            }
            if not generator_state_dict:
                prefixes = sorted({k.split(".")[0] for k in raw_state_dict})
                raise RuntimeError(
                    "no 'generator.*' keys found in the pix2pix checkpoint's state_dict "
                    f"(top-level key prefixes found: {prefixes}) — this backend's "
                    "'generator.' prefix assumption (Pix2Pix's own `self.generator = "
                    "GeneratorUNet()` attribute name) doesn't match this checkpoint. "
                    "See NOTICE.md's checkpoint-status section."
                )
            model.load_state_dict(generator_state_dict, strict=True)
            logger.info(f"ganonymization generator: {len(generator_state_dict)} keys loaded, strict=True")

            seg_weights_path = Path(self.segmentation_weights)
            seg_sha256 = hashlib.sha256(seg_weights_path.read_bytes()).hexdigest()
            logger.info(f"Loading head-segmentation model from {seg_weights_path} (sha256={seg_sha256})")
            seg_model, seg_resolution = load_head_segmentation(seg_weights_path, device)

        model.eval()
        model.to(device)
        seg_model.eval()
        seg_model.to(device)
        self._model = model
        self._seg_model = seg_model
        self._seg_resolution = seg_resolution

    def _facemesh_points(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """478x2 pixel-space points from `image_rgb`, or None if no face found."""
        results = self._face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        h, w = image_rgb.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)

    def _run_generator(self, dot_canvas: np.ndarray):
        import torch
        tensor = torch.from_numpy(dot_canvas.astype(np.float32) / 255.0).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5  # -> [-1, 1], matches upstream's Normalize(0.5,0.5,0.5)
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(tensor)[0]
        out = (torch.clamp(out, -1, 1) + 1) / 2  # -> [0, 1]
        return (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # RGB

    def _try_generate(self, crop: np.ndarray, use_rotation: bool) -> Optional[np.ndarray]:
        h, w = crop.shape[:2]
        rot_m_inv = None
        if use_rotation:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            points = self._facemesh_points(crop_rgb)
            if points is None:
                return None
            rot_m = _rotation_matrix(points, (w / 2.0, h / 2.0))
            working = cv2.warpAffine(crop, rot_m, (w, h), flags=_IMG_INTERP, borderMode=cv2.BORDER_REPLICATE)
            rot_m_inv = cv2.invertAffineTransform(rot_m)
        else:
            working = crop

        letterboxed, geom = _letterbox_resize(working, self.img_size)
        letterboxed_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        # Second, independent detection pass on the FINAL canvas — matches
        # the real upstream ordering, not a rescale of the first pass's
        # points (see module docstring point 2). Optionally fed a
        # contrast-boosted copy (detection only — predict_head_mask() below
        # still gets the unmodified image, see _enhance_for_detection()).
        detection_input = _enhance_for_detection(letterboxed_rgb) if self.enhance_detection_input else letterboxed_rgb
        points512 = self._facemesh_points(detection_input)
        if points512 is None:
            return None

        dot_canvas = _landmark_canvas(points512, self.img_size)
        generated512 = self._run_generator(dot_canvas)
        if self.sharpen_generated:
            generated512 = _sharpen(generated512)
        mask512 = predict_head_mask(self._seg_model, self._seg_resolution, letterboxed_rgb, self._device)

        gen_working = _letterbox_unresize(generated512, geom, _IMG_INTERP)
        mask_working = _letterbox_unresize(mask512.astype(np.float32), geom, cv2.INTER_LINEAR) > 0.5

        if rot_m_inv is not None:
            gen_full_rgb = cv2.warpAffine(gen_working, rot_m_inv, (w, h), flags=_IMG_INTERP,
                                           borderMode=cv2.BORDER_REPLICATE)
            mask_full = cv2.warpAffine(mask_working.astype(np.float32), rot_m_inv, (w, h),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0.0) > 0.5
        else:
            gen_full_rgb, mask_full = gen_working, mask_working

        gen_full_bgr = cv2.cvtColor(gen_full_rgb, cv2.COLOR_RGB2BGR)
        composite_fn = poisson_composite if self.blend_mode == "poisson" else feathered_alpha_composite
        return composite_fn(crop, gen_full_bgr, mask_full)

    def generate(self, crop: np.ndarray, seed: int) -> Optional[np.ndarray]:
        """Anonymize the face in `crop` (BGR uint8). `seed` is accepted for
        Backend-contract compatibility but has no effect — see
        `identity_class()`.

        Returns a same-shape/dtype BGR image, or None if no usable
        landmarks could be extracted on either detection pass (caller
        decides the fallback).
        """
        self._load()
        out = self._try_generate(crop, use_rotation=self.align_rotation)
        if out is None and self.align_rotation:
            # Rotation itself may have pushed the face into an
            # undetectable pose for a borderline case — retry once
            # unrotated before giving up. See NOTICE.md for how often
            # this actually triggers on real footage.
            out = self._try_generate(crop, use_rotation=False)
        return out
