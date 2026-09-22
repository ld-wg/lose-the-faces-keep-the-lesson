"""CIAGAN backend — conditional-identity face anonymization (CVPR 2020).

Generator input/output shape and the composition formula below are verified
directly against `vendor/arch_unet_flex.py` and the pinned upstream
`test.py`/`process_data.py`/`util_data.py` (see `../NOTICE.md`), not
reconstructed from the paper alone.

Two upstream details this backend must reproduce exactly to match the
released checkpoint's training distribution — both easy to "fix" by
accident, so flagged here:

1. **Landmark line-art keeps its channel order as drawn, unswapped.**
   Upstream draws lines with `cv2.line(canvas, ..., (0, 0, 255), ...)` (a
   BGR-style red) directly into a numpy array, then hands that array to
   `PIL.Image.fromarray()` with no `cv2.cvtColor` — PIL treats the array's
   axis order as-is (R, G, B), so the "red" line is actually saved as blue.
   The network trained on that exact (technically mislabeled) channel
   order. `_landmark_canvas()` below reproduces it by never converting.
   The real face crop *is* converted (`cv2.imwrite`'s BGR write + PIL's RGB
   read round-trip correctly in the original data pipeline), so only that
   branch gets `cv2.cvtColor(BGR2RGB)`.

2. **dlib runs on the raw BGR array, not RGB.** Upstream's
   `process_data.py` passes `cv2.imread()`'s BGR output straight to
   `dlib.get_frontal_face_detector()`/`shape_predictor` with no conversion.
   Reproduced here for the same reason as (1): matching what actually
   produced the training landmarks matters more than matching dlib's
   documented convention.

Everything else here is a from-scratch reimplementation (not vendored) of
upstream's two-stage per-image geometry (`process_data.py`'s dlib-crop
stage + `util_data.py`'s resize/center-crop stage), fused into a single
affine transform derived by composing both stages' scale/translate math,
plus a rotation-correction term upstream itself never had — see
`_face_transform()`'s docstring for the derivation and why. Operating on
this project's own in-memory crop instead of round-tripping through disk.

**Crop size is not a free choice — it measurably affects output quality,
not just framing.** `run.py` cuts its own crop directly from the source
video around the detected box, sized via `models.DEFAULT_CONTEXT_RATIO` —
counterintuitively, a *tighter* crop than you'd expect from padding alone,
because a looser one degrades dlib's own detected landmarks enough to
visibly increase generation noise (see that dict's comment and
`../NOTICE.md` for the full empirical finding). `portrait_scale` and
`context_ratio` were calibrated together against real footage; re-check
both if you change either.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Upstream's `process_data.py` aligns crops to this aspect ratio (CelebA's
#: own aligned-crop convention) before its own second resize/crop stage.
_ALIGN_W, _ALIGN_H = 178.0, 218.0

#: dlib 68-point (iBUG) scheme indices used by upstream's line-art/mask —
#: jaw contour, nose bridge, inner mouth. Not the full 68 points.
_JAW = list(range(0, 17))
_NOSE_BRIDGE = list(range(27, 31))
_INNER_MOUTH = list(range(60, 68))


def _face_transform(points: np.ndarray, portrait_scale: float = 1.0) -> Optional[np.ndarray]:
    """The composed affine mapping crop-space landmarks -> 128x128 network space.

    Derived by composing upstream's two independent stages symbolically
    (both are pure per-axis scale+translate in the original, no rotation —
    see "Rotation correction" below for why this version adds one):

    Stage A (`process_data.py`): crop centered on the midpoint of the inner
    eye corners (points 39, 42), radius set from their distance `dx`
    (`h_r = dx*5`, `w_r = h_r * 178/218`), resized to 178x218.

    Stage B (`util_data.py`'s `load_img`, test/non-augment path): resize
    178x218 -> 144x144 (anisotropic — this squashes the aspect ratio, kept
    faithfully since the released checkpoint was trained on it), then a
    fixed centered 128x128 crop (margin 8px each side), then a no-op
    128->128 resize.

    Composing both stages algebraically collapses to an anisotropic scale
    about (c_x, c_y) landing at the 128-canvas's center (64, 64) — verified
    by hand against both stages' source, not empirically fitted. Returns
    None if the eye points are degenerate (near-zero distance — a bad
    landmark fit, not a real face).

    `portrait_scale` multiplies `h_r`/`w_r` (default 1.0 = upstream's exact
    formula, and what this project's footage actually needs). Exists as a
    knob because `h_r = dx*5` bakes in CelebA's own aligned-photo
    convention, which isn't guaranteed to transfer to a different
    camera/framing regime — but the size of the generated region isn't a
    function of `portrait_scale` alone: the crop this is computed from
    matters too, since a looser crop measurably degrades dlib's own
    detected `dx` and the resulting output quality (see `../NOTICE.md` and
    `models/DEFAULT_CONTEXT_RATIO`'s comment for the full finding). Treat
    `portrait_scale` and `context_ratio` as a pair to re-calibrate
    together, empirically against real footage, not independently.

    **Rotation correction (added, not in upstream):** upstream's own
    `process_data.py` never corrects for in-plane head tilt — it only ever
    centers/scales, so a tilted head gets a perfectly upright generated
    face pasted back, visibly misaligned with the real head/neck beneath
    it. Standard face-alignment practice elsewhere (InsightFace/ArcFace's
    `face_align.py`, the FFHQ/StyleGAN dataset-prep script) always includes
    rotation as part of a similarity transform derived from eye landmarks —
    verified directly against both, not assumed. Added here as an angle
    computed from the eye-to-eye vector, applied before the anisotropic
    scale: this derotates the network's input toward upright (closer to
    the mostly-upright photos the checkpoint trained on) and, since the
    same matrix inverts for paste-back, correctly re-rotates the generated
    content to match the real head's actual tilt in the frame. Also
    switches `dx` from upstream's x-only inter-eye distance to the full
    Euclidean eye-to-eye distance — the x-only version is itself only
    correct for a near-upright face, the same gap rotation correction
    closes.
    """
    eye_dx = float(points[42, 0] - points[39, 0])
    eye_dy = float(points[42, 1] - points[39, 1])
    dx = float(np.hypot(eye_dx, eye_dy))
    if dx <= 1.0:
        return None
    h_r = dx * 5.0 * portrait_scale
    w_r = h_r * (_ALIGN_W / _ALIGN_H)
    if w_r <= 1.0 or h_r <= 1.0:
        return None
    c_x = float(points[42, 0] + points[39, 0]) / 2.0
    c_y = float(points[42, 1] + points[39, 1]) / 2.0
    sx, sy = 72.0 / w_r, 72.0 / h_r

    theta = float(np.arctan2(eye_dy, eye_dx))  # eye-line's tilt from horizontal
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotate by -theta (level the eye line), then apply the anisotropic
    # scale, composed into one 2x2 linear map:
    a00, a01 = sx * cos_t, sx * sin_t
    a10, a11 = -sy * sin_t, sy * cos_t
    tx = 64.0 - (a00 * c_x + a01 * c_y)
    ty = 64.0 - (a10 * c_x + a11 * c_y)
    return np.array([[a00, a01, tx], [a10, a11, ty]], dtype=np.float64)


def _transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply the full 2x3 affine `m` (rotation + scale + translation) to `points`."""
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    return homogeneous @ m.T


def _landmark_canvas(points128: np.ndarray) -> np.ndarray:
    """White 128x128x3 canvas with upstream's exact line set — see module docstring re: channel order."""
    canvas = np.full((128, 128, 3), 255, dtype=np.uint8)

    def line(i: int, j: int) -> None:
        p1 = tuple(np.round(points128[i]).astype(int))
        p2 = tuple(np.round(points128[j]).astype(int))
        cv2.line(canvas, p1, p2, (0, 0, 255), 1)

    for i in _JAW[:-1]:
        line(i, i + 1)
    for i in _NOSE_BRIDGE[:-1]:
        line(i, i + 1)
    for i in _INNER_MOUTH[:-1]:
        line(i, i + 1)
    line(67, 60)
    return canvas


def _poisson_composite(crop: np.ndarray, generated: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Gradient-domain blend of `generated` into `crop` within `mask_bool`.

    Replaces a hard-edged paste (a visible skin-tone/lighting seam at
    CIAGAN's jaw+eyebrow mask boundary — the model itself has no blending,
    see the module docstring) with `cv2.seamlessClone`: it solves for pixel
    values inside the mask whose *gradients* match `generated`'s, so the
    interior content is preserved but the boundary blends into the
    surrounding lighting/color instead of cutting sharply across it.

    Falls back to the hard-mask paste if `seamlessClone` raises — it needs
    a non-degenerate mask that doesn't touch the image border, which can
    happen for a face very close to the crop's edge. A cosmetic blending
    step failing shouldn't take down the whole frame.
    """
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return crop.copy()
    center = ((int(xs.min()) + int(xs.max())) // 2, (int(ys.min()) + int(ys.max())) // 2)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    try:
        return cv2.seamlessClone(generated, crop, mask_u8, center, cv2.NORMAL_CLONE)
    except cv2.error:
        output = crop.copy()
        output[mask_bool] = generated[mask_bool]
        return output


def _mask_canvas(points128: np.ndarray) -> np.ndarray:
    """128x128 float32 mask, 1.0 over the face+forehead polygon, else 0.0.

    Polygon: jaw contour (points 0-16) capped above at the eyebrow height
    (points 19/24) — same construction as upstream's `process_data.py`.
    """
    contour = np.concatenate([
        [[points128[0, 0], points128[19, 1]]],
        points128[_JAW],
        [[points128[16, 0], points128[24, 1]]],
    ]).astype(np.int32)
    mask = np.zeros((128, 128), dtype=np.float32)
    cv2.fillPoly(mask, [contour], 1.0)
    return mask


class Backend:
    """CIAGAN via a vendored PyTorch `Generator` (see `../NOTICE.md`).

    Unlike the Phase 1 ONNX backends, this one keeps `vendor/`'s
    architecture loaded natively at inference time — no verified ONNX
    exporter exists for its conditional (one-hot) forward pass. See
    `../NOTICE.md`'s "Design note" for why.
    """

    def __init__(
        self,
        weights: Optional[Path] = None,
        num_classes: int = 1200,
        img_size: int = 128,
        dlib_predictor: Optional[Path] = None,
        ctx_id: int = 0,
        random_init: bool = False,
        portrait_scale: float = 1.0,
    ):
        if img_size != 128:
            # The architecture itself branches on `img_size == 128` (extra
            # conv/deconv layers), and `_face_transform()`'s constants (72,
            # 64, 144, 8) are derived specifically for it — not a free knob.
            raise NotImplementedError(
                "ciagan backend only supports img_size=128 (the released "
                "checkpoint's architecture and this backend's crop geometry "
                "are both derived specifically for that size)"
            )
        self.weights = weights
        self.num_classes = num_classes
        self.img_size = img_size
        self.dlib_predictor = dlib_predictor
        self.ctx_id = ctx_id
        self.random_init = random_init
        self.portrait_scale = portrait_scale
        self._model = None  # lazy load
        self._dlib_detector = None
        self._dlib_predictor = None
        self._device = None

    def identity_class(self, seed: int) -> int:
        """Deterministic identity-vector index for `seed`. Collisions across
        tracks are expected (the checkpoint's identity space is a fixed,
        finite 1200 classes) — not a bug to fix here."""
        return seed % self.num_classes

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import dlib
        except ImportError as e:
            raise ImportError(
                "dlib is required for the ciagan backend (68-point landmark "
                "extraction). Install with: uv sync --extra phase2-ciagan"
            ) from e
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required for the ciagan backend. Install with: "
                "uv sync --extra phase2-ciagan"
            ) from e

        if self.dlib_predictor is None or not Path(self.dlib_predictor).is_file():
            raise FileNotFoundError(
                f"dlib shape predictor not found: {self.dlib_predictor}\n"
                "Download 'shape_predictor_68_face_landmarks.dat' from "
                "http://dlib.net/files/ (bz2-compressed — decompress before "
                "use) and pass its path via --dlib-predictor."
            )
        self._dlib_detector = dlib.get_frontal_face_detector()
        self._dlib_predictor = dlib.shape_predictor(str(self.dlib_predictor))

        if self.ctx_id < 0:
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self._device = device

        from .vendor.arch_unet_flex import Generator
        model = Generator(input_nc=6, num_classes=self.num_classes, img_size=self.img_size)

        if self.random_init:
            logger.warning(
                "ciagan backend running with random_init=True: output is NOT "
                "real anonymization — smoke-test only, do not use for a real run."
            )
        else:
            if self.weights is None or not Path(self.weights).is_file():
                raise FileNotFoundError(
                    f"CIAGAN generator weights not found: {self.weights}\n"
                    "See models/ciagan/NOTICE.md for the checkpoint link and "
                    "its caveats, or pass --random-init for a smoke test "
                    "(not real anonymization)."
                )
            weights_path = Path(self.weights)
            sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            logger.info(f"Loading CIAGAN generator from {weights_path} (sha256={sha256})")
            state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)

        model.eval()
        model.to(device)
        self._model = model

    def _landmarks68(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Run dlib on `crop` (BGR, see module docstring re: no RGB convert).

        Falls back to treating the whole crop as the face ROI if the HOG
        detector finds nothing — Phase 1 crops are already tightly bounded
        around one face, so a missed detection is more likely a hard case
        worth still trying than a true negative.
        """
        h, w = crop.shape[:2]
        dets = self._dlib_detector(crop, 1)
        if len(dets) > 0:
            rect = dets[0]
        else:
            import dlib
            rect = dlib.rectangle(0, 0, max(1, w - 1), max(1, h - 1))
        shape = self._dlib_predictor(crop, rect)
        return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float64)

    def generate(self, crop: np.ndarray, seed: int) -> Optional[np.ndarray]:
        """Anonymize the face in `crop` (BGR uint8), seeded by `seed`.

        `seed` should be `Identity.seed` (contracts.derive_seed) — same seed
        always maps to the same identity class (see `identity_class()`), so
        every frame of a track converges on the same synthetic identity.

        Returns a same-shape/dtype BGR image, unchanged outside the detected
        face region, or None if no usable 68-point landmarks could be
        extracted (degenerate crop — caller decides the fallback).
        """
        import torch

        self._load()
        points = self._landmarks68(crop)
        if points is None:
            return None
        m = _face_transform(points, self.portrait_scale)
        if m is None:
            return None

        points128 = _transform_points(points, m)
        lndm_canvas = _landmark_canvas(points128)  # NOT cvtColor'd — see module docstring
        mask128 = _mask_canvas(points128)

        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face128 = cv2.warpAffine(face_rgb, m, (128, 128), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE)

        device = self._device
        lndm_t = torch.from_numpy(lndm_canvas.astype(np.float32) / 255.0).permute(2, 0, 1)
        face_t = torch.from_numpy(face128.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask128).unsqueeze(0)  # (1,128,128), broadcasts over 3 channels
        input_gen = torch.cat([lndm_t, face_t * (1 - mask_t)], dim=0).unsqueeze(0).to(device)

        onehot = torch.zeros(1, self.num_classes, dtype=torch.float32, device=device)
        onehot[0, self.identity_class(seed)] = 1.0

        with torch.no_grad():
            gen_out = self._model(input_gen, onehot=onehot)[0]
        gen128 = (torch.clamp(gen_out, 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Composite at the ORIGINAL crop's resolution, not the 128 canvas:
        # only the mask region is ever overwritten, so background pixels
        # stay bit-identical to the input regardless of resampling — a
        # round-trip through the 128 canvas for the whole image would
        # introduce lossy resampling everywhere, not just inside the mask.
        h, w = crop.shape[:2]
        m_inv = cv2.invertAffineTransform(m)
        gen_full_rgb = cv2.warpAffine(gen128, m_inv, (w, h), flags=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_REPLICATE)
        mask_full = cv2.warpAffine(mask128, m_inv, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        mask_full_bool = mask_full > 0.5
        gen_full_bgr = cv2.cvtColor(gen_full_rgb, cv2.COLOR_RGB2BGR)

        return _poisson_composite(crop, gen_full_bgr, mask_full_bool)
