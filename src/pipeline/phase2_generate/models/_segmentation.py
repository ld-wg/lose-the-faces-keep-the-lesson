"""Shared head-segmentation inference for Phase 2 backends.

Used as ganonymization's primary compositing mask (full-head coverage —
the whole reason that backend exists) and, opt-in, as a mask-boundary
refinement for ciagan's own composite polygon (see
ciagan/backend.py's `refine_mask`).

Reimplements (not vendors) the pre/post-processing from
wiktorlazarski/head-segmentation's `segmentation_pipeline.py` /
`image_processing.py` (pinned commit
c1b4b9b4f12f168f13d603747b81e8f264bcf501 — see
`_vendor/LICENSE_head_segmentation`), around the verbatim-vendored
`HeadSegmentationModel` in `_vendor/head_segmentation_model.py`. Kept
separate from that vendored file, and not depending on the upstream
`head-segmentation` package's own `HumanHeadSegmentationPipeline` wrapper,
specifically to avoid that package's heavy, unpinned `requirements.txt`
(`gdown`, `loguru`, `streamlit`, `wandb`, `hydra-core`, `albumentations`,
...) for what is, in the actual inference path, about a dozen lines of
glue — see ganonymization/NOTICE.md for the full reasoning.

Two things verified directly against upstream source, easy to get quietly
wrong (a wrong-channel-order face often still "sort of" segments, so this
class of bug doesn't announce itself with a crash):

1. Input must be RGB, not BGR — upstream's own preprocessing hands the
   array straight to `PIL.Image.fromarray()` with no channel conversion.
2. Normalization is ImageNet mean/std (this model has a pretrained-encoder
   convention) — NOT the 0.5/0.5/0.5 this project's pix2pix generator
   uses. The two vendored models are not interchangeable here.

`load_head_segmentation()` deliberately does NOT call the vendored
`HeadSegmentationModel.load_from_checkpoint()` — see that method's own
docstring in `_vendor/head_segmentation_model.py` for why (a `weights_only`
default-flip landmine on torch>=2.6). This reimplements the identical
logic with `weights_only=False` explicit instead.

All of `torch`/`torchvision`/`PIL`/the vendored model are imported lazily,
inside the functions below, not at module level — this module is imported
unconditionally at the top of both `ciagan/backend.py` and
`ganonymization/backend.py`, and `ciagan`'s own `phase2-ciagan` extra does
NOT include `torchvision`/`segmentation-models-pytorch` (only
`phase2-ganonymization` does). A module-level import here would break
plain `--model ciagan` (even without `--refine-mask`) for anyone who only
installed `phase2-ciagan` — the same lazy-import discipline
`ciagan/backend.py` already applies to `dlib`/`torch` itself.

Also shared here (used by both backends' `_load()`, not segmentation-
specific, but small enough not to warrant a separate module): `hash_and_log()`
and `resolve_torch_device()` — the checkpoint-hashing and cuda/mps/cpu
device-selection logic every backend needs, previously copy-pasted at each
call site. And `load_or_random_head_segmentation()`, which folds the
"real checkpoint vs. --random-init smoke test" branch (identical in both
backends) into one place.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_detection_transforms: dict = {}  # resolution -> torchvision Compose, built once per resolution


def hash_and_log(path: Path, label: str) -> str:
    """sha256 of `path`, logged as "Loading {label} from {path} (sha256=...)".

    The one checkpoint-loading step genuinely identical across every model
    this package loads (ciagan's and ganonymization's own generators, and
    the head-segmentation model below) — previously copy-pasted at each
    call site instead of shared.
    """
    sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    logger.info(f"Loading {label} from {path} (sha256={sha256})")
    return sha256


def resolve_torch_device(ctx_id: int):
    """cuda > mps > cpu, forced cpu if `ctx_id < 0` — identical cascade both backends need."""
    import torch

    if ctx_id < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_head_segmentation(weights_path: Path, device) -> Tuple[object, int]:
    """Load the vendored model and its native input resolution.

    Both come from the same checkpoint file (a PyTorch Lightning
    checkpoint: a `state_dict` plus a `hyper_parameters` dict, not a bare
    tensor-only file like ciagan's `.pth`) — `weights_only=False` is
    required to unpickle the hyperparameters dict; this is this project's
    own downloaded, sha256-recorded copy, not arbitrary untrusted input,
    so that trust boundary is acceptable here the same way it already is
    for ciagan's checkpoint. Returns the model on CPU, un-`eval()`'d — see
    `load_or_random_head_segmentation()`, the caller both backends actually
    use, for the device placement/eval step shared with the random-init path.
    """
    import torch

    from ._vendor.head_segmentation_model import HeadSegmentationModel

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    resolution = hparams["nn_image_input_resolution"]

    model = HeadSegmentationModel(
        encoder_name=hparams["encoder_name"],
        encoder_depth=hparams["encoder_depth"],
        pretrained=False,
        nn_image_input_resolution=resolution,
    )
    state_dict = {k.replace("neural_net.", ""): v for k, v in ckpt["state_dict"].items()}
    # strict=False is upstream's own choice (see _vendor/head_segmentation_model.py's
    # docstring) — the real checkpoint's state_dict also carries a
    # `criterion.weight` key (the training loss's class-weight buffer, not
    # part of the model), which strict=False silently drops as
    # "unexpected". Log the actual counts rather than trust that silence:
    # any *missing* key (a real model parameter this checkpoint doesn't
    # provide) would mean an incompletely-initialized model, quietly.
    result = model.load_state_dict(state_dict, strict=False)
    num_model_keys = len(state_dict) - len(result.unexpected_keys)
    logger.info(
        f"head-segmentation state_dict load: {num_model_keys - len(result.missing_keys)} "
        f"of {num_model_keys} real model keys matched (checkpoint also carried "
        f"{len(result.unexpected_keys)} unrelated key(s), correctly ignored); "
        f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
    )
    if result.missing_keys:
        raise RuntimeError(
            f"head-segmentation checkpoint is missing real model keys: {result.missing_keys} "
            "— the model would be partially randomly-initialized, not a real fix to paper over "
            "with strict=False. See models/ganonymization/NOTICE.md."
        )
    return model, resolution


def load_or_random_head_segmentation(
    weights_path: Optional[Path], random_init: bool, device, resolution_if_random: int = 512,
) -> Tuple[object, int]:
    """Real checkpoint, or a randomly-initialized stand-in for a
    `--random-init` smoke test — the branch both backends that use this
    model (ganonymization always, ciagan opt-in via `--refine-mask`) need
    identically. `resolution_if_random` is a plain default, unrelated to
    either backend's own generator resolution — an earlier version of this
    code mistakenly reused ciagan's 128px generator size here for its
    random-init path, which has no relationship to the segmentation
    model's own resolution (harmless only because random-init is
    smoke-test-only, but wrong regardless).
    """
    from ._vendor.head_segmentation_model import HeadSegmentationModel

    if random_init:
        model = HeadSegmentationModel(
            encoder_name="resnet18", encoder_depth=5,
            pretrained=False, nn_image_input_resolution=resolution_if_random,
        )
        resolution = resolution_if_random
    else:
        hash_and_log(weights_path, "head-segmentation model")
        model, resolution = load_head_segmentation(weights_path, device)
    model.eval()
    model.to(device)
    return model, resolution


def predict_head_mask(model, resolution: int, image_rgb: np.ndarray, device) -> np.ndarray:
    """Boolean full-head mask, same (H, W) as `image_rgb` (RGB uint8, HWC — see module docstring).

    Called once per frame, so avoids two easy sources of per-call overhead:
    the transform pipeline is built once per `resolution` and cached
    (`_detection_transforms`), and `ToTensor()` runs directly on the numpy
    array rather than round-tripping through `PIL.Image.fromarray()` first
    — behaviorally identical for a uint8 HWC array (`ToTensor` handles
    `np.ndarray` natively), just without the extra copy.
    """
    import torch
    from torchvision import transforms

    tfm = _detection_transforms.get(resolution)
    if tfm is None:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        _detection_transforms[resolution] = tfm
    inp = tfm(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    label_map = out.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(label_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return resized == 1  # LABEL2INDEX: 0=background, 1=head (verified against constants.py)
