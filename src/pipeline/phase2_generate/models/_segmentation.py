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
inside the two functions below, not at module level — this module is
imported unconditionally at the top of both `ciagan/backend.py` and
`ganonymization/backend.py`, and `ciagan`'s own `phase2-ciagan` extra does
NOT include `torchvision`/`segmentation-models-pytorch` (only
`phase2-ganonymization` does). A module-level import here would break
plain `--model ciagan` (even without `--refine-mask`) for anyone who only
installed `phase2-ciagan` — the same lazy-import discipline
`ciagan/backend.py` already applies to `dlib`/`torch` itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_head_segmentation(weights_path: Path, device) -> Tuple[object, int]:
    """Load the vendored model and its native input resolution.

    Both come from the same checkpoint file (a PyTorch Lightning
    checkpoint: a `state_dict` plus a `hyper_parameters` dict, not a bare
    tensor-only file like ciagan's `.pth`) — `weights_only=False` is
    required to unpickle the hyperparameters dict; this is this project's
    own downloaded, sha256-recorded copy, not arbitrary untrusted input,
    so that trust boundary is acceptable here the same way it already is
    for ciagan's checkpoint.
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
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, resolution


def predict_head_mask(model, resolution: int, image_rgb: np.ndarray, device) -> np.ndarray:
    """Boolean full-head mask, same (H, W) as `image_rgb` (RGB uint8, HWC — see module docstring)."""
    import torch
    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resolution, resolution)),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    inp = tfm(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    label_map = out.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(label_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return resized == 1  # LABEL2INDEX: 0=background, 1=head (verified against constants.py)
