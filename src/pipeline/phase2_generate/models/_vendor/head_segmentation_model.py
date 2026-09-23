# Vendored verbatim from wiktorlazarski/head-segmentation, tag v1.3.0,
# commit c1b4b9b4f12f168f13d603747b81e8f264bcf501,
# head_segmentation/model.py. See ../ganonymization/NOTICE.md for
# provenance/license (LICENSE_head_segmentation in this directory).
#
# NOT used by this project's own loader
# (see ../_segmentation.py::load_head_segmentation) — kept here verbatim
# for attribution/completeness only. Its `load_from_checkpoint()`'s
# `torch.load(ckpt_path, ...)` call below has no explicit `weights_only`
# argument; PyTorch 2.6 flipped that default to `True`, which cannot
# unpickle this checkpoint format (a full PyTorch Lightning checkpoint,
# not a bare state_dict — it carries a `hyper_parameters` dict alongside
# the tensors). This project's `phase2-ganonymization` extra allows
# torch up to <2.7, so 2.6.x is reachable. `_segmentation.py` reimplements
# this same loading logic with `weights_only=False` explicit instead of
# calling this method. Verify this reasoning against whatever torch
# version is actually installed before trusting it blindly.
from __future__ import annotations

import typing as t

import segmentation_models_pytorch as smp
import torch


class HeadSegmentationModel(smp.Unet):
    @staticmethod
    def load_from_checkpoint(ckpt_path: str) -> HeadSegmentationModel:
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))

        hparams = ckpt["hyper_parameters"]
        neural_net = HeadSegmentationModel(
            encoder_name=hparams["encoder_name"],
            encoder_depth=hparams["encoder_depth"],
            pretrained=False,
            nn_image_input_resolution=hparams["nn_image_input_resolution"],
        )

        weigths = {
            k.replace("neural_net.", ""): v for k, v in ckpt["state_dict"].items()
        }
        neural_net.load_state_dict(weigths, strict=False)

        return neural_net

    def __init__(
        self,
        encoder_name: str,
        encoder_depth: int,
        pretrained: bool,
        nn_image_input_resolution: int,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights="imagenet" if pretrained else None,
            decoder_use_batchnorm=True,
            decoder_channels=self._decoder_channels(
                nn_image_input_resolution, encoder_depth
            ),
            decoder_attention_type=None,
            in_channels=3,
            classes=2,
        )

    def _decoder_channels(
        self, nn_image_input_resolution: int, encoder_depth: int
    ) -> t.Tuple[int]:
        return tuple(
            [nn_image_input_resolution // (2 ** i) for i in range(encoder_depth)]
        )
