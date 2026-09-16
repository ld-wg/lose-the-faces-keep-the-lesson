# Vendored from deepinsight/insightface, detection/scrfd/mmdet/core/bbox/transforms.py
# Source: https://github.com/deepinsight/insightface/blob/80e20d868f25344051ae30729257953ed4d3bc43/detection/scrfd/mmdet/core/bbox/transforms.py
# Commit: 80e20d868f25344051ae30729257953ed4d3bc43 (fetched 2026-08-12)
# License: Apache-2.0 (see ../LICENSE)
#
# `distance2kps`/`kps2distance` are SCRFD-specific additions to mmdet.core
# that do not exist in vanilla mmdetection (whose `mmdet.core.bbox.transforms`
# only has the bbox-oriented `distance2bbox`/`bbox2distance`). They are pulled
# out into their own module — rather than vendoring the whole transforms.py,
# whose other ~10 functions are unmodified copies of vanilla mmdet — so that
# `scrfd_head.py` can import just this SCRFD-only slice instead of shadowing
# `mmdet.core` wholesale.
#
# Only reproduced verbatim from the real source; not reimplemented from
# memory.
import torch


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints (5-point landmarks).

    Not called on the SCRFD-34G forward/export path (the 34G config sets
    ``use_kps=False``), but imported unconditionally by `scrfd_head.py` at
    module load time, so it must exist.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to each keypoint.
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded kps.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, -1)


def kps2distance(points, kps, max_dis=None, eps=0.1):
    """Encode keypoints (5-point landmarks) as distances from anchor points.

    Only reached at training time when ``use_kps=True`` (not the case for
    SCRFD-34G's `scrfd_34g.py` config), but imported unconditionally by
    `scrfd_head.py` at module load time, so it must exist.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        kps (Tensor): Shape (n, K), "xyxy" format.
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=.

    Returns:
        Tensor: Encoded distances.
    """
    preds = []
    for i in range(0, kps.shape[1], 2):
        px = kps[:, i] - points[:, i % 2]
        py = kps[:, i + 1] - points[:, i % 2 + 1]
        if max_dis is not None:
            px = px.clamp(min=0, max=max_dis - eps)
            py = py.clamp(min=0, max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, -1)
