"""Registers the SCRFD-34GF-specific model classes with mmdet's global
component registries (``BACKBONES``, ``HEADS``, ``DETECTORS``).

`scrfd_34g.py`'s ``model=dict(...)`` references three types that do not
exist in vanilla mmdetection -- ``ResNetV1e`` (backbone), ``SCRFDHead``
(bbox_head), and ``SCRFD`` (detector) -- plus one, ``PAFPN`` (neck), that
vanilla mmdetection already ships unchanged and is therefore *not*
vendored here; mmdet's own registered ``PAFPN`` is used as-is.

Importing this package (or importing ``convert.py``, which imports it for
you) runs the registrations as a side effect, mirroring the pattern
mmdet's own ``mmdet/models/__init__.py`` uses. Do this before calling
``mmcv.Config.fromfile`` / ``build_detector`` on `scrfd_34g.py`.
"""
from . import resnet  # noqa: F401  (registers ResNetV1e)
from . import scrfd_head  # noqa: F401  (registers SCRFDHead)
from . import scrfd_detector  # noqa: F401  (registers SCRFD)

__all__: list = []
