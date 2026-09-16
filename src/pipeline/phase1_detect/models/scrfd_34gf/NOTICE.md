# NOTICE — vendored SCRFD-34GF conversion code

`vendor/` contains a minimal slice of code vendored from `detection/scrfd/`
inside the **deepinsight/insightface** repository, needed to convert the
official SCRFD-34GF PyTorch checkpoint to ONNX. It is used only by
`convert.py` (a one-time, offline conversion tool); it is not imported by
`backend.py` or any other inference-time code, which loads the already
-converted `.onnx` file directly.

- **Source repository:** https://github.com/deepinsight/insightface
- **Source subdirectory:** `detection/scrfd/` (this subdirectory ships its
  own `LICENSE`, distinct from the top-level `deepinsight/insightface`
  repo, which has no declared license)
- **Commit SHA fetched:** `80e20d868f25344051ae30729257953ed4d3bc43`
  (latest commit touching `detection/scrfd` on the `master` branch at fetch
  time, obtained via
  `curl -s "https://api.github.com/repos/deepinsight/insightface/commits?path=detection/scrfd&per_page=1"`)
- **Fetch date:** 2026-08-12
- **License:** Apache License 2.0 — see `vendor/LICENSE`, copied verbatim
  from `detection/scrfd/LICENSE` at the commit above.
  Copyright 2018-2019 Open-MMLab. Portions © Jia Guo and the SCRFD authors.
  Reused here under the terms of that Apache-2.0 license.

## What was vendored, and why

`configs/scrfd/scrfd_34g.py` has **no `_base_` config chain** — it is a
complete, standalone mmdetection config with no `_base_ = [...]` line, so
`configs/scrfd/base_34g.py` (present in the upstream repo) is not referenced
by it and was not vendored.

`scrfd2onnx.py`'s conversion path (`generate_inputs_and_wrap_model` ->
`build_model_from_cfg` -> `mmdet.models.build_detector`) resolves the four
component types the config's `model=dict(...)` names, via mmdet's
registries:

| Registry | `type=` in config | Vendored? | Why |
|---|---|---|---|
| `DETECTORS` | `SCRFD` | Yes — `custom_modules/scrfd_detector.py` | Does not exist in vanilla mmdetection. |
| `BACKBONES` | `ResNetV1e` | Yes — `custom_modules/resnet.py` | Does not exist in vanilla mmdetection; needs a customized `ResNet.__init__` (`block_cfg`, `no_pool33`, `arch_settings[0]`) to build the NAS-searched architecture the 34G config describes (`depth=0, block_cfg=dict(block='Bottleneck', stage_blocks=(17,16,2,8), stage_planes=[56,56,144,184])`). |
| `NECKS` | `PAFPN` | **No** | Traced this fork's `mmdet/models/necks/pafpn.py` and confirmed it is functionally identical to the `PAFPN` vanilla mmdetection (2.11–2.13) already ships and registers under the same name. Vendoring it too would collide with mmdet's own registration (`KeyError: PAFPN is already registered`). Vanilla mmdet's `PAFPN` is used unmodified. |
| `HEADS` | `SCRFDHead` | Yes — `custom_modules/scrfd_head.py` | Does not exist in vanilla mmdetection. |

Additionally, `scrfd_head.py` imports `distance2kps`/`kps2distance` from
`mmdet.core` — two SCRFD-specific functions this fork adds to its own copy
of `mmdet/core/bbox/transforms.py` that are **not present** in vanilla
mmdet.core (which only has the bbox-oriented `distance2bbox`/
`bbox2distance`, both of which vanilla mmdet does provide and which are
used as-is). Rather than vendor the fork's entire `transforms.py` (whose
other ~10 functions are unmodified vanilla mmdet code), only these two
functions were extracted verbatim into `custom_modules/kps_transforms.py`.
Neither is actually reached at inference/export time for the 34G config
(which sets `use_kps=False`), but both are imported unconditionally at
module load time, so they must exist for the module to import.

`ResNet`/`ResNetV1d` (the base classes `ResNetV1e` subclasses, in the same
`resnet.py`) keep their code but had their `@BACKBONES.register_module()`
decorators removed in the vendored copy — vanilla mmdet already registers
classes named `ResNet` and `ResNetV1d` in the same global registry, and
`ResNetV1e` reaches them through direct Python subclassing, not through the
registry, so no functionality is lost. See the header comment in
`custom_modules/resnet.py` for the full explanation.

## Registry-collision risk (real, but scoped)

This vendoring approach relies on `mmdet==2.11.0` (installed per `convert.py`'s
docstring — a separate Python 3.8 environment, not this project's own) not
already registering anything named `SCRFD`, `SCRFDHead`, or `ResNetV1e` —
confirmed true for mainline mmdetection as of this writing, since none of
these three names/architectures exist upstream. If a future `mmdet` release
were to add a same-named class, importing `vendor/custom_modules` would
raise `KeyError: "... is already registered in ..."` at import time — a
loud, immediate failure, not silent misbehavior.

## Not vendored / out of scope

- `configs/scrfd/base_34g.py` — not referenced by `scrfd_34g.py` (no
  `_base_` chain), so not needed.
- The rest of the fork's `mmdet/` tree (datasets, training pipelines,
  non-SCRFD detectors/heads/backbones, `search_tools/`, etc.) — irrelevant
  to a checkpoint -> ONNX conversion of an already-trained model.
- The `.pth` checkpoint itself — distributed only via a OneDrive link in
  the upstream README, not fetchable/redistributable here; must be
  downloaded manually and passed to `convert.py --checkpoint`.

No time-box overrun: the custom-code surface turned out to be four small,
cleanly separable files (one of which — `kps_transforms.py` — is two
functions), not an extensive fork-wide coupling, so nothing was cut short
here.
