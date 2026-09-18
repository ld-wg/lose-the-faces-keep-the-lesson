# NOTICE — vendored third-party code

`vendor/` in this directory contains a slice of
[**Krasjet-Yu/YOLO-FaceV2**](https://github.com/Krasjet-Yu/YOLO-FaceV2)'s
model-definition code, needed to load a checkpoint from that repo (a custom
YOLOv5 architecture — CBAM/SE/EMA attention + RFEM/MultiSEAM modules — not
loadable via stock `ultralytics`/`torch.hub`) for a **one-time conversion to
ONNX** (`convert.py`). Mirroring `../scrfd_34gf`'s shape: `backend.py` never
imports `vendor/` or `torch` — it only loads the resulting `.onnx` file via
`onnxruntime`, exactly like the other two detector backends. `vendor/` and
the licensing situation below matter only if you run `convert.py` yourself.

**This is the "-s" (small) checkpoint, not "-l" (large) as originally
planned** — see "Only -s actually works" below for why.

## Provenance

- **Source repo:** https://github.com/Krasjet-Yu/YOLO-FaceV2
- **Pinned commit:** `2c125edc693a2fbb9d225a023a9514abef9ecb59` (the repo's
  `master` HEAD at fetch time, via `GET /repos/Krasjet-Yu/YOLO-FaceV2/commits?per_page=1`)
- **Fetch date:** 2026-08-12
- **Fetched via:** `curl`/GitHub API against `api.github.com` and
  `raw.githubusercontent.com` directly (not reconstructed from memory or
  approximated).

## Licensing situation upstream — read before any public release of this repo

Confirmed two ways, at fetch time:

1. `GET /repos/Krasjet-Yu/YOLO-FaceV2` → `"license": null`.
2. `LICENSE`, `LICENSE.md`, and `LICENSE.txt` at the repo root all return
   HTTP 404 on `raw.githubusercontent.com`.

**There is no LICENSE file in this repository.** Under default copyright
law (and GitHub's own terms of service framing for unlicensed repos), that
means all rights are reserved by the authors — there is no license granting
permission to copy, modify, or redistribute this code, notwithstanding that
it's publicly visible on GitHub.

Confusingly, the repo's own `README.md` states, in its "Contact" section:
*"We use code's license is MIT License. The code can be used for business
inquiries or professional support requests."* This is **not backed by an
actual LICENSE file anywhere in the repository** — a README claim is not a
license grant, and the two sources disagree (a null `license` field is
GitHub's own detection of the absence of a machine-recognized license file,
independent of anything asserted in prose). Treat the repository as
all-rights-reserved, not as MIT-licensed, until/unless the authors add an
actual LICENSE file or grant permission in writing.

**This was flagged to and explicitly accepted by the project owner before
vendoring**, on the condition that it be clearly isolated (this `vendor/`
subtree) and flagged in-repo (this file) rather than buried in a commit
message or code comment. That decision is not being re-litigated here. What
*did* change after the initial vendoring: the vendored code's role shrank
from "imported on every inference call" to "imported only if you run
`convert.py`" (see "Design change" below) — smaller blast radius for the
same accepted risk, not a different risk.

**Recommendation: before any public release of this repo**, do one of:
- Seek written permission from the YOLO-FaceV2 authors (Ziping Yu et al.) to
  redistribute this vendored slice, or
- Reimplement `convert.py`'s use of this vendored code independently from
  the paper (Yu et al., *Pattern Recognition* 155:110714, 2024,
  arXiv:2208.02019), without reference to this vendored code, or
- Drop the `yolo-facev2-s` backend from any public release entirely (the
  other two detector backends, `scrfd-10gf` and `scrfd-34gf`, have no such
  encumbrance).

## Only "-s" actually works — "-l"/"-m"/"-n" are broken as published

This is the finding that determined which checkpoint size this backend
targets, found by actually running all four sizes through `convert.py`
against real footage (`video-demo.mov`) and inspecting the model directly
when the results looked wrong — not assumed from specs or naming.

**What happened:** `yolo-facev2l-preweight.pt` was the original target (see
`research/stages/identification.md`'s Tier 0.5 rationale — largest verified
Hard AP gain among the candidates, from the repo's own Preweight table).
Converting and running it produced **zero detections** on real classroom-like
footage that SCRFD-10GF finds 1112 faces in. Debugging in order:

1. Confirmed the checkpoint has no landmark channels (see below) — fixed the
   decode, still zero detections.
2. Checked objectness/class confidence directly: max ~0.02 across all 17,640
   anchors, only 13-14 distinct values — essentially no variation at all,
   not just "low confidence."
3. Ruled out `attempt_load()`'s `.fuse()` step (BatchNorm folding) as the
   cause: loading the raw, unfused checkpoint gives identical numbers.
4. Ruled out the decode/conversion code itself: tested `yolo-facev2n-preweight.pt`
   (a different size, same code path) — nearly **identical** degenerate
   confidence statistics. Two independently-trained checkpoints producing
   the same narrow broken output through the same code pointed at a shared
   bug in *our* code, not two independently bad models — so this needed
   one more check, not less.
5. Hooked the backbone's own output (before the `Detect` head's final conv):
   **std 0.0004–0.02** across all three feature scales, on an input tensor
   with std ~0.2. A working backbone processing real image content doesn't
   produce that — this is the network not responding to the input at all,
   upstream of any decode logic.
6. Tested `-m` and `-s` too. `-m` matches `-l`/`-n`'s broken pattern exactly
   (backbone std 0.0004–0.02). **`-s` is different**: backbone std
   0.4–1.0 (healthy), objectness up to 0.85, and a full run on
   `video-demo.mov` found 1178 detections across 12 tracks — in line with
   SCRFD-10GF's 1112/15 on the same footage.

**Conclusion:** `-l`, `-m`, and `-n` are functionally broken in the
published v2.1 release, independent of anything in this vendored slice or
`convert.py` (the same code correctly handles `-s`). `-s` is the only size
this backend supports. Its real AP (98.3/97.0/89.3 Easy/Medium/Hard, from
the repo's own table) is lower than `-l`'s claimed 98.6/97.9/91.9, but it's
a number backed by an actually-working checkpoint, which `-l`'s isn't.

**Not fully understood, flagged rather than overstated:** the working
theory was "the broken sizes are stale checkpoints carried over from the
pre-landmark v1.0 release, never retrained" — supported by `v1.0`'s
`preweight.pt` being byte-for-byte the same size as `-s`'s v2.1 file. But
that match is against `-s`, the size that *works* — so it doesn't actually
explain why `-l`/`-m`/`-n` specifically are broken. Treat "-s works, the
other three don't" as a confirmed empirical result, and the byte-size
coincidence as an interesting but unproven side note, not a full
explanation.

## Design change: conversion-only, not a runtime dependency

The first version of this backend loaded the checkpoint directly with
`torch` on every `detect()` call, which required: (1) `torch`/`torchvision`
as a hard runtime dependency, (2) inserting `vendor/` onto `sys.path` as
bare `models`/`utils` packages so `torch.load()`'s unpickler could resolve
the checkpoint's pickled class references (upstream's own repo layout has
`models`/`utils` as top-level packages), and (3)
`torch.load(..., weights_only=False)`, needed because this checkpoint
pickles a full `Model` instance rather than a plain state_dict —
`weights_only=False` executes arbitrary code during unpickling, fine only
for a checkpoint whose provenance you trust.

That's now replaced by a one-time `convert.py` (see its docstring) that
exports the checkpoint to a dynamic-shape `.onnx` file — the same pattern
`../scrfd_34gf` already uses, and the standard ML-serving pattern generally
(ONNX Model Zoo / Triton Inference Server / Hugging Face Hub all ship a
portable graph + a uniform runtime, not a framework-specific Python module
tree). `backend.py` now only does `onnxruntime.InferenceSession(...)` +
plain-numpy NMS/letterbox — no `torch`, no `sys.path` mutation, no
unpickling of untrusted pickled objects at inference time. All three
detector backends in `models/` are now the same shape at runtime.

## What's vendored, and why (conversion-time only)

`vendor/` is now only what `convert.py` needs to build the model and trace
it — not the old backend's inference-time postprocessing (that's plain
numpy in `backend.py` now, not vendored).

| File | Role |
|---|---|
| `vendor/models/yolo.py` | `Model`/`Detect`/`parse_model` — the architecture graph. `convert.py` monkey-patches `Detect.forward` with a corrected decode rather than using this file's own inference branch — see `convert.py`'s docstring for why (its `export_cat` path calls a method, `_make_grid_new`, that doesn't exist anywhere in this file or upstream — dead code, not something to rely on). |
| `vendor/models/common.py` | Layer building blocks (`Conv`, `C3`, `SPP`, `Focus`, `RFEM`, `SEAM`, `MultiSEAM`, `StemBlock`, etc.) that `parse_model` resolves the YAML's module names against. |
| `vendor/models/experimental.py` | `attempt_load()` — loads the `.pt` checkpoint. |
| `vendor/models/attention/{cbam,se,ema}.py` | `CBAM`/`SE`/`EMA` attention modules, verbatim. |
| `vendor/models/yolov5s_v2_RFEM_MultiSEAM.yaml` | The "-s" (small) model config — reference/documentation only, not actually loaded by `convert.py` (`attempt_load()` unpickles the fully-constructed model directly from the checkpoint, no YAML needed). |
| `vendor/utils/general.py` | Trimmed to `make_divisible`/`check_file`/`set_logging` only — the three names `models/yolo.py` imports at module load time. The old NMS/coordinate-rescaling functions (`non_max_suppression_face`, `scale_coords`, `scale_coords_landmarks`, `check_img_size`, `xywh2xyxy`, `clip_coords`) were dropped along with the old direct-`.pt`-loading backend — `backend.py` reimplements NMS/letterbox itself in plain numpy instead. |
| `vendor/utils/torch_utils.py`, `vendor/utils/autoanchor.py` | Support functions `models/yolo.py` imports at module load time. |
| `vendor/utils/google_utils.py` | **Not verbatim** — see below. |

`vendor/utils/datasets.py` (upstream's `letterbox()`) was **removed** — it
was only used by the old direct-`.pt` backend; `convert.py` never needs
preprocessing, only `backend.py` does, and it has its own numpy
implementation now.

Two remaining files are worth calling out:

- **`vendor/utils/google_utils.py` is rewritten, not trimmed.** Upstream's
  `attempt_download()` silently reaches out to GitHub releases / Google
  Cloud Storage over the network if a weights file is missing. This
  replacement keeps the same name/signature but raises `FileNotFoundError`
  instead — no silent network access, no hard dependency on `requests`.
- **`vendor/models/experimental.py`'s `attempt_load()` adds
  `weights_only=False`** to its `torch.load()` call, not present upstream.
  Only run `convert.py` against a checkpoint you trust the provenance of
  (see "Design change" above).

## Conversion-time import mechanics — read if you touch `convert.py` or `vendor/`

The checkpoint pickles its classes qualified as bare `models.yolo.Model`,
`models.common.Conv`, etc. — upstream's own repo layout has `models/` and
`utils/` as sibling top-level packages at the repo root. For
`torch.load()` to unpickle the checkpoint, those exact bare names must be
importable at load time. `convert.py` handles this by inserting this
directory's `vendor/` onto `sys.path` (so `import models`/`import utils`
resolve to the vendored code) before calling `attempt_load()` — not by
nesting these as a "properly" namespaced subpackage, which would load a
second, distinct copy of each module under a different qualified name and
break unpickling. See `vendor/models/experimental.py`'s header comment for
the long version.

This is now confined to `convert.py`'s own one-time, short-lived process —
`backend.py` (the code that runs on every pipeline invocation) never
imports `vendor/` and never mutates `sys.path`.
