"""One-time PyTorch checkpoint -> ONNX conversion for YOLO-FaceV2-s.

Mirrors `../scrfd_34gf/convert.py`: the only place `vendor/`, `torch`, and
the sys.path/unpickling mechanics they need are touched. `backend.py` never
imports `vendor/` — it only loads the resulting `.onnx` via `onnxruntime`.

Usage:

    uv sync --extra yolo-facev2-convert   # torch — conversion only
    python -m src.pipeline.phase1_detect.models.yolo_facev2_s.convert \\
        --checkpoint <downloaded yolo-facev2s-preweight.pt> \\
        --output weights/yolo_facev2s.onnx

This is "-s" (small), not "-l" (large) — deliberately, not a naming
leftover. All four release sizes have **no landmark outputs**, contrary to
the repo's "landmarks version" (v2.1) release label: every checkpoint's
`Detect` layer is `nc=1, no=6, na=3` (standard `[cx, cy, w, h, obj_conf,
cls_conf]` per anchor), not the 16-wide landmark-carrying layout
`vendor/models/yolo.py`'s `Detect.forward()` assumes (it hardcodes a
landmark-offset index that's out of bounds for a 6-wide tensor — confirmed
this isn't a vendoring mistake by diffing against the real upstream file at
the pinned commit). `contracts.py`'s `Landmarks` field is `Optional` for
exactly this kind of case — this backend's `Detection.landmarks` is always
`None`, for any size.

More importantly: `-l`, `-m`, and `-n` are functionally broken as
published. Loading each and inspecting the backbone's own output on a real
image shows near-zero variance (std 0.0004-0.02, on an input with std
~0.2) — the network isn't responding to image content at all, at any of
those three sizes, regardless of decode logic. `-s` is the only one that
works: real backbone variance (std 0.4-1.0), confidence up to 0.85, and a
real video test found 1178 detections across 12 tracks, in line with
SCRFD-10GF's 1112/15 on the same footage. Likely explanation: `v1.0`'s
`preweight.pt` (2022, pre-landmark) is byte-for-byte identical in size to
`v2.1`'s `yolo-facev2s-preweight.pt`... except that match is against `-s`,
which works — so the carried-over-checkpoint theory doesn't fully explain
why `-l`/`-m`/`-n` are the broken ones specifically. Treat "-l/-m/-n are
broken, -s works" as an empirical finding, not a fully understood one — see
NOTICE.md for the full investigation.

Because of that, this module does **not** call the vendored `Detect.forward()`
at all: it monkey-patches the loaded `Detect` instance's `forward` with a
corrected decode for its real `no=6` structure (same box-decode formulas as
upstream's own code, minus the landmark terms and the buggy hardcoded
offset), reusing the real trained backbone/neck/head weights unchanged.
That patched decode already returns a single tensor
(`torch.cat(z, 1)`, shape `(1, num_anchors, 6)`), so the traced graph needs
no separate output-selection wrapper.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

_VENDOR_DIR = str(Path(__file__).resolve().parent / "vendor")


def _decode_without_landmarks(self, x):
    """Corrected `Detect.forward()` for a landmark-free (`no=6`) instance.

    Same box-decode formulas as the vendored `Detect.forward()`'s
    inference branch, minus the landmark terms and the hardcoded
    landmark-offset indexing that assumes a 16-wide (`no=16`) tensor —
    see module docstring for why the original crashes on this checkpoint.
    """
    import torch

    z = []
    for i in range(self.nl):
        x[i] = self.m[i](x[i])
        bs, _, ny, nx = x[i].shape
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.grid[i].shape[2:4] != x[i].shape[2:4]:
            self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
        z.append(y.view(bs, -1, self.no))
    return torch.cat(z, 1)


def convert(
    checkpoint: Path,
    output: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 12,
) -> None:
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "YOLO-FaceV2-s conversion needs the 'yolo-facev2-convert' optional "
            "dependency group (torch). Install with: uv sync --extra yolo-facev2-convert"
        ) from e

    # The checkpoint pickles its classes as bare `models.yolo.Model`, etc.
    # (upstream's own repo layout has `models/`/`utils/` as top-level
    # packages) — see vendor/models/experimental.py's header and NOTICE.md.
    # Confined to this one-time conversion process; backend.py never does this.
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    from models.experimental import attempt_load

    checkpoint = Path(checkpoint)
    output = Path(output)

    model = attempt_load(str(checkpoint), map_location="cpu")
    model.eval()

    detect = model.model[-1]
    if detect.no != detect.nc + 5:
        raise RuntimeError(
            f"Expected a landmark-free Detect head (no == nc + 5), got no={detect.no} "
            f"nc={detect.nc} — this checkpoint's structure doesn't match what this "
            "converter was written for (see module docstring). Don't silently proceed."
        )
    detect.forward = types.MethodType(_decode_without_landmarks, detect)

    height, width = input_size
    dummy = torch.zeros(1, 3, height, width)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["images"],
        output_names=["predictions"],
        dynamic_axes={
            "images": {2: "height", 3: "width"},
            "predictions": {1: "num_predictions"},
        },
        opset_version=opset_version,
        dynamo=False,  # dynamic_axes/input_names/output_names above are the
                       # legacy TorchScript-based exporter's parameter style;
                       # torch >=2.x defaults to the newer dynamo exporter,
                       # which needs the separate `onnxscript` package and a
                       # different parameter surface. Force legacy explicitly.
    )
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-FaceV2-s PyTorch checkpoint to ONNX (see module docstring)."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the manually downloaded 'yolo-facev2s-preweight.pt' checkpoint "
             "(https://github.com/Krasjet-Yu/YOLO-FaceV2/releases)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .onnx path, e.g. weights/yolo_facev2s.onnx",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=[640, 640],
        metavar=("HEIGHT", "WIDTH"),
        help="Dummy shape used only to trace the graph (default: 640 640). "
             "The exported ONNX accepts any height/width at inference (dynamic axes).",
    )
    parser.add_argument("--opset-version", type=int, default=12)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    convert(
        checkpoint=args.checkpoint,
        output=args.output,
        input_size=(args.input_size[0], args.input_size[1]),
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()
