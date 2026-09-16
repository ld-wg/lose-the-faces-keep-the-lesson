"""One-time PyTorch checkpoint -> ONNX conversion for YOLO-FaceV2-l.

Mirrors `../scrfd_34gf/convert.py`: the only place `vendor/`, `torch`, and
the sys.path/unpickling mechanics they need are touched. `backend.py` never
imports `vendor/` — it only loads the resulting `.onnx` via `onnxruntime`.

Usage:

    uv sync --extra yolo-facev2-convert   # torch — conversion only
    python -m src.pipeline.phase1_detect.models.yolo_facev2_l.convert \\
        --checkpoint <downloaded yolo-facev2l-preweight.pt> \\
        --output weights/yolo_facev2l.onnx

`model.model[-1].export_cat = True` matters: `vendor/models/yolo.py`'s
`Detect` layer otherwise returns a `(decoded_predictions, raw_feature_maps)`
tuple meant for training-time loss computation. `export_cat` (a flag this
fork's own `Detect` class defines for this purpose) switches to a single
tensor per anchor — `[cx, cy, w, h, obj_conf, landmark_x1..y5, cls_conf]`,
decode math already applied — so `backend.py` only has to do confidence
filtering + NMS, not reimplement anchor decoding.

Unverified end-to-end: no real checkpoint was available when this was
written. Verify the exported graph's output shape and a few sample
detections before trusting results from the converted `.onnx`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_VENDOR_DIR = str(Path(__file__).resolve().parent / "vendor")


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
            "YOLO-FaceV2-l conversion needs the 'yolo-facev2-convert' optional "
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
    detect.export_cat = True  # see module docstring

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
    )
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-FaceV2-l PyTorch checkpoint to ONNX (see module docstring)."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the manually downloaded 'yolo-facev2l-preweight.pt' checkpoint "
             "(https://github.com/Krasjet-Yu/YOLO-FaceV2/releases)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .onnx path, e.g. weights/yolo_facev2l.onnx",
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
