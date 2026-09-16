"""One-time PyTorch checkpoint -> ONNX conversion for SCRFD-34GF.

No pre-built ONNX is published for SCRFD-34GF (unlike SCRFD-10GF, shipped
inside InsightFace's `buffalo_l` pack as `det_10g.onnx`). The official
source, `deepinsight/insightface`'s `detection/scrfd/`, distributes only a
PyTorch checkpoint (OneDrive link in its README — download it yourself)
plus a `tools/scrfd2onnx.py` script that assumes a full checkout of that
mmdetection-based repo. This is a standalone adaptation of that script's
logic against the minimal vendored slice in `vendor/` (see `NOTICE.md`).

One-time, offline tool, not part of the inference pipeline: `backend.py`
loads the resulting `.onnx` via `insightface.model_zoo.model_zoo.ModelRouter`,
no `torch`/`mmcv`/`mmdet` at inference time. Needs the `scrfd-34gf-convert`
optional dependency group (`torch`, `mmcv` 1.3.3-1.3.x, `mmdet` 2.11-2.13,
`onnx`, `onnxsim`) — not installed by a plain `uv sync`.

Usage:

    uv sync --extra scrfd-34gf-convert
    python -m src.pipeline.phase1_detect.models.scrfd_34gf.convert \\
        --checkpoint <path-to-manually-downloaded .pth> \\
        --output weights/scrfd_34g.onnx

Always exports with dynamic H/W axes, unlike upstream: `scrfd2onnx.py`'s
`--shape` flag silently switches between a static export (shape given) and
dynamic (shape omitted) — easy to trip over, and a static export would
break `backend.py`, which calls `model.detect(frame, input_size=self.det_size)`
with a runtime-configurable size on every call.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

_CONFIG_PATH = Path(__file__).resolve().parent / "vendor" / "configs" / "scrfd_34g.py"

_OUTPUT_NAMES = [
    "score_8", "score_16", "score_32",
    "bbox_8", "bbox_16", "bbox_32",
]


def _register_custom_modules() -> None:
    """Register the vendored SCRFD backbone/head/detector classes with
    mmdet's global component registries (`BACKBONES`, `HEADS`, `DETECTORS`)
    as a side effect of importing them. Must happen before
    `scrfd_34g.py` is parsed by `mmcv.Config`/built by mmdet — see
    vendor/custom_modules/__init__.py and NOTICE.md for exactly what's
    registered and why (`PAFPN`, the config's neck, is *not* vendored: it's
    unmodified from vanilla mmdet, and vendoring it too would collide with
    mmdet's own registration of the same name).
    """
    from .vendor import custom_modules  # noqa: F401


def _make_dummy_image(path: Path, height: int, width: int) -> None:
    """Write a throwaway RGB image used only to trace the model's graph.

    mmdet's export helper (`generate_inputs_and_wrap_model`) needs a real
    image file to build img_metas from, but the exported graph's weights and
    structure don't depend on pixel content — only on `height`/`width`,
    which come from `--input-size` — so random noise is fine here.
    """
    import cv2

    img = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"failed to write dummy tracing image to {path}")


def convert(
    checkpoint: Path,
    output: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 11,
    simplify: bool = True,
    mean: tuple[float, float, float] = (127.5, 127.5, 127.5),
    std: tuple[float, float, float] = (128.0, 128.0, 128.0),
) -> None:
    """Convert an SCRFD-34GF `.pth` checkpoint to a dynamic-shape ONNX file.

    Adapted from `pytorch2onnx()` in upstream `detection/scrfd/tools/
    scrfd2onnx.py` (see NOTICE.md for the exact source/commit). The upstream
    optimizer-stripping step (checkpoints often bundle an `optimizer` state
    dict, larger and irrelevant for inference) is preserved.
    """
    try:
        import onnx
        import torch
        from mmdet.core import build_model_from_cfg  # noqa: F401  (import-availability check)
        from mmdet.core import generate_inputs_and_wrap_model
    except ImportError as e:
        raise ImportError(
            "SCRFD-34GF conversion needs the 'scrfd-34gf-convert' optional "
            "dependency group (torch, mmcv 1.3.3-1.3.x, mmdet 2.11-2.13, "
            "onnx, onnxsim). Install with: uv sync --extra scrfd-34gf-convert"
        ) from e

    _register_custom_modules()

    checkpoint = Path(checkpoint)
    output = Path(output)
    height, width = input_size
    input_shape = (1, 3, height, width)

    # Upstream strips the optimizer state (irrelevant for inference, and can
    # be a large fraction of checkpoint size) before loading.
    raw_checkpoint = torch.load(str(checkpoint), map_location="cpu")
    checkpoint_to_load = checkpoint
    slimmed_path: Path | None = None
    if isinstance(raw_checkpoint, dict) and "optimizer" in raw_checkpoint:
        del raw_checkpoint["optimizer"]
        slimmed_path = checkpoint.with_name(checkpoint.name + "_slim.pth")
        torch.save(raw_checkpoint, slimmed_path)
        checkpoint_to_load = slimmed_path

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_img_path = Path(tmp_dir) / "dummy.jpg"
            _make_dummy_image(dummy_img_path, height, width)

            input_config = {
                "input_shape": input_shape,
                "input_path": str(dummy_img_path),
                "normalize_cfg": {"mean": list(mean), "std": list(std)},
            }

            model, tensor_data = generate_inputs_and_wrap_model(
                str(_CONFIG_PATH), str(checkpoint_to_load), input_config
            )
    finally:
        if slimmed_path is not None and slimmed_path.exists():
            slimmed_path.unlink()

    output_names = list(_OUTPUT_NAMES)
    if "stride_kps" in str(model):
        output_names += ["kps_8", "kps_16", "kps_32"]

    input_names = ["input.1"]
    dynamic_axes = {name: {0: "?", 1: "?"} for name in output_names}
    dynamic_axes[input_names[0]] = {0: "?", 2: "?", 3: "?"}

    output.parent.mkdir(parents=True, exist_ok=True)
    export_target = output.with_name(output.stem + "_ori.onnx") if simplify else output

    torch.onnx.export(
        model,
        tensor_data,
        str(export_target),
        keep_initializers_as_inputs=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    if simplify:
        from onnxsim import simplify as onnxsim_simplify

        onnx_model = onnx.load(str(export_target))
        input_shapes = {onnx_model.graph.input[0].name: list(input_shape)}
        onnx_model, ok = onnxsim_simplify(
            onnx_model, input_shapes=input_shapes, dynamic_input_shape=True
        )
        if not ok:
            raise RuntimeError("onnxsim could not validate the simplified ONNX model")
        onnx.save(onnx_model, str(output))
        export_target.unlink()

    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an SCRFD-34GF PyTorch checkpoint to a dynamic-shape "
            "ONNX file, using the vendored scrfd_34g.py config (see "
            "vendor/ and NOTICE.md)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help=(
            "Path to the manually downloaded SCRFD-34GF .pth checkpoint. "
            "Upstream distributes it only via a OneDrive link (see NOTICE.md) "
            "— not fetchable by this tool."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .onnx path, e.g. weights/scrfd_34g.onnx",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=[640, 640],
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "Dummy shape used only to trace the graph (default: 640 640). "
            "The exported ONNX accepts any height/width at inference — see "
            "the module docstring's note on dynamic vs. static export."
        ),
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version (mmdet 2.x's export path only supports 11).",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip the onnxsim simplification pass.",
    )
    parser.add_argument("--mean", nargs=3, type=float, default=[127.5, 127.5, 127.5])
    parser.add_argument("--std", nargs=3, type=float, default=[128.0, 128.0, 128.0])
    args = parser.parse_args()

    if args.opset_version != 11:
        parser.error("mmdet 2.x's ONNX export path only supports opset 11.")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    convert(
        checkpoint=args.checkpoint,
        output=args.output,
        input_size=(args.input_size[0], args.input_size[1]),
        opset_version=args.opset_version,
        simplify=not args.no_simplify,
        mean=(args.mean[0], args.mean[1], args.mean[2]),
        std=(args.std[0], args.std[1], args.std[2]),
    )


if __name__ == "__main__":
    main()
