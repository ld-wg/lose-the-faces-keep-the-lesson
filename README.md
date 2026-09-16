# Lose the Faces, Keep the Lesson

Expression-preserving face de-identification for educational research. Detects faces in classroom videos and replaces them with synthetic, demographically similar ones while preserving expressions, gaze, and affect.

## Pipeline

```
Input Video (classroom lecture)
    ↓
[phase1_detect]   Detection + Tracking
                  SCRFD-10GF (default, pretrained on WIDER FACE) + ByteTrack;
                  SCRFD-34GF / YOLO-FaceV2-l selectable for comparison (--model)
    ↓ per-frame [frame_id, track_id, box, conf] + optional padded face crops
[phase2_generate] Generation + Compositing
                  Synthetic surrogate face, conditioned and composited in place;
                  fixed seed per track → one consistent pseudonymous identity per student
    ↓
[phase3_temporal] Temporal Stabilization
                  Anchors each frame to the track's canonical latent (gradient injection)
    ↓
Output Video (de-identified, expression-preserved) + audio
```

## Project Structure

```
├── src/
│   ├── config.py                # Central path/env configuration
│   └── pipeline/
│       ├── phase1_detect/       # Detection + tracking
│       │   ├── detector.py      # FaceDetector facade — dispatches to a model backend
│       │   ├── models/          # One backend per detector, selected via --model:
│       │   │   ├── scrfd_10gf/      #   SCRFD-10GF via InsightFace buffalo_l pack (default)
│       │   │   ├── scrfd_34gf/      #   SCRFD-34GF via a converted .onnx (see convert.py)
│       │   │   └── yolo_facev2_l/   #   YOLO-FaceV2-l (vendored inference code, see NOTICE.md)
│       │   ├── tracker.py       # ByteTrack-style multi-face tracker
│       │   └── run.py           # CLI: video / webcam / image dir → detections.jsonl
│       ├── phase2_generate/     # Synthetic face generation + compositing
│       └── phase3_temporal/     # Temporal identity stabilization
├── docs/                        # Notes and guides
├── paper/                       # LaTeX source (main paper)
├── research/                    # Research vault — decisions, paper notes, open questions
├── config.example.json          # Path config template
├── pyproject.toml               # Project metadata + dependencies (uv)
└── uv.lock                      # Locked dependency versions
```

## Configuration

Dataset and output paths are resolved by `src/config.py` with this priority:

1. **Environment variables** (highest): `PPY_WIDERFACE`, `PPY_DATASET_DIR`, `PPY_WEIGHTS_DIR`, `PPY_RUNS_DIR`
2. **`config.local.json`** at the repo root (gitignored — copy `config.example.json` as a template)
3. **Built-in defaults** (paths under the repo)

`widerface_root` must point to an extracted WIDER FACE directory containing
`wider_face_split/`, `WIDER_train/`, and `WIDER_val/`.

Check your setup:

```bash
uv run python -m src.config
```

### Detector models (`--model`)

Phase 1's detector is swappable — SCRFD-10GF (default), SCRFD-34GF, and YOLO-FaceV2-l are the three candidates under head-to-head comparison (`research/stages/identification.md`, Tier 0.5). All three run through the same runtime at inference time (`onnxruntime` on an `.onnx` file) — SCRFD-10GF needs nothing beyond the base install; the other two need a one-time, offline conversion first, since neither ships a ready-to-use ONNX file:

- **SCRFD-34GF** — no pre-built ONNX is published anywhere; convert one yourself from the official checkpoint:
  ```bash
  uv sync --extra scrfd-34gf-convert          # mmcv/mmdet, one-time, conversion only
  # download the SCRFD-34GF checkpoint (.pth) from the OneDrive link in
  # deepinsight/insightface's detection/scrfd README, then:
  uv run python -m src.pipeline.phase1_detect.models.scrfd_34gf.convert \
      --checkpoint /path/to/scrfd_34g.pth --output weights/scrfd_34g.onnx
  ```
- **YOLO-FaceV2-l** — same pattern: download `yolo-facev2l-preweight.pt` from [Krasjet-Yu/YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2), then convert it:
  ```bash
  uv sync --extra yolo-facev2-convert         # torch, one-time, conversion only
  uv run python -m src.pipeline.phase1_detect.models.yolo_facev2_l.convert \
      --checkpoint /path/to/yolo-facev2l-preweight.pt --output weights/yolo_facev2l.onnx
  ```
  Note: this conversion step vendors a slice of YOLO-FaceV2's model code, which upstream ships with no LICENSE file — see `src/pipeline/phase1_detect/models/yolo_facev2_l/NOTICE.md` before any public release of this repo. Inference itself never touches that vendored code.

For both, inference afterward uses only the base `insightface`/`onnxruntime` stack already required by SCRFD-10GF — no `mmcv`/`mmdet`/`torch` needed at run time. Both default to `weights/<file>` under `CONFIG.weights_dir`; override with `--weights <path>`.

## Quick Start

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync                                    # creates .venv, installs locked dependencies
cp config.example.json config.local.json   # then edit paths as needed
```

Run detection + tracking on a video:

```bash
uv run python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/phase1 --save-crops --preview
```

Swap the detector (see "Detector models" above for one-time setup per model):

```bash
uv run python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/scrfd34 --model scrfd-34gf
uv run python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/yolov2 --model yolo-facev2-l
```

Live webcam preview:

```bash
uv run python -m src.pipeline.phase1_detect.run --webcam
```

Each run writes `detections.jsonl` — one record per frame: `{frame_id, tracks: [{track_id, box, conf}]}` — plus, when requested, padded face crops (for the generation stage) and an annotated `preview.mp4`.

## Status

| Stage                                                              | State       |
| ------------------------------------------------------------------- | ----------- |
| Detection + Tracking (SCRFD-10GF default + ByteTrack; SCRFD-34GF / YOLO-FaceV2-l selectable) | Implemented |
| Generation + Compositing                                           | Not started |
| Temporal Stabilization                                             | Not started |
| Privacy / utility evaluation                                       | Not started |

See `research/` for the design rationale, candidate shortlist, and open questions behind each stage.

## References

- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- InsightFace: https://github.com/deepinsight/insightface
- ByteTrack: https://github.com/ifzhang/ByteTrack
