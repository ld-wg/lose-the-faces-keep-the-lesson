# Lose the Faces, Keep the Lesson

Expression-preserving face de-identification for educational research. Detects faces in classroom videos and replaces them with synthetic, demographically similar ones while preserving expressions, gaze, and affect.

## Pipeline

```
Input video
  -> phase1_detect   detection + tracking (SCRFD-10GF default, ByteTrack)
  -> phase2_generate synthetic face, seeded per track
  -> phase3_temporal temporal stabilization
  -> Output video, de-identified, expression preserved
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp config.example.json config.local.json   # point at your WIDER FACE / weights / runs dirs
```

## Usage

```bash
uv run python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/phase1
uv run python -m src.pipeline.phase1_detect.run --webcam
```

The detector is swappable with `--model`: `scrfd-10gf` (default), `scrfd-34gf`, `yolo-facev2-s`. The latter two need a one-time PyTorch -> ONNX conversion first — see `convert.py` under each model's folder in `src/pipeline/phase1_detect/models/`. `yolo-facev2-s` vendors third-party code with no upstream license; read its `NOTICE.md` before any public release of this repo.

Each run writes `detections.jsonl` (per-frame boxes and confidence) and, for video, `tracks.json` (per-track seed for phase 2).

### Phase 2 — generation

```bash
uv sync --extra phase2-ciagan
uv run python -m src.pipeline.phase2_generate.run --phase1-dir runs/phase1 --out runs/phase2 \
    --weights weights/ciagan_generator.pth --dlib-predictor weights/shape_predictor_68_face_landmarks.dat
```

`--phase1-dir` just needs `detections.jsonl` + `tracks.json` (no `--save-crops` required — Phase 2 reads pixels straight from the source video). Two files aren't in the repo and must be fetched manually before running for real: the CIAGAN checkpoint and dlib's `shape_predictor_68_face_landmarks.dat` — see `src/pipeline/phase2_generate/models/ciagan/NOTICE.md` for exact links, SHA256, and licensing notes. Without them, `--random-init` runs the same pipeline with random weights as a smoke test (not real anonymization).

## Configuration

Paths resolve via `src/config.py`: env vars (`PPY_WIDERFACE`, `PPY_DATASET_DIR`, `PPY_WEIGHTS_DIR`, `PPY_RUNS_DIR`) override `config.local.json`, which overrides the built-in defaults. Check your setup with:

```bash
uv run python -m src.config
```

## Status

| Stage | State |
|---|---|
| Detection + tracking | Implemented |
| Generation + compositing | CIAGAN baseline implemented |
| Temporal stabilization | Not started |
| Evaluation | Not started |

See `research/` for design rationale and open questions.

## References

- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- InsightFace: https://github.com/deepinsight/insightface
- ByteTrack: https://github.com/ifzhang/ByteTrack
