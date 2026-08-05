# Lose the Faces, Keep the Lesson

Expression-preserving face de-identification for educational research. Detects faces in classroom videos and replaces them with synthetic, demographically similar ones while preserving expressions, gaze, and affect.

## Pipeline

```
Input Video (classroom lecture)
    ↓
[phase1_detect]   Detection + Tracking
                  RetinaFace (pretrained, WIDER FACE) + ByteTrack
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
│       │   ├── detector.py      # RetinaFace via InsightFace
│       │   ├── tracker.py       # ByteTrack-style multi-face tracker
│       │   └── run.py           # CLI: video / webcam / image dir → detections.jsonl
│       ├── phase2_generate/     # Synthetic face generation + compositing
│       └── phase3_temporal/     # Temporal identity stabilization
├── weights/                     # Model weights
├── docs/                        # Notes and guides
├── paper/                       # LaTeX source (main paper)
├── research/                    # Research vault — decisions, paper notes, open questions
├── experiments/                 # Experiment result CSVs
├── config.example.json          # Path config template
└── requirements.txt
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
python -m src.config
```

## Quick Start

```bash
pip install -r requirements.txt
cp config.example.json config.local.json   # then edit paths as needed
```

Run detection + tracking on a video:

```bash
python -m src.pipeline.phase1_detect.run --input lecture.mp4 --out runs/phase1 --save-crops --preview
```

Live webcam preview:

```bash
python -m src.pipeline.phase1_detect.run --webcam
```

Each run writes `detections.jsonl` — one record per frame: `{frame_id, tracks: [{track_id, box, conf}]}` — plus, when requested, padded face crops (for the generation stage) and an annotated `preview.mp4`.

## Status

| Stage | State |
|-------|-------|
| Detection + Tracking (RetinaFace + ByteTrack) | ✅ Implemented |
| Generation + Compositing | ⬜ Not started |
| Temporal Stabilization | ⬜ Not started |
| Privacy / utility evaluation | ⬜ Not started |

See `research/` for the design rationale, candidate shortlist, and open questions behind each stage.

## References

- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- InsightFace: https://github.com/deepinsight/insightface
- ByteTrack: https://github.com/ifzhang/ByteTrack
