# Lose the Faces, Keep the Lesson

Facial de-identification for classroom video by synthetic identity swap (undergraduate thesis, UFPR). Detects every face, swaps it for one fixed synthetic identity per student (an SDXL source identity, swapped per frame by inswapper + GFPGAN), pushes that identity away from the real one, and composes a fail-closed output video that hides every detected face without a valid replacement. It covers far more faces than the released BLANKET with lower re-identification per shown face; blur, censoring and GANs still score lower on final-video rank-1 (they hide or erase the face). Expression error is about 4x its noise floor and ties GANonymization; emotion is not kept, and the output is not anonymized data in the LGPD's sense (non-face cues remain). Results and limits: `paper/tcc-latex/`.

## Pipeline

```
Input video
  -> Stage 1  src/pipeline/phase1_detect     detection + tracking + recall post-pass (SCRFD-10GF, ByteTrack-style)
  -> Stage 2  src/pipeline/phase2_generate   face replacement per track (BLANKET + P2 push; censor/GAN baselines),
                                             release policies (privacy gate, hide-low), frame sampling
  -> Stage 3  phase2_generate/compose_video  fail-closed composition: generated, reused or a gray fill
  -> Output video + compose.jsonl (what every face box shows)
```

Temporal consistency comes from the fixed identity per track and the reuse of the last generated face; there is no separate stabilization stage. Experiments: `experiments/` (E0–E5, each with its README).

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

Two generation backends, selected via `--model`:

```bash
# ciagan (default) — face-proper only, no full-head coverage (see its NOTICE.md)
uv sync --extra phase2-ciagan
uv run python -m src.pipeline.phase2_generate.run --phase1-dir runs/phase1 --out runs/phase2 \
    --weights weights/ciagan_generator.pth --dlib-predictor weights/shape_predictor_68_face_landmarks.dat

# ganonymization — full-head coverage via a real segmentation model
uv sync --extra phase2-ganonymization
uv run python -m src.pipeline.phase2_generate.run --phase1-dir runs/phase1 --out runs/phase2 \
    --model ganonymization --weights weights/ganonymization_pix2pix_50.ckpt \
    --segmentation-weights weights/head_segmentation.ckpt
```

`--phase1-dir` just needs `detections.jsonl` + `tracks.json` (no `--save-crops` required — Phase 2 reads pixels straight from the source video). Neither backend's weights are in the repo and must be fetched manually before running for real — see each backend's own `NOTICE.md` (`src/pipeline/phase2_generate/models/ciagan/NOTICE.md`, `.../ganonymization/NOTICE.md`) for exact links, SHA256, and licensing notes (both GANonymization checkpoints carry a CelebA-derived non-commercial restriction — flagged there, not here). Without weights, `--random-init` runs either pipeline with random weights as a smoke test (not real anonymization).

`ciagan` also has an opt-in `--refine-mask` flag that intersects its composite mask against the same head-segmentation model, to clean up (not expand) its seam at extreme angles — see `models/ciagan/NOTICE.md`.

`run.py` only writes per-face crop PNGs, not a video. To watch the actual result, paste them back into the full source video with `compose_video.py`:

```bash
uv run python -m src.pipeline.phase2_generate.compose_video \
    --phase1-dir runs/phase1 --phase2-dir runs/phase2 --out runs/phase2/output.mp4
```

## Configuration

Paths resolve via `src/config.py`: env vars (`PPY_WIDERFACE`, `PPY_DATASET_DIR`, `PPY_WEIGHTS_DIR`, `PPY_RUNS_DIR`) override `config.local.json`, which overrides the built-in defaults. Check your setup with:

```bash
uv run python -m src.config
```

## Status

| Stage | State |
|---|---|
| 1 Detection + tracking + recall | Implemented (SCRFD-10GF; SCRFD-34GF and YOLO-FaceV2-s as alternatives) |
| 2 Replacement | blur, mosaic, CIAGAN, GANonymization, BLANKET (bridge repo); C2 track identity, P2 push; P1/P3 implemented, negative |
| 3 Fail-closed composition | Implemented |
| Evaluation | E0–E5 run and exported to `paper/generated/` |
| Tests | `tests/` (synthetic arrays only): `uv run --with pytest pytest` |

Design notes and the decision log live in a local research vault (not versioned); `experiments/DECISIONS.md` is the versioned decision record.

## References

- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- InsightFace: https://github.com/deepinsight/insightface
- ByteTrack: https://github.com/ifzhang/ByteTrack
