# Lose the Faces, Keep the Lesson

Facial de-identification for classroom video by synthetic identity swap (undergraduate thesis, UFPR). Detects every face, swaps it for one fixed synthetic identity per student (an SDXL source identity, swapped per frame by inswapper + GFPGAN), pushes that identity away from the real one, and composes a fail-closed output video that hides every detected face without a valid replacement. Thesis (Portuguese): `paper/tcc-latex/`; every number below comes from `paper/generated/`.

## Paper in short

**Motivation.** Classroom recordings are evidence for education research (interaction, affect, turn-taking), but they also identify students, many of them minors. Under Brazil's LGPD a face is biometric data and children's data must be handled in their best interest. Blur and mosaic remove the identity, and with it the behavior researchers want to study (expression, head pose, emotion). The question: does replacing each face with a synthetic one cut automatic re-identification while keeping that behavior, measured on the video a viewer actually sees, including the faces the method fails to replace?

**Solution.** A three-stage pipeline built around the hybrid method of BLANKET:

1. Detection and tracking (SCRFD-10GF, ByteTrack-style tracker, recall post-pass).
2. One synthetic identity per track: SDXL generates a source face, inswapper + GFPGAN swap it onto every frame. Our changes: the swap uses Stage 1's detections (the released code misses small faces), and the swap embedding is pushed away from the track's real identity (P2: `e = normalize(e_synth − β·e_real)`, β = 1.2). A privacy gate hides faces still too close to the real person, and adaptive sampling regenerates only when the face moves.
3. Fail-closed composition: every detected face shows either a valid synthetic face or a gray fill, never the real one.

Evaluation: two public classroom clips (22 s, 41 tracks, about 10k faces), two recognizers (FaceNet as the independent judge, ArcFace as the one the push optimizes against), a closed gallery of real faces, and a naive attacker. Experiments E0–E5 compare blur, mosaic, CIAGAN, GANonymization, released BLANKET and the pipeline on the same detections; one clip was seen during the pilot, the other only after the pipeline was frozen.

**Results.**

| | Released BLANKET | Final pipeline |
|---|---|---|
| Faces shown with a synthetic face | 22% (the rest is published real) | 78% (the rest hidden) |
| FaceNet rank-1 among shown faces | 0.68 | 0.34 (0.22 seen clip, 0.45 new clip; chance ≈ 0.05) |

- Re-identification drops but stays 4 to 9 times above chance. An attacker deciding once per student is right for 51% of them. Blur reaches 0.09 on the final video, so blur still protects more.
- The leak is in the face swap, not in the diffusion identity. P2 lowers rank-1 among swapped faces from 0.50 to 0.39 (ArcFace: 0.71 → 0.35), with no utility cost, but this is not significant after Holm correction (p = 0.22). Removing all real-identity content from the source (β = c) only gives 0.50 → 0.46. Guiding the diffusion (P1) or CIAGAN's identity code (P3) changes nothing. SimSwap instead of inswapper leaks more without the push (0.58) and ties with it (0.42).
- Expression error is 0.12 inter-ocular units (about 5× the noise floor; blur 0.23, GANs 0.13–0.18). Head pose error is 9.7° (blur and GANs 14–18°). Emotion is not kept by any method (κ ≈ 0.2).
- The new clip is harder than the pilot clip: the detector finds about half the boxes a second detector finds, and faces are more turned.
- Cost on a GTX 1080: 197 GPU-hours per class hour, 77 with adaptive sampling (per-face loop only).
- 8.9% of swaps returned the face unchanged; the fail-closed composition now hides them.

**Conclusions.**

- Per-track synthetic swap de-identifies classroom video only partially.
- Most of the gain over the published method is coverage, from simple engineering: reuse Stage 1's detections and hide whatever fails.
- The diffusion step, the most expensive part, matters least for privacy. The remaining leak comes from the swapper keeping the real face's shape.
- The output is face de-identification, not anonymized data in the LGPD's sense: hair, body, voice and seat remain.
- When privacy is the requirement, blur is still the choice. Face swap is justified only when visible behavior matters and the residual risk is accepted.
- Limits: two short public clips (not a consented dataset of children), a naive attacker, no human evaluation.

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
| Evaluation | E0–E3 (incl. E3h swapper comparison) and E5 run and exported to `paper/generated/`; E4 proposed, not run |
| Tests | `tests/` (synthetic arrays only): `uv run --with pytest pytest` |

Design notes and the decision log live in a local research vault (not versioned); `experiments/DECISIONS.md` is the versioned decision record.

## References

- WIDER FACE: http://shuoyang1213.me/WIDERFACE/
- InsightFace: https://github.com/deepinsight/insightface
- ByteTrack: https://github.com/ifzhang/ByteTrack
