"""Track-level identity estimation — the thesis contribution's shared core (C2).

For every Phase 1 track, estimate the *real* person's identity from all of
the track's frames, weighted by per-frame quality, so later stages can push
a generated face away from it (C1) and pick the best frame to build a
synthetic identity from. Design and sources:
`research/next-steps/contribution-implementation-plan.md` (decisions D1–D5).

Modules:
    align     — 5-point similarity alignment to the ArcFace template
    embedder  — ArcFace `w600k_r50` embeddings and 3D-68 head pose (ONNX, buffalo_l)
    quality   — per-observation quality score
    aggregate — per-track aggregation modes, outlier gating
    prepass   — one pass over the video producing a `TrackIdentity` per track

LGPD: face embeddings are biometric data (Lei 13.709/2018, Art. 5 II). Every
embedding computed here lives in memory only. Nothing in this package writes
an embedding to disk; reports built on it carry scalars only.
"""
