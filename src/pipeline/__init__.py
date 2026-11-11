"""Privacy-preserving classroom video pipeline: the thesis's three stages.

    Stage 1  phase1_detect     detection + tracking + recall post-pass
                               (SCRFD-10GF, ByteTrack-style tracker, fill.py)
    Stage 2  phase2_generate   face replacement per track (blur/mosaic, CIAGAN,
                               GANonymization, BLANKET), the P2 push and the release
                               policies (run.py); C2 lives in identity/
    Stage 3  phase2_generate/compose_video.py
                               fail-closed composition of the output video

Temporal consistency comes from one fixed synthetic identity per track (Stage 2) and
the composition's reuse of the track's last generated face (Stage 3); there is no
separate stabilization stage. contracts.py is the data contract between the stages.
"""
