"""Privacy-preserving classroom video pipeline.

Phases map to research/pipeline.md:
    phase1_detect    — face detection + tracking (SCRFD-10GF + ByteTrack)
    phase2_generate  — synthetic face generation + inpainting (diffusion/GAN)
    phase3_temporal  — temporal consistency / identity locking
"""
