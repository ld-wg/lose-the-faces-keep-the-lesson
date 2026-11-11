"""Per-face generation policies (phase2_generate/policies.py) and the shared landmark error."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.identity.align import fit_similarity, landmark_error
from src.pipeline.phase2_generate.policies import changed_fraction, should_generate


# --- changed_fraction: the fail-open guard ---------------------------------------------------

def test_identical_crop_changes_nothing():
    a = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    assert changed_fraction(a, a.copy(), (5, 5, 45, 35)) == 0.0


def test_only_pixels_inside_the_box_count():
    a = np.zeros((40, 50, 3), np.uint8)
    b = a.copy()
    b[10:20, 10:20] = 255
    assert changed_fraction(a, b, (10, 10, 30, 30)) == 0.25
    c = a.copy()
    c[0:5, 0:5] = 1
    assert changed_fraction(a, c, (10, 10, 30, 30)) == 0.0


def test_one_channel_is_enough():
    a = np.zeros((10, 10, 3), np.uint8)
    b = a.copy()
    b[..., 2] = 1
    assert changed_fraction(a, b, (0, 0, 10, 10)) == 1.0


def test_box_is_clipped_and_empty_box_is_zero():
    a = np.zeros((10, 10, 3), np.uint8)
    b = a + 1
    assert changed_fraction(a, b, (-5, -5, 20, 20)) == 1.0
    assert changed_fraction(a, b, (5, 5, 5, 8)) == 0.0


def test_resized_output_is_not_a_passthrough():
    assert changed_fraction(np.zeros((10, 10, 3), np.uint8), np.zeros((12, 10, 3), np.uint8), (0, 0, 5, 5)) == 1.0


# --- should_generate: frame sampling ---------------------------------------------------------

def simulate(n: int, **kw) -> list[int]:
    last, out = None, []
    for f in range(n):
        if should_generate(None if last is None else f - last, **kw):
            last = f
            out.append(f)
    return out


def test_first_frame_always_generates():
    assert should_generate(None, every=5)
    assert should_generate(None, adaptive_tau=0.1, change=0.0)


def test_fixed_sampling():
    assert simulate(10) == list(range(10))
    assert simulate(10, every=3) == [0, 3, 6, 9]
    assert simulate(7, every=2) == [0, 2, 4, 6]


def test_adaptive_sampling():
    assert should_generate(2, adaptive_tau=0.08, max_gap=6, change=0.1)
    assert not should_generate(2, adaptive_tau=0.08, max_gap=6, change=0.05)
    assert should_generate(2, adaptive_tau=0.08, max_gap=6, change=0.08)
    assert should_generate(6, adaptive_tau=0.08, max_gap=6, change=0.0)
    assert should_generate(2, adaptive_tau=0.08, max_gap=6, change=None)   # unknown counts as changed


# --- landmark_error: one definition for the evaluator and the sampler ------------------------

def points(seed: int = 0, n: int = 106) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 100, (n, 2)).astype(np.float32)


def test_zero_for_the_same_points():
    p = points()
    assert landmark_error(p, p.copy(), 10.0) == pytest.approx(0.0, abs=1e-4)


def test_invariant_to_similarity_transforms():
    p = points()
    t = np.deg2rad(20)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32)
    moved = (1.7 * p @ rot.T + np.array([30, -12], np.float32)).astype(np.float32)
    assert landmark_error(p, moved, 10.0) < 1e-3


def test_matches_the_evaluator_formula():
    real, anon, iod = points(1), points(2), 23.0
    m = fit_similarity(anon, real)
    reference = float(np.linalg.norm(anon @ m[:, :2].T + m[:, 2] - real, axis=1).mean() / iod)
    assert landmark_error(real, anon, iod) == reference


def test_in_inter_ocular_units():
    real, anon = points(1), points(2)
    assert landmark_error(real, anon, 10.0) == pytest.approx(2 * landmark_error(real, anon, 20.0))


def test_tolerance_ignores_jpeg_noise_but_not_an_inpainted_face():
    from src.pipeline.phase2_generate.policies import IDENTITY_TOL, MIN_IDENTITY_CHANGE
    rng = np.random.default_rng(5)
    real = rng.integers(40, 200, (60, 60, 3), dtype=np.uint8)
    noisy = np.clip(real.astype(int) + rng.integers(-6, 7, real.shape), 0, 255).astype(np.uint8)
    box = (10, 10, 50, 50)
    assert changed_fraction(real, noisy, box) > 0.5                       # strict: everything "changed"
    assert changed_fraction(real, noisy, box, tol=IDENTITY_TOL) < MIN_IDENTITY_CHANGE
    painted = noisy.copy()
    painted[18:44, 16:44] = rng.integers(0, 256, (26, 28, 3), dtype=np.uint8)
    assert changed_fraction(real, painted, box, tol=IDENTITY_TOL) > MIN_IDENTITY_CHANGE
