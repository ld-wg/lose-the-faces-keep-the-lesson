"""Statistics and result keys of the experiments harness (experiments/_common.py)."""

from __future__ import annotations

import numpy as np
import pytest

from _common import (_kappa, holm, obs_interval, obs_value, paired_kappa_diff, paired_obs, register_metric,
                     result_key, track_signflip_p)


def rows(values: dict[int, list[float]], metric: str = "m", video: str = "v") -> list[dict]:
    out, fid = [], 0
    for tid, vs in values.items():
        for v in vs:
            out.append({"video": video, "frame_id": fid, "track_id": tid, metric: v})
            fid += 1
    return out


# --- per-face values -------------------------------------------------------------------------

def test_obs_value_ranks_and_columns():
    r = {"facenet_rank": 1, "facenet_cos": 0.4, "arcface_rank": 7, "expression": 0.1, "pose_err": None}
    assert obs_value(r, "facenet.rank1") == 1.0
    assert obs_value(r, "arcface.rank1") == 0.0
    assert obs_value(r, "arcface.rank5") == 0.0
    assert obs_value(r, "facenet.cos") == 0.4
    assert obs_value(r, "expression") == 0.1
    assert obs_value(r, "pose_err") is None
    assert obs_value({}, "facenet.rank1") is None


# --- track bootstrap -------------------------------------------------------------------------

def test_interval_is_face_weighted_and_contains_the_estimate():
    data = rows({1: [1.0] * 9, 2: [0.0]})
    est, lo, hi, n = obs_interval(data, "m")
    assert est == pytest.approx(0.9)
    assert n == 10
    assert lo <= est <= hi


def test_one_track_has_no_spread():
    est, lo, hi, _ = obs_interval(rows({1: [0.2, 0.4]}), "m")
    assert lo == hi == pytest.approx(est)


def test_empty_interval():
    assert obs_interval([], "m") == (None, None, None, 0)


def test_signflip_needs_two_tracks_and_detects_a_shift():
    assert track_signflip_p(rows({1: [1.0]}), "m") is None
    assert track_signflip_p(rows({t: [1.0, 1.0] for t in range(12)}), "m") < 0.01
    balanced = rows({t: [1.0 if t % 2 else -1.0] for t in range(12)})
    assert track_signflip_p(balanced, "m") == pytest.approx(1.0)


def test_paired_obs_joins_on_video_frame_track():
    a = rows({1: [1.0, 1.0], 2: [1.0]})
    b = rows({1: [0.0, 0.0]})          # track 2 only in a
    res = paired_obs(a, b, "m")
    assert res["n"] == 2
    assert res["diff"] == pytest.approx(1.0)


# --- kappa -----------------------------------------------------------------------------------

def test_kappa_perfect_and_chance():
    x = np.array([0, 1, 2, 3] * 10)
    assert _kappa(x, x) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 8, 20000), rng.integers(0, 8, 20000)
    assert abs(_kappa(a, b)) < 0.03


def test_kappa_undefined_when_one_label():
    z = np.zeros(10, dtype=int)
    assert np.isnan(_kappa(z, z))


def test_paired_kappa_diff():
    def labelled(agree: bool) -> list[dict]:
        return [{"video": "v", "frame_id": i, "track_id": i % 6, "r": i % 4, "a": i % 4 if agree else (i + 1) % 4}
                for i in range(60)]
    res = paired_kappa_diff(labelled(True), labelled(False), "r", "a", n=200)
    assert res["n"] == 60
    assert res["diff"] > 1.0          # 1 minus a negative kappa
    assert res["excludes_zero"]


# --- multiple comparisons and keys -----------------------------------------------------------

def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)   # monotone: never below the previous step
    assert adj["d"] is None


def test_holm_caps_at_one():
    assert holm({"a": 0.5, "b": 0.6}) == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}


def test_result_key_is_ascii():
    assert result_key("e3c/demo1/β 1.2/κ") == "e3c/demo1/b-1.2/kappa"
    assert result_key("Δ ≥ 96°") == "d-ge-96deg"


def test_register_metric_rejects_collisions():
    store, sources = {}, {}
    register_metric(store, sources, "e3/β 1.2", 1)
    register_metric(store, sources, "e3/β 1.2", 2)          # same key twice: an update
    assert store == {"e3/b-1.2": 2}
    with pytest.raises(ValueError, match="collision"):
        register_metric(store, sources, "e3/b 1.2", 3)
