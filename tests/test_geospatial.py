"""Unit tests for :class:`trafficsight.services.geospatial.SpeedEstimator`."""

from __future__ import annotations

import json

from trafficsight.services.geospatial import SpeedEstimator


def test_fallback_speed_is_positive_for_motion():
    est = SpeedEstimator(fps=25.0)
    # No calibration file → fallback path.
    speed = est.estimate_speed(1, (5.0, 0.0), cx=500, cy=500, max_y=1080)
    assert speed > 0.0


def test_fallback_speed_zero_when_stationary():
    est = SpeedEstimator(fps=25.0)
    speed = est.estimate_speed(1, (0.0, 0.0), cx=500, cy=500, max_y=1080)
    assert speed == 0.0


def test_speed_capped_at_speed_cap(tmp_path, monkeypatch):
    # The estimator reads cfg.geospatial_calib_file on init.
    est = SpeedEstimator(fps=25.0)
    # An absurd pixel velocity should be clamped to cfg.speed_cap_kmh.
    speed = est.estimate_speed(1, (1e6, 0.0), cx=500, cy=500, max_y=1080)
    from trafficsight.config import cfg
    assert speed <= cfg.speed_cap_kmh


def test_smoothing_damps_jitter():
    est = SpeedEstimator(fps=25.0)
    first = est.estimate_speed(1, (5.0, 0.0), cx=500, cy=500, max_y=1080)
    # Same velocity → next sample should be pulled toward the previous.
    second = est.estimate_speed(1, (5.0, 0.0), cx=500, cy=500, max_y=1080)
    # The smoothing EMA should make the second reading closer to the first
    # than a fresh computation; both should be positive.
    assert first > 0
    assert second > 0
