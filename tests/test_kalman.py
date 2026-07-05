"""Unit tests for :class:`trafficsight.services.kalman.KalmanBoxTracker`."""

from __future__ import annotations

import numpy as np

from trafficsight.services.kalman import KalmanBoxTracker


def test_initial_state_matches_bbox():
    KalmanBoxTracker.reset_counter()
    tracker = KalmanBoxTracker((100, 200, 200, 280), "car", 0.95)
    pred = tracker.predict()
    # After predict() the state should still be very close to the init bbox.
    np.testing.assert_allclose(pred, [100, 200, 200, 280], atol=1.0)


def test_velocity_starts_at_zero():
    KalmanBoxTracker.reset_counter()
    tracker = KalmanBoxTracker((100, 100, 200, 200), "car", 0.9)
    vx, vy = tracker.get_velocity()
    assert vx == 0.0
    assert vy == 0.0


def test_velocity_reflects_motion():
    KalmanBoxTracker.reset_counter()
    tracker = KalmanBoxTracker((100, 100, 110, 110), "car", 0.9)
    # Move right by 10 px for several frames.
    for _ in range(5):
        tracker.predict()
        tracker.update((110, 100, 120, 110))
    vx, _ = tracker.get_velocity()
    assert vx > 0


def test_id_counter_monotonic():
    KalmanBoxTracker.reset_counter()
    a = KalmanBoxTracker((0, 0, 10, 10), "car", 0.9)
    b = KalmanBoxTracker((0, 0, 10, 10), "car", 0.9)
    assert b.id == a.id + 1
