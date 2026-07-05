"""Per-track Kalman filter for bounding-box smoothing.

State vector (8-dim)::
    [cx, cy, w, h, vx, vy, vw, vh]

Observation vector (4-dim)::
    [cx, cy, w, h]

The filter is a constant-velocity model. Process noise is tuned for
25 FPS traffic footage; tune ``Q`` down for slower streams or up for
faster ones.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """One Kalman filter per tracked object."""

    count = 0  # monotonic ID generator for fresh trackers

    def __init__(self, bbox: Tuple[int, int, int, int],
                 class_name: str, confidence: float) -> None:
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition: constant-velocity model.
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Measurement: we observe [cx, cy, w, h].
        self.kf.H = np.eye(4, 8)

        # Noise covariances — see module docstring for tuning notes.
        self.kf.R[2:, 2:] *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q *= 0.01
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(4, 1)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_name = class_name
        self.confidence = confidence

    # ----- API --------------------------------------------------------------
    def predict(self) -> np.ndarray:
        """Advance state by one step and return the predicted bbox."""
        self.kf.predict()
        cx, cy, w, h = self.kf.x[:4].flatten()
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """Correct the state with a fresh observation."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        self.kf.update(np.array([cx, cy, w, h]).reshape(4, 1))

    def get_velocity(self) -> Tuple[float, float]:
        """Return (vx, vy) in pixels-per-frame."""
        return float(self.kf.x[4, 0]), float(self.kf.x[5, 0])

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the monotonic ID counter (mainly for tests)."""
        cls.count = 0
