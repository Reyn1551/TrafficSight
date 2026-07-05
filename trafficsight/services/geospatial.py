"""Geospatial speed estimation.

Two paths:

1. **Calibrated** — uses a 4-point homography stored in
   :data:`cfg.geospatial_calib_file`. Pixel velocities are projected into
   the bird's-eye plane, divided by ``ppm_x`` / ``ppm_y`` to get meters,
   and converted to km/h with ``× fps × 3.6``.
2. **Fallback** — when no calibration file is present, a perspective
   heuristic assumes the horizon is at ``y=200`` and linearly interpolates
   ``ppm`` from 1/0.4 (near) to 1/0.02 (far). This produces *relative*
   speeds only — use it for trend analysis, not absolute numbers.

A per-track EMA smoother (0.35·new + 0.65·prev) damps frame-to-frame
jitter before the speed is returned.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a hard runtime dep
    cv2 = None  # type: ignore[assignment]

from ..config import cfg
from ..logger import write_log


_HORIZON_Y = 200
_NEAR_PPM = 0.4   # larger near-camera objects
_FAR_PPM = 0.02   # smaller far-camera objects
_EMA_ALPHA = 0.35
_SMOOTH_WINDOW = 10


class SpeedEstimator:
    """Per-track speed smoother with optional homography projection."""

    def __init__(self, fps: Optional[float] = None) -> None:
        self.fps: float = fps or cfg.fallback_fps
        self.speed_smoothing: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=_SMOOTH_WINDOW)
        )
        self.calibration_data: Optional[Dict[str, Any]] = None
        self._transform: Optional[np.ndarray] = None
        self._ppm_x: float = 1.0
        self._ppm_y: float = 1.0
        self._load_calibration()

    # ----- calibration ------------------------------------------------------
    def _load_calibration(self) -> None:
        path = cfg.geospatial_calib_file
        if not os.path.exists(path):
            write_log(
                "Peringatan: File kalibrasi geospasial tidak ditemukan. "
                "Menggunakan estimasi fallback (perspektif)."
            )
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            write_log(f"Gagal memuat {path}: {exc}")
            return

        if not (data and data.get("calibrated")):
            write_log("Kalibrasi geospasial ada tapi belum dikalibrasi.")
            return

        self.calibration_data = data
        try:
            self._transform = np.array(
                data.get("transform_matrix", []), dtype=np.float32
            )
        except (TypeError, ValueError) as exc:
            write_log(f"Transform matrix tidak valid: {exc}")
            return
        self._ppm_x = float(data.get("ppm_x", 1.0)) or 1.0
        self._ppm_y = float(data.get("ppm_y", 1.0)) or 1.0
        write_log(f"Berhasil memuat data kalibrasi geospasial: {path}")

    # ----- estimation -------------------------------------------------------
    def estimate_speed(self, track_id: int, velocity: Tuple[float, float],
                       cx: int, cy: int, max_y: int) -> float:
        vx, vy = velocity

        if self._transform is not None and cv2 is not None:
            speed_kmh = self._calibrated_speed(vx, vy, cx, cy)
        else:
            speed_kmh = self._fallback_speed(vx, vy, cy, max_y)

        speed_kmh = min(speed_kmh, cfg.speed_cap_kmh)
        return self._smooth(track_id, speed_kmh)

    # ----- paths ------------------------------------------------------------
    def _calibrated_speed(self, vx: float, vy: float,
                          cx: int, cy: int) -> float:
        assert cv2 is not None and self._transform is not None
        p1 = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        p2 = np.array([[[float(cx + vx), float(cy + vy)]]], dtype=np.float32)
        p1w = cv2.perspectiveTransform(p1, self._transform)
        p2w = cv2.perspectiveTransform(p2, self._transform)
        dx_m = (p2w[0, 0, 0] - p1w[0, 0, 0]) / max(self._ppm_x, 1e-6)
        dy_m = (p2w[0, 0, 1] - p1w[0, 0, 1]) / max(self._ppm_y, 1e-6)
        return math.sqrt(dx_m ** 2 + dy_m ** 2) * self.fps * 3.6

    def _fallback_speed(self, vx: float, vy: float,
                        cy: int, max_y: int) -> float:
        t = max(0.0, min(1.0, (cy - _HORIZON_Y) / max(max_y - _HORIZON_Y, 1)))
        ppm = 1.0 / (_NEAR_PPM * (1 - t) + _FAR_PPM * t)
        return math.sqrt(vx ** 2 + vy ** 2) * self.fps / ppm * 3.6

    def _smooth(self, track_id: int, speed_kmh: float) -> float:
        buffer = self.speed_smoothing[track_id]
        if buffer:
            smoothed = _EMA_ALPHA * speed_kmh + (1 - _EMA_ALPHA) * buffer[-1]
        else:
            smoothed = speed_kmh
        buffer.append(smoothed)
        return round(smoothed, 1)

    # ----- maintenance ------------------------------------------------------
    def forget(self, track_id: int) -> None:
        """Drop smoothing state for a track that has disappeared."""
        self.speed_smoothing.pop(track_id, None)
