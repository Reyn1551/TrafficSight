import os
import json
import math

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from collections import defaultdict, deque
from config import GEOSPATIAL_CALIB_FILE, FALLBACK_FPS
from logger import write_log


class SpeedEstimator:
    def __init__(self, fps: float = FALLBACK_FPS):
        self.fps = fps
        self.speed_smoothing = defaultdict(lambda: deque(maxlen=10))
        self.calibration_data = None
        self._load_calibration()

    def _load_calibration(self) -> None:
        if os.path.exists(GEOSPATIAL_CALIB_FILE):
            try:
                with open(GEOSPATIAL_CALIB_FILE, "r", encoding="utf-8") as file:
                    self.calibration_data = json.load(file)
                write_log(f"Berhasil memuat data kalibrasi geospasial: {GEOSPATIAL_CALIB_FILE}")
            except Exception as exc:
                write_log(f"Gagal memuat {GEOSPATIAL_CALIB_FILE}: {exc}")
        else:
            write_log("Peringatan: File kalibrasi geospasial tidak ditemukan. Menggunakan estimasi fallback.")

    def estimate_speed(self, track_id: int, velocity: tuple, cx: int, cy: int, max_y: int) -> float:
        vx, vy = velocity

        if self.calibration_data and self.calibration_data.get("calibrated") and cv2 is not None:
            transform = np.array(self.calibration_data.get("transform_matrix", []), dtype=np.float32)
            ppm_x = self.calibration_data.get("ppm_x", 1.0)
            ppm_y = self.calibration_data.get("ppm_y", 1.0)

            p1 = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
            p2 = np.array([[[float(cx + vx), float(cy + vy)]]], dtype=np.float32)
            p1w = cv2.perspectiveTransform(p1, transform)
            p2w = cv2.perspectiveTransform(p2, transform)

            dx_m = (p2w[0, 0, 0] - p1w[0, 0, 0]) / max(ppm_x, 1e-6)
            dy_m = (p2w[0, 0, 1] - p1w[0, 0, 1]) / max(ppm_y, 1e-6)
            speed_kmh = math.sqrt(dx_m**2 + dy_m**2) * self.fps * 3.6
        else:
            horizon_y = 200
            t = max(0.0, min(1.0, (cy - horizon_y) / max(max_y - horizon_y, 1)))
            ppm = 1.0 / (0.4 * (1 - t) + 0.02 * t)
            speed_kmh = math.sqrt(vx**2 + vy**2) * self.fps / ppm * 3.6

        buffer = self.speed_smoothing[track_id]
        if buffer:
            smoothed = 0.35 * speed_kmh + 0.65 * buffer[-1]
        else:
            smoothed = speed_kmh

        buffer.append(smoothed)
        return round(smoothed, 1)
