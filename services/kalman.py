import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: tuple, class_name: str, confidence: float):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.eye(4, 8)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q *= 0.01
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(4, 1)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_name = class_name
        self.confidence = confidence

    def predict(self):
        self.kf.predict()
        cx, cy, w, h = self.kf.x[:4].flatten()
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def update(self, bbox: tuple):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        measurement = np.array([cx, cy, w, h]).reshape(4, 1)
        self.kf.update(measurement)

    def get_velocity(self):
        vx = float(self.kf.x[4, 0])
        vy = float(self.kf.x[5, 0])
        return vx, vy
