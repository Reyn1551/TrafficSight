"""Per-frame detection + tracking + speed + persistence loop.

``DetectionThread`` is a ``QThread`` that:

1. Loads the YOLO model on startup (so the UI does not block).
2. Receives frames from ``VideoThread`` via :meth:`update_frame`.
3. Runs ``YOLO.track`` to get per-frame boxes + persistent IDs.
4. Maintains a :class:`KalmanBoxTracker` per active ID for smoothing.
5. Computes speed via :class:`SpeedEstimator` and direction via
   :meth:`_classify_direction`.
6. Persists a :class:`DetectionEvent` to the repository every
   ``_db_interval`` seconds per track (throttled).
7. Notifies the :class:`VirtualLineCounter` of new centroids so line
   crossings can be detected.

The thread is **idempotent**: dropping a frame simply delays the next
inference cycle. It never blocks the UI.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import QThread

from ..config import cfg
from ..domain.entities import DetectionEvent, DetectionResult
from ..domain.ports import DetectionRepository
from ..logger import write_log
from ..adapters.postgres_repository import PostgresDetectionRepository
from .geospatial import SpeedEstimator
from .kalman import KalmanBoxTracker
from .line_counter import VirtualLineCounter


class DetectionThread(QThread):
    """QThread that runs YOLO inference and downstream analytics."""

    def __init__(self, model_path: str, camera_name: str = "Unknown",
                 line_counter: Optional[VirtualLineCounter] = None,
                 repo: Optional[DetectionRepository] = None,
                 db_write_interval: float = 2.0) -> None:
        super().__init__()
        self.model_path = model_path
        self.camera_name = camera_name
        self.line_counter = line_counter
        self.repo: DetectionRepository = repo or PostgresDetectionRepository()
        self._db_interval = db_write_interval

        self.running = True
        self._lock = threading.Lock()
        self._frame_to_process: Optional[object] = None
        self._detections: list[DetectionResult] = []

        self._kf_trackers: dict[int, KalmanBoxTracker] = {}
        self._last_db_write: dict[int, float] = defaultdict(float)
        self.speed_estimator = SpeedEstimator()
        self.model = None  # lazily loaded in run()

    # ----- QThread ----------------------------------------------------------
    def run(self) -> None:  # noqa: D401 - QThread contract
        try:
            from ultralytics import YOLO

            write_log(f"Loading YOLO model {self.model_path}...")
            self.model = YOLO(self.model_path)
            write_log("YOLO model loaded.")
        except Exception as exc:  # pragma: no cover - depends on ultralytics
            write_log(f"Error loading YOLO model: {exc}")
            return

        while self.running:
            frame = self._take_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            self._process_frame(frame)

    # ----- public API -------------------------------------------------------
    def update_frame(self, frame) -> None:
        """Submit a new frame for inference (overwrites any pending one)."""
        with self._lock:
            self._frame_to_process = frame

    def get_detections(self) -> list[DetectionResult]:
        with self._lock:
            return list(self._detections)

    def stop(self) -> None:
        self.running = False
        self.wait()
        self.repo.close()

    # ----- internals --------------------------------------------------------
    def _take_frame(self):
        with self._lock:
            frame = self._frame_to_process
            self._frame_to_process = None
            return frame

    def _process_frame(self, frame) -> None:
        try:
            results = self.model.track(frame, persist=True, verbose=False)
        except Exception as exc:
            write_log(f"Inference error: {exc}")
            return

        current_ids: list[int] = []
        detections: list[DetectionResult] = []

        for r in results:
            for box in r.boxes:
                if box.id is None:
                    continue
                obj_id = int(box.id[0].cpu().numpy())
                current_ids.append(obj_id)

                b = box.xyxy[0].cpu().numpy().astype(int)
                name = self.model.names[int(box.cls[0].cpu().numpy())]
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = b

                tracker = self._get_or_create_tracker(obj_id, b, name, conf)
                tracker.predict()
                tracker.update(b)
                vx, vy = tracker.get_velocity()

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                speed_kmh = self.speed_estimator.estimate_speed(
                    obj_id, (vx, vy), cx, cy, frame.shape[0]
                )
                speed_kmh = min(speed_kmh, cfg.speed_cap_kmh)
                direction = self._classify_direction(vx, vy)

                if self.line_counter is not None:
                    self.line_counter.update(
                        obj_id, cx, cy, name, speed_kmh, self.camera_name
                    )

                self._maybe_persist_detection(
                    obj_id, name, cx, cy, speed_kmh, direction
                )

                detections.append(
                    DetectionResult(
                        bbox=tuple(b.tolist()),
                        class_name=name,
                        confidence=conf,
                        speed_kmh=speed_kmh,
                        track_id=obj_id,
                        vx=vx,
                        vy=vy,
                        direction=direction,
                    )
                )

        self._forget_stale(current_ids)

        with self._lock:
            self._detections = detections

    def _get_or_create_tracker(self, obj_id: int, bbox, name: str,
                               conf: float) -> KalmanBoxTracker:
        if obj_id not in self._kf_trackers:
            self._kf_trackers[obj_id] = KalmanBoxTracker(bbox, name, conf)
        return self._kf_trackers[obj_id]

    def _maybe_persist_detection(self, obj_id: int, name: str,
                                 cx: int, cy: float, speed_kmh: float,
                                 direction: str) -> None:
        now = time.time()
        if now - self._last_db_write[obj_id] < self._db_interval:
            return
        self.repo.insert_detection(
            DetectionEvent(
                timestamp=datetime.now(),
                camera=self.camera_name,
                track_id=obj_id,
                class_name=name,
                speed_kmh=speed_kmh,
                cx=cx,
                cy=cy,
                direction=direction,
                is_overspeed=speed_kmh > cfg.overspeed_kmh,
            )
        )
        self._last_db_write[obj_id] = now

    def _forget_stale(self, current_ids: list[int]) -> None:
        stale = [tid for tid in self._kf_trackers if tid not in current_ids]
        for tid in stale:
            self._kf_trackers.pop(tid, None)
            self.speed_estimator.forget(tid)
            self._last_db_write.pop(tid, None)

    @staticmethod
    def _classify_direction(vx: float, vy: float) -> str:
        if abs(vx) < 0.5 and abs(vy) < 0.5:
            return "Diam"
        angle = math.degrees(math.atan2(-vy, vx))
        if -45 <= angle < 45:
            return "→ Timur"
        if 45 <= angle < 135:
            return "↑ Utara"
        if angle >= 135 or angle < -135:
            return "← Barat"
        return "↓ Selatan"
