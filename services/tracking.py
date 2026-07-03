import time
import math
import threading
from datetime import datetime
from collections import defaultdict
from PyQt6.QtCore import QThread
from services.geospatial import SpeedEstimator
from logger import write_log
from adapters.postgres_repository import PostgresDetectionRepository
from domain.entities import DetectionResult, DetectionEvent, LineCrossingEvent
from services.kalman import KalmanBoxTracker
from services.line_counter import VirtualLineCounter


class DetectionThread(QThread):
    def __init__(self, model_path: str, camera_name: str = "Unknown", line_counter=None):
        super().__init__()
        self.model_path = model_path
        self.camera_name = camera_name
        self.running = True
        self._lock = threading.Lock()
        self.frame_to_process = None
        self.detections = []

        self.kf_trackers = {}
        self.speed_estimator = SpeedEstimator()
        self.repo = PostgresDetectionRepository()
        self.line_counter = line_counter
        self._last_db_write = defaultdict(float)
        self._db_interval = 2.0

    def run(self):
        try:
            from ultralytics import YOLO
            write_log(f"Loading YOLO model {self.model_path}...")
            self.model = YOLO(self.model_path)
            write_log("YOLO model loaded.")
        except Exception as exc:
            write_log(f"Error loading YOLO model: {exc}")
            return

        while self.running:
            frame = None
            with self._lock:
                frame = self.frame_to_process
                self.frame_to_process = None

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                results = self.model.track(frame, persist=True, verbose=False)
                current_ids = []
                detections = []

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

                        if obj_id not in self.kf_trackers:
                            self.kf_trackers[obj_id] = KalmanBoxTracker(b, name, conf)

                        tracker = self.kf_trackers[obj_id]
                        tracker.predict()
                        tracker.update(b)
                        vx, vy = tracker.get_velocity()

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        speed_kmh = self.speed_estimator.estimate_speed(obj_id, (vx, vy), cx, cy, frame.shape[0])
                        speed_kmh = min(speed_kmh, 140.0)
                        direction = self._classify_direction(vx, vy)

                        if self.line_counter:
                            direction_label = self.line_counter.update(obj_id, cx, cy, name, speed_kmh, self.camera_name)
                        else:
                            direction_label = None

                        now = time.time()
                        if now - self._last_db_write[obj_id] >= self._db_interval:
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
                                    is_overspeed=speed_kmh > 60.0,
                                )
                            )
                            self._last_db_write[obj_id] = now

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

                stale_ids = [tid for tid in self.kf_trackers if tid not in current_ids]
                for tid in stale_ids:
                    self.kf_trackers.pop(tid, None)
                    self.speed_estimator.speed_smoothing.pop(tid, None)
                    self._last_db_write.pop(tid, None)

                with self._lock:
                    self.detections = detections
            except Exception as exc:
                write_log(f"Inference error: {exc}")

    def update_frame(self, frame):
        with self._lock:
            self.frame_to_process = frame

    def get_detections(self):
        with self._lock:
            return list(self.detections)

    def stop(self):
        self.running = False
        self.wait()
        self.repo.close()

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
