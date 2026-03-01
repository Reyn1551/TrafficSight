import subprocess
import numpy as np
import cv2
import threading
import queue
import time
import os
import sys
import json
import sqlite3
import math
from datetime import datetime
from collections import defaultdict, deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QStatusBar,
                            QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from filterpy.kalman import KalmanFilter

# ============================================================
#  SIGAP — Sistem Intelijen Geospasial Analitik Pemantauan
#  Lalu Lintas Kota Yogyakarta
#  v2.0 — Fixed cx bug + SQLite logging + Virtual Line Counter
# ============================================================

STREAM_URLS = {
    "Sugeng Jeroni 2":       "http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8",
    "Simpang Wirosaban Barat":"https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/playlist.m3u8",
    "Wirobrajan":            "https://cctvjss.jogjakota.go.id/atcs/ATCS_wirobrajan.stream/playlist.m3u8",
}
CURRENT_STREAM_URL = STREAM_URLS["Sugeng Jeroni 2"]

WIDTH          = 1920
HEIGHT         = 1080
LOG_FILE       = "sigap_log.txt"
DB_FILE        = "sigap_traffic.db"
BUFFER_SECONDS = 60
FALLBACK_FPS   = 25.0
OVERSPEED_KMH  = 60.0
SPEED_CAP_KMH  = 140.0
LINES_FILE     = "counting_lines.json"

# ── Virtual counting line (4 Lengan Persimpangan) ──
COUNTING_LINES = {
    "Utara":  {"type": "H", "y": 310, "x1": 350, "x2": 700},
    "Selatan":{"type": "H", "y": 580, "x1": 350, "x2": 750},
    "Barat":  {"type": "V", "x": 150, "y1": 310, "y2": 580},
    "Timur":  {"type": "V", "x": 970, "y1": 260, "y2": 520},
}

def load_lines_config():
    global COUNTING_LINES
    if os.path.exists(LINES_FILE):
        try:
            with open(LINES_FILE, "r") as f:
                saved = json.load(f)
                if CURRENT_STREAM_URL in saved:
                    COUNTING_LINES.clear()
                    COUNTING_LINES.update(saved[CURRENT_STREAM_URL])
        except Exception as e:
            write_log(f"Gagal memuat {LINES_FILE}: {e}")

def save_lines_config():
    saved = {}
    if os.path.exists(LINES_FILE):
        with open(LINES_FILE, "r") as f:
            try:
                saved = json.load(f)
            except: pass
    saved[CURRENT_STREAM_URL] = COUNTING_LINES
    with open(LINES_FILE, "w") as f:
        json.dump(saved, f, indent=4)

# ===============================================================
#  LOGGING
# ===============================================================
def write_log(text):
    ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {text}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ===============================================================
#  DATABASE (SQLite)
# ===============================================================
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            camera      TEXT    NOT NULL,
            track_id    INTEGER NOT NULL,
            class_name  TEXT    NOT NULL,
            speed_kmh   REAL    NOT NULL,
            cx          INTEGER,
            cy          INTEGER,
            direction   TEXT,
            is_overspeed INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS line_crossings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            camera      TEXT    NOT NULL,
            track_id    INTEGER NOT NULL,
            class_name  TEXT    NOT NULL,
            speed_kmh   REAL    NOT NULL,
            direction   TEXT
        )
    """)
    conn.commit()
    write_log(f"Database siap: {DB_FILE}")
    return conn

DB_CONN = init_db()
DB_LOCK = threading.Lock()

def db_insert_detection(camera, track_id, class_name, speed_kmh, cx, cy, direction):
    ts = datetime.now().isoformat(sep=" ", timespec="milliseconds")
    with DB_LOCK:
        DB_CONN.execute(
            "INSERT INTO detections (timestamp,camera,track_id,class_name,speed_kmh,cx,cy,direction,is_overspeed) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, camera, track_id, class_name, round(speed_kmh,1), cx, cy, direction,
             1 if speed_kmh > OVERSPEED_KMH else 0)
        )
        DB_CONN.commit()

def db_insert_crossing(camera, track_id, class_name, speed_kmh, direction):
    ts = datetime.now().isoformat(sep=" ", timespec="milliseconds")
    with DB_LOCK:
        DB_CONN.execute(
            "INSERT INTO line_crossings (timestamp,camera,track_id,class_name,speed_kmh,direction) "
            "VALUES (?,?,?,?,?,?)",
            (ts, camera, track_id, class_name, round(speed_kmh,1), direction)
        )
        DB_CONN.commit()


# ===============================================================
#  FPS DETECTION
# ===============================================================
def detect_stream_fps(url, timeout=15):
    write_log("Mendeteksi FPS asli stream...")
    cmd = ['ffprobe','-v','error','-select_streams','v:0',
           '-show_entries','stream=r_frame_rate,avg_frame_rate',
           '-of','csv=p=0', url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        for part in result.stdout.replace('\n',',').split(','):
            part = part.strip()
            if '/' in part:
                try:
                    n, d = part.split('/')
                    fps = float(n)/float(d)
                    if 1.0 < fps < 120.0:
                        write_log(f"FPS terdeteksi: {fps:.3f}")
                        return fps
                except Exception:
                    pass
            elif part:
                try:
                    fps = float(part)
                    if 1.0 < fps < 120.0:
                        write_log(f"FPS terdeteksi (plain): {fps:.3f}")
                        return fps
                except Exception:
                    pass
    except Exception as e:
        write_log(f"ffprobe error: {e}")
    write_log(f"Pakai fallback FPS: {FALLBACK_FPS}")
    return FALLBACK_FPS


# ===============================================================
#  STABLE STREAMER
# ===============================================================
class StableStreamer:
    def __init__(self, src, width, height, fps):
        self.src, self.width, self.height, self.fps = src, width, height, fps
        self.buffer_size = int(BUFFER_SECONDS * fps)
        self.q       = queue.Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.proc    = None
        self._lock   = threading.Lock()
        self.cmd = [
            'ffmpeg','-reconnect','1','-reconnect_streamed','1',
            '-reconnect_delay_max','10','-i',src,
            '-vsync','passthrough','-f','rawvideo',
            '-pix_fmt','bgr24','-s',f'{width}x{height}',
            '-loglevel','error','pipe:1'
        ]

    def start(self):
        self.proc   = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, bufsize=10**8)
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        frame_size = self.width * self.height * 3
        while not self.stopped:
            raw = self.proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                write_log("Koneksi putus, mencoba reconnect...")
                with self._lock: self.proc.kill()
                time.sleep(2)
                with self._lock:
                    self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE, bufsize=10**8)
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            try:
                self.q.put(frame, timeout=5.0)
            except queue.Full:
                pass   # drop frame silently

    def read(self, timeout=10.0):
        return self.q.get(timeout=timeout)

    def queue_size(self):
        return self.q.qsize()

    def stop(self):
        self.stopped = True
        with self._lock:
            if self.proc: self.proc.kill()


# ===============================================================
#  HELPER: ARAH GERAK
# ===============================================================
def classify_direction(vx: float, vy: float) -> str:
    if abs(vx) < 0.5 and abs(vy) < 0.5:
        return "Diam"
    angle = math.degrees(math.atan2(-vy, vx))
    if -45 <= angle < 45:   return "→ Timur"
    if 45  <= angle < 135:  return "↑ Utara"
    if angle >= 135 or angle < -135: return "← Barat"
    return "↓ Selatan"


# ===============================================================
#  KALMAN TRACKER
# ===============================================================
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: tuple, class_name: str, confidence: float):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],
        ])
        self.kf.H = np.eye(4, 8)
        self.kf.R[2:,2:] *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.Q       *= 0.01
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P        *= 10.0

        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w,  h  = x2-x1,      y2-y1
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(4,1)

        self.id         = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_name = class_name
        self.confidence = confidence

    def predict(self):
        self.kf.predict()
        cx, cy, w, h = self.kf.x[:4].flatten()
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

    def update(self, bbox: tuple):
        x1,y1,x2,y2 = bbox
        cx,cy = (x1+x2)/2, (y1+y2)/2
        w, h  = x2-x1,      y2-y1
        self.kf.update(np.array([cx,cy,w,h]).reshape(4,1))

    def get_velocity(self):
        return float(self.kf.x[4,0]), float(self.kf.x[5,0])


# ===============================================================
#  SPEED ESTIMATOR — BUG cx SUDAH DIPERBAIKI
# ===============================================================
class SpeedEstimator:
    def __init__(self, fps: float = FALLBACK_FPS):
        self.fps = fps
        self.speed_smoothing = defaultdict(lambda: deque(maxlen=10))
        self.calibration_data = None
        self._load_calibration()

    def _load_calibration(self):
        calib_file = "geospatial_calibration.json"
        if os.path.exists(calib_file):
            try:
                with open(calib_file) as f:
                    self.calibration_data = json.load(f)
                # ── Hitung ppm secara dinamis dari data kalibrasi ──
                w_res  = self.calibration_data["warped_resolution"]["width"]
                h_res  = self.calibration_data["warped_resolution"]["height"]
                w_real = self.calibration_data["real_world_m"]["width"]
                l_real = self.calibration_data["real_world_m"]["length"]
                self.calibration_data["ppm_x"] = w_res / w_real
                self.calibration_data["ppm_y"] = h_res / l_real
                write_log(f"Kalibrasi geospasial dimuat. ppm_x={self.calibration_data['ppm_x']:.2f} ppm_y={self.calibration_data['ppm_y']:.2f}")
            except Exception as e:
                write_log(f"Gagal memuat kalibrasi: {e}")
        else:
            write_log("⚠ File kalibrasi tidak ditemukan — pakai fallback perspektif.")

    # ─────────────────────────────────────────────────────────
    #  estimate_speed — cx SEKARANG DITERIMA DAN DIPAKAI
    # ─────────────────────────────────────────────────────────
    def estimate_speed(self, track_id: int, velocity: tuple,
                       cx: int, cy: int, max_y: int) -> float:
        vx, vy = velocity

        if self.calibration_data and self.calibration_data.get("calibrated"):
            M     = np.array(self.calibration_data["transform_matrix"], dtype=np.float32)
            ppm_x = self.calibration_data["ppm_x"]
            ppm_y = self.calibration_data["ppm_y"]

            # ✅ Gunakan koordinat KENDARAAN ITU SENDIRI (cx, cy), bukan tengah frame
            p1 = np.array([[[float(cx),      float(cy)     ]]], dtype=np.float32)
            p2 = np.array([[[float(cx + vx), float(cy + vy)]]], dtype=np.float32)

            p1w = cv2.perspectiveTransform(p1, M)
            p2w = cv2.perspectiveTransform(p2, M)

            dx_m = (p2w[0,0,0] - p1w[0,0,0]) / ppm_x
            dy_m = (p2w[0,0,1] - p1w[0,0,1]) / ppm_y

            speed_kmh = math.sqrt(dx_m**2 + dy_m**2) * self.fps * 3.6
        else:
            # Fallback perspektif lama
            horizon_y = 200
            t = max(0.0, min(1.0, (cy - horizon_y) / max(max_y - horizon_y, 1)))
            ppm = 1.0 / (0.4*(1-t) + 0.02*t)
            speed_kmh = math.sqrt(vx**2 + vy**2) * self.fps / ppm * 3.6

        # Exponential weighted smoothing (lebih responsif dari rata-rata sederhana)
        alpha = 0.35
        buf   = self.speed_smoothing[track_id]
        if buf:
            smoothed = alpha * speed_kmh + (1 - alpha) * buf[-1]
        else:
            smoothed = speed_kmh
        buf.append(smoothed)
        return round(smoothed, 1)


# ===============================================================
#  VIRTUAL LINE COUNTER
# ===============================================================
class VirtualLineCounter:
    def __init__(self):
        self.prev_pos   = {}   # track_id → (cx, cy)
        self.counted    = defaultdict(set)  # arm → set of track_id
        self.counts_arm = defaultdict(lambda: defaultdict(int))
        self.unique_total = 0
        self._lock      = threading.Lock()

    def update(self, track_id: int, cx: int, cy: int, class_name: str,
               speed_kmh: float, camera: str) -> bool:
        """Returns True jika crossing terjadi pada frame ini."""
        with self._lock:
            prev = self.prev_pos.get(track_id)
            self.prev_pos[track_id] = (cx, cy)
            if prev is None:
                return False

            px, py = prev
            crossed_any = False
            for arm, cfg in COUNTING_LINES.items():
                if track_id in self.counted[arm]:
                    continue
                crossed = False
                if cfg["type"] == "H":
                    # Garis horizontal: cek apakah lintasi y, dan cx dalam range x
                    in_range = cfg["x1"] <= cx <= cfg["x2"]
                    crossed  = in_range and ((py < cfg["y"] <= cy) or (py > cfg["y"] >= cy))
                else:
                    # Garis vertikal: cek apakah lintasi x, dan cy dalam range y
                    in_range = cfg["y1"] <= cy <= cfg["y2"]
                    crossed  = in_range and ((px < cfg["x"] <= cx) or (px > cfg["x"] >= cx))

                if crossed:
                    # Tentukan event masuk / keluar berdasarkan lengan persimpangan
                    if arm == "Utara":
                        event = "masuk" if cy > py else "keluar"
                    elif arm == "Selatan":
                        event = "masuk" if cy < py else "keluar"
                    elif arm == "Barat":
                        event = "masuk" if cx > px else "keluar"
                    elif arm == "Timur":
                        event = "masuk" if cx < px else "keluar"
                    else:
                        event = "lintas"

                    # Hitung unique vehicles
                    is_new_unique = True
                    for a, track_set in self.counted.items():
                        if track_id in track_set:
                            is_new_unique = False
                            break
                    if is_new_unique:
                        self.unique_total += 1

                    self.counted[arm].add(track_id)
                    self.counts_arm[arm][event] += 1
                    
                    # Store as Arm-Event
                    dir_label = f"{arm}-{event}"
                    db_insert_crossing(camera, track_id, class_name, speed_kmh, dir_label)
                    write_log(f"[LINE] {class_name} #{track_id} {dir_label} {speed_kmh:.1f} km/h")
                    crossed_any = True

            return crossed_any

    def get_summary(self) -> dict:
        with self._lock:
            # Mengembalikan total unik dan dictionary masuk/keluar per lengan
            return {"unique_total": self.unique_total, "per_arm": {k: dict(v) for k, v in self.counts_arm.items()}}

    def remove(self, track_id: int):
        with self._lock:
            self.prev_pos.pop(track_id, None)
            for arm in self.counted.values():
                arm.discard(track_id)


# ===============================================================
#  DETECTION THREAD
# ===============================================================
class DetectionThread(QThread):
    def __init__(self, model_path: str, camera_name: str = "Unknown"):
        super().__init__()
        self.model_path  = model_path
        self.camera_name = camera_name
        self.running     = True
        self._lock       = threading.Lock()
        self.frame_to_process = None
        self.detections       = []

        self.kf_trackers    = {}
        self.speed_estimator = SpeedEstimator(fps=FALLBACK_FPS)
        self.line_counter    = VirtualLineCounter()

        # Throttle DB insert: jangan tulis setiap frame
        self._last_db_write  = defaultdict(float)
        self._DB_INTERVAL    = 2.0  # detik antar insert per kendaraan

    def run(self):
        try:
            from ultralytics import YOLO
            write_log(f"Loading YOLO model {self.model_path}...")
            self.model = YOLO(self.model_path)
            write_log("YOLO model loaded.")
        except Exception as e:
            write_log(f"Error loading YOLO: {e}")
            return

        while self.running:
            frame = None
            with self._lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process
                    self.frame_to_process = None

            if frame is not None:
                try:
                    results = self.model.track(frame, persist=True, verbose=False)
                    dets, current_ids = [], []

                    for r in results:
                        for box in r.boxes:
                            if box.id is None:
                                continue
                            obj_id = int(box.id[0].cpu().numpy())
                            current_ids.append(obj_id)

                            b    = box.xyxy[0].cpu().numpy().astype(int)
                            cls  = int(box.cls[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            name = self.model.names[cls]
                            x1,y1,x2,y2 = b

                            if obj_id not in self.kf_trackers:
                                self.kf_trackers[obj_id] = KalmanBoxTracker((x1,y1,x2,y2), name, conf)

                            tracker = self.kf_trackers[obj_id]
                            tracker.predict()
                            tracker.update((x1,y1,x2,y2))
                            vx, vy = tracker.get_velocity()

                            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                            max_y  = frame.shape[0]

                            # ✅ Teruskan cx ke estimate_speed (bug fix)
                            speed_kmh = self.speed_estimator.estimate_speed(
                                obj_id, (vx, vy), cx, cy, max_y)
                            speed_kmh = min(speed_kmh, SPEED_CAP_KMH)

                            direction = classify_direction(vx, vy)

                            # Virtual line crossing
                            self.line_counter.update(obj_id, cx, cy, name, speed_kmh, self.camera_name)

                            # DB insert (throttled)
                            now = time.time()
                            if now - self._last_db_write[obj_id] >= self._DB_INTERVAL:
                                db_insert_detection(self.camera_name, obj_id, name,
                                                    speed_kmh, cx, cy, direction)
                                self._last_db_write[obj_id] = now

                            dets.append((b, name, conf, speed_kmh, obj_id, vx, vy, direction))

                    # Cleanup stale trackers
                    for k in [k for k in self.kf_trackers if k not in current_ids]:
                        del self.kf_trackers[k]
                        self.speed_estimator.speed_smoothing.pop(k, None)
                        self.line_counter.remove(k)
                        self._last_db_write.pop(k, None)

                    with self._lock:
                        self.detections = dets

                except Exception as e:
                    write_log(f"Inference error: {e}")
            else:
                time.sleep(0.01)

    def update_frame(self, frame):
        with self._lock:
            self.frame_to_process = frame

    def get_detections(self):
        with self._lock:
            return list(self.detections)

    def get_line_summary(self):
        return self.line_counter.get_summary()

    def stop(self):
        self.running = False
        self.wait()


# ===============================================================
#  VIDEO THREAD
# ===============================================================
class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    stats_ready = pyqtSignal(dict)

    def __init__(self, streamer, fps, detection_thread=None):
        super().__init__()
        self.streamer        = streamer
        self.fps             = fps
        self.detection_thread = detection_thread
        self.frame_duration  = 1.0 / fps
        self.running = True
        self.paused  = False
        self.frame_count  = 0
        self.health_timer = time.time()
        self.next_frame_time = time.time()

        self.trajectory_mask  = None
        self.background_frame = None
        self.last_centroids   = {}

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            try:
                frame = self.streamer.read(timeout=10.0)
            except queue.Empty:
                self.next_frame_time = time.time()
                time.sleep(0.5)
                continue

            if self.detection_thread:
                self.detection_thread.update_frame(frame.copy())
                detections = self.detection_thread.get_detections()
                frame = frame.copy()

                h_frame, w_frame = frame.shape[:2]
                if self.trajectory_mask is None:
                    self.trajectory_mask  = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
                if self.background_frame is None:
                    self.background_frame = frame.copy()
                if not detections:
                    self.background_frame = frame.copy()

                # ── Gambar virtual counting line multi-arah ──
                for arm, cfg in COUNTING_LINES.items():
                    if cfg["type"] == "H":
                        cv2.line(frame, (cfg["x1"], cfg["y"]), (cfg["x2"], cfg["y"]), (0,255,255), 2)
                        cv2.putText(frame, arm, (cfg["x1"], cfg["y"]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    else:
                        cv2.line(frame, (cfg["x"], cfg["y1"]), (cfg["x"], cfg["y2"]), (0,255,255), 2)
                        cv2.putText(frame, arm, (cfg["x"]+5, cfg["y1"]+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                if detections:
                    current_ids = []
                    for det in detections:
                        b, name, conf, speed, obj_id, vx, vy, direction = det
                        x1,y1,x2,y2 = b
                        cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                        current_ids.append(obj_id)

                        # Trajectory
                        val = sum(ord(c) for c in name)
                        tc  = ((val*45)%255, (val*89+100)%255, (val*123+150)%255)
                        if obj_id in self.last_centroids:
                            cv2.line(self.trajectory_mask, self.last_centroids[obj_id], (cx,cy), tc, 2)
                        self.last_centroids[obj_id] = (cx, cy)

                        # Bounding box warna
                        color = (0,0,255) if speed > OVERSPEED_KMH else ((val*45)%255, (val*89)%255, (val*123)%255)
                        l = 15
                        for px,py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                            dx = l if px==x1 else -l
                            dy = l if py==y1 else -l
                            cv2.line(frame,(px,py),(px+dx,py),color,2)
                            cv2.line(frame,(px,py),(px,py+dy),color,2)

                        # Velocity arrow
                        cv2.arrowedLine(frame,(cx,cy),(int(cx+vx*5),int(cy+vy*5)),(255,0,255),2)
                        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

                        # Label
                        text = f"[{obj_id}] {name.upper()} {speed:.1f}km/h {direction}"
                        (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                        cv2.rectangle(frame,(x1,y1-22),(x1+tw+4,y1),color,-1)
                        cv2.putText(frame,text,(x1+2,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)

                    stale = [k for k in self.last_centroids if k not in current_ids]
                    for k in stale: del self.last_centroids[k]
                else:
                    self.last_centroids.clear()

            # Timing
            now  = time.time()
            wait = self.next_frame_time - now
            if wait > 0: time.sleep(wait)
            lag = time.time() - self.next_frame_time
            self.next_frame_time = (time.time() + self.frame_duration
                                    if lag > 1.0 else self.next_frame_time + self.frame_duration)

            self.frame_ready.emit(frame)
            self.frame_count += 1

            now = time.time()
            if now - self.health_timer >= 5.0:
                elapsed    = now - self.health_timer
                fps_actual = self.frame_count / elapsed
                q          = self.streamer.queue_size()
                self.stats_ready.emit({
                    'fps': fps_actual, 'buffer': q,
                    'delay': q/self.fps, 'target_fps': self.fps
                })
                self.frame_count  = 0
                self.health_timer = now

    def pause(self):
        self.paused = not self.paused
        return self.paused

    def stop(self):
        self.running = False
        self._save_trajectory()
        self.wait()

    def _save_trajectory(self):
        if self.trajectory_mask is None or self.background_frame is None:
            return
        result = cv2.add(self.background_frame, self.trajectory_mask)
        cam    = next((n for n,u in STREAM_URLS.items() if u==CURRENT_STREAM_URL), "Unknown")
        cv2.putText(result,"SIGAP — TRAJECTORY MAP",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.putText(result,f"Location: {cam}",(30,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(result,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),(30,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
        fname = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(fname, result)
        write_log(f"Trajectory saved: {fname}")


# ===============================================================
#  EDIT LINES DIALOG
# ===============================================================
from PyQt6.QtWidgets import QDialog, QFormLayout, QSpinBox, QDialogButtonBox

class LineEditorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Edit Koordinat Garis")
        self.setMinimumWidth(350)
        self.setStyleSheet("""
            QDialog {background:#0f172a; color:#e2e8f0; font-family: 'Segoe UI', Arial;}
            QLabel {color:#e2e8f0; font-weight:bold;}
            QFrame#LineCard {background:#1e293b; border-radius:6px; border:1px solid #334155; padding: 10px;}
            QSpinBox {background:#0f172a; color:#00ffcc; border:1px solid #334155; padding:4px;}
            QPushButton {background:#3b82f6; color:white; border-radius:4px; padding:6px 12px; font-weight:bold;}
            QPushButton:hover {background:#60a5fa;}
        """)
        layout = QVBoxLayout(self)
        self.spins = {}

        for arm, cfg in COUNTING_LINES.items():
            gb = QFrame(); gb.setObjectName("LineCard")
            gl = QFormLayout(gb)
            lbl = QLabel(f"LENGAN {arm.upper()} ({'Horizontal' if cfg['type']=='H' else 'Vertikal'})")
            lbl.setStyleSheet("color:#f59e0b;")
            layout.addWidget(lbl)
            
            arm_spins = {}
            keys = ["y", "x1", "x2"] if cfg["type"] == "H" else ["x", "y1", "y2"]
            for k in keys:
                sb = QSpinBox()
                sb.setRange(0, 3000)
                sb.setValue(cfg[k])
                gl.addRow(k.upper(), sb)
                arm_spins[k] = sb
            
            layout.addWidget(gb)
            self.spins[arm] = arm_spins

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.Save).setStyleSheet("background:#10b981;")
        layout.addWidget(btns)

    def get_new_config(self):
        new_cfg = {}
        for arm, cfg in COUNTING_LINES.items():
            c = {"type": cfg["type"]}
            for k, sb in self.spins[arm].items():
                c[k] = sb.value()
            new_cfg[arm] = c
        return new_cfg

# ===============================================================
#  MAIN WINDOW
# ===============================================================
class SIGAPWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚦 SIGAP — Sistem Intelijen Geospasial Analitik Pemantauan Lalu Lintas")
        self.setMinimumSize(1500, 900)
        self._setup_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(15)

        # ── LEFT: Video + Controls ──
        left = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        vf = QFrame(); vf.setObjectName("VideoFrame")
        vl = QVBoxLayout(vf); vl.setContentsMargins(0,0,0,0)
        vl.addWidget(self.video_label)
        left.addWidget(vf, stretch=1)

        # Controls bar
        ctrl = QFrame(); ctrl.setObjectName("ControlFrame")
        cl   = QHBoxLayout(ctrl); cl.setContentsMargins(15,10,15,10)

        self.status_labels = {}
        for item in ['FPS','Buffer','Delay','Status']:
            card = QFrame(); card.setObjectName("MetricCard")
            vb   = QVBoxLayout(card); vb.setSpacing(2); vb.setContentsMargins(12,8,12,8)
            t = QLabel(item); t.setStyleSheet("color:#00ffcc;font-size:10px;font-weight:bold;")
            v = QLabel("--"); v.setStyleSheet("color:#fff;font-size:16px;font-weight:bold;")
            v.setFont(QFont("Consolas",12))
            vb.addWidget(t); vb.addWidget(v)
            cl.addWidget(card)
            self.status_labels[item.lower()] = v

        self.stream_combo = QComboBox()
        self.stream_combo.setObjectName("StreamCombo")
        for n in STREAM_URLS: self.stream_combo.addItem(n)
        self.stream_combo.setFixedHeight(32)
        self.stream_combo.currentTextChanged.connect(self.change_stream)
        cl.addWidget(QLabel("📹")); cl.addWidget(self.stream_combo)
        cl.addStretch()

        self.btn_edit_lines = QPushButton("⚙️ Edit Garis"); self.btn_edit_lines.setObjectName("BtnNormal")
        self.btn_pause = QPushButton("⏸ Pause"); self.btn_pause.setObjectName("BtnWarning")
        self.btn_stop  = QPushButton("⏹ Stop");  self.btn_stop.setObjectName("BtnDanger")
        
        self.btn_edit_lines.setFixedSize(110,32)
        for b in [self.btn_pause, self.btn_stop]: b.setFixedSize(110,32)
        
        self.btn_edit_lines.clicked.connect(self.open_edit_lines)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_stream)
        
        cl.addWidget(self.btn_edit_lines); cl.addWidget(self.btn_pause); cl.addWidget(self.btn_stop)
        left.addWidget(ctrl)
        root.addLayout(left, stretch=7)

        # ── RIGHT: Analytics Panel ──
        right = QVBoxLayout(); right.setSpacing(10)

        hdr = QLabel("SIGAP ANALYTICS")
        hdr.setStyleSheet("color:#00ffcc;font-size:15px;font-weight:800;letter-spacing:2px;padding:8px;")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(hdr)

        # Metric cards row
        mrow = QHBoxLayout()
        self.lbl_total     = QLabel("0")
        self.lbl_overspeed = QLabel("0")
        self.lbl_crossings = QLabel("0")
        for title, lbl, color in [
            ("TERDETEKSI",  self.lbl_total,     "#3498db"),
            ("OVERSPEED",   self.lbl_overspeed,  "#ff4757"),
            ("LINE CROSS",  self.lbl_crossings,  "#2ecc71"),
        ]:
            card = QFrame(); card.setObjectName("MetricCard")
            vb   = QVBoxLayout(card)
            t = QLabel(title); t.setStyleSheet(f"color:{color};font-size:9px;font-weight:bold;")
            lbl.setStyleSheet("color:white;font-size:22px;font-weight:bold;")
            vb.addWidget(t); vb.addWidget(lbl)
            mrow.addWidget(card)
        right.addLayout(mrow)

        # Geospatial status
        gs = QFrame(); gs.setObjectName("MetricCard")
        gl = QVBoxLayout(gs)
        ghr = QHBoxLayout()
        ghr.addWidget(QLabel("⚙️ GEOSPATIAL"))
        self.lbl_calib = QLabel("✅ AKTIF")
        self.lbl_calib.setStyleSheet("color:#2ecc71;font-size:13px;font-weight:bold;")
        ghr.addStretch(); ghr.addWidget(self.lbl_calib)
        gl.addLayout(ghr)
        info = QLabel("Homografi + cx per-kendaraan\nppm_x & ppm_y dihitung dinamis")
        info.setStyleSheet("color:#718096;font-size:10px;")
        gl.addWidget(info)
        right.addWidget(gs)

        # Line counter summary
        lc_frame = QFrame(); lc_frame.setObjectName("MetricCard")
        lc_layout = QVBoxLayout(lc_frame)
        lc_title = QLabel("📊 KENDARAAN LEWAT GARIS")
        lc_title.setStyleSheet("color:#00ffcc;font-size:11px;font-weight:bold;")
        lc_layout.addWidget(lc_title)
        self.lbl_line_detail = QLabel("Motor: 0 | Mobil: 0 | Truk: 0")
        self.lbl_line_detail.setStyleSheet("color:#e2e8f0;font-size:11px;")
        self.lbl_line_detail.setWordWrap(True)
        lc_layout.addWidget(self.lbl_line_detail)
        right.addWidget(lc_frame)

        # Detection table
        self.log_table = QTableWidget(0, 5)
        self.log_table.setHorizontalHeaderLabels(["ID","Kelas","Spd","Arah","Status"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setObjectName("LogTable")
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        right.addWidget(self.log_table, stretch=1)

        root.addLayout(right, stretch=3)

        self.known_ids     = set()
        self.overspeed_cnt = 0
        self.status_bar    = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("SIGAP siap.")

        # Timer update line counter info setiap 2 detik
        self._line_timer = QTimer()
        self._line_timer.timeout.connect(self._update_line_counter_label)
        self._line_timer.start(2000)

        self.init_stream()

    # ── THEME ──────────────────────────────────────────────
    def _setup_theme(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#0a0e17;}
            QLabel{color:#e2e8f0;}
            QFrame#VideoFrame{background:#000;border:2px solid #1e293b;border-radius:8px;}
            QFrame#ControlFrame{background:#0f172a;border-radius:8px;border:1px solid #1e293b;}
            QFrame#MetricCard{background:#1e293b;border-radius:6px;border:1px solid #334155;}
            QPushButton{color:#fff;border:none;border-radius:5px;font-weight:bold;background:#3b82f6;}
            QPushButton:hover{background:#60a5fa;}
            QPushButton#BtnNormal{background:#3b82f6;}
            QPushButton#BtnNormal:hover{background:#60a5fa;}
            QPushButton#BtnWarning{background:#f59e0b;}
            QPushButton#BtnWarning:hover{background:#fbbf24;}
            QPushButton#BtnDanger{background:#ef4444;}
            QPushButton#BtnDanger:hover{background:#f87171;}
            QComboBox{background:#1e293b;color:#fff;border:1px solid #334155;border-radius:4px;padding:4px 8px;font-weight:bold;min-width:160px;}
            QComboBox QAbstractItemView{background:#0f172a;color:#fff;selection-background-color:#3b82f6;}
            QTableWidget#LogTable{background:#0f172a;alternate-background-color:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;gridline-color:#334155;font-size:11px;}
            QHeaderView::section{background:#1e293b;color:#94a3b8;font-weight:bold;border:none;border-bottom:2px solid #334155;padding:4px;}
            QStatusBar{background:#0a0e17;color:#94a3b8;border-top:1px solid #1e293b;}
        """)

    # ── STREAM INIT ────────────────────────────────────────
    def init_stream(self):
        if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
        write_log("=== SIGAP v2.0 START ===")
        load_lines_config()
        self.stream_fps = detect_stream_fps(CURRENT_STREAM_URL)

        cam_name = next((n for n,u in STREAM_URLS.items() if u==CURRENT_STREAM_URL), "Unknown")
        self.streamer = StableStreamer(CURRENT_STREAM_URL, WIDTH, HEIGHT, self.stream_fps).start()

        model_file = "/home/reynboo/YOLO26/ModelTest/best_traffic_model.pt"
        self.detection_thread = DetectionThread(model_file, cam_name)
        self.detection_thread.speed_estimator.fps = self.stream_fps
        self.detection_thread.start()

        self.status_bar.showMessage("Menghubungkan ke stream...")
        self.warmup_timer = QTimer()
        self.warmup_timer.timeout.connect(self._check_warmup)
        self.warmup_timer.start(200)
        self.target_fill = min(15, self.streamer.buffer_size)

    def _check_warmup(self):
        q = self.streamer.queue_size()
        self.status_labels['buffer'].setText(str(q))
        self.status_labels['delay'].setText(f"{q/self.stream_fps:.1f}s")
        self.status_labels['status'].setText("Warming up...")
        if q >= self.target_fill:
            self.warmup_timer.stop()
            self._start_video()

    def _start_video(self):
        self.video_thread = VideoThread(self.streamer, self.stream_fps, self.detection_thread)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.stats_ready.connect(self.update_stats)
        self.video_thread.start()
        self.status_labels['status'].setText("● LIVE")
        self.status_labels['status'].setStyleSheet("color:#10b981;font-size:16px;font-weight:bold;")
        self.status_bar.showMessage(f"Stream aktif — SIGAP v2.0 | DB: {DB_FILE}")

    # ── FRAME UPDATE ───────────────────────────────────────
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pix    = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pix)

        if not hasattr(self, 'detection_thread'): return
        dets = self.detection_thread.get_detections()
        visible = sorted(dets, key=lambda x: x[3], reverse=True)[:30]
        self.log_table.setRowCount(len(visible))

        for i, det in enumerate(visible):
            _, name, conf, speed, obj_id, vx, vy, direction = det
            if obj_id not in self.known_ids:
                self.known_ids.add(obj_id)
                self.lbl_total.setText(str(len(self.known_ids)))
                if speed > OVERSPEED_KMH:
                    self.overspeed_cnt += 1
                    self.lbl_overspeed.setText(str(self.overspeed_cnt))

            is_over = speed > OVERSPEED_KMH
            color   = QColor("#ff4757") if is_over else QColor("#e2e8f0")
            status  = "⚠️ OVERSPEED" if is_over else "✅ Normal"
            cols    = [f"#{obj_id}", name.upper(), f"{speed:.1f}", direction, status]
            for j, text in enumerate(cols):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(color)
                self.log_table.setItem(i, j, item)

    def _update_line_counter_label(self):
        if not hasattr(self, 'detection_thread'): return
        summary = self.detection_thread.get_line_summary()
        self.lbl_crossings.setText(str(summary.get('unique_total', 0)))
        
        detail_lines = []
        for arm in ["Utara", "Selatan", "Barat", "Timur"]:
            counts = summary.get('per_arm', {}).get(arm, {})
            m = counts.get('masuk', 0); k = counts.get('keluar', 0)
            if m > 0 or k > 0:
                detail_lines.append(f"{arm} → in:{m} out:{k}")
            
        detail = "\n".join(detail_lines) if detail_lines else "Belum ada"
        self.lbl_line_detail.setText(detail)

    def update_stats(self, stats):
        self.status_labels['fps'].setText(f"{stats['fps']:.1f}")
        self.status_labels['buffer'].setText(str(stats['buffer']))
        self.status_labels['delay'].setText(f"{stats['delay']:.1f}s")

    # ── CONTROLS ───────────────────────────────────────────
    def toggle_pause(self):
        if not hasattr(self, 'video_thread'): return
        paused = self.video_thread.pause()
        self.btn_pause.setText("▶ Resume" if paused else "⏸ Pause")
        self.status_labels['status'].setText("⏸ PAUSED" if paused else "● LIVE")
        self.status_labels['status'].setStyleSheet(
            "color:#f0883e;font-size:16px;font-weight:bold;" if paused
            else "color:#3fb950;font-size:16px;font-weight:bold;")

    def open_edit_lines(self):
        global COUNTING_LINES
        dlg = LineEditorDialog(self)
        if dlg.exec():
            COUNTING_LINES.update(dlg.get_new_config())
            save_lines_config()
            write_log("Koordinat garis counting berhasil di-update dan disimpan.")

    def stop_stream(self):
        if hasattr(self, 'video_thread'):   self.video_thread.stop()
        if hasattr(self, 'detection_thread'): self.detection_thread.stop()
        self.streamer.stop()
        self.status_labels['status'].setText("⏹ STOPPED")
        self.status_labels['status'].setStyleSheet("color:#da3633;font-size:16px;font-weight:bold;")
        self.btn_pause.setEnabled(False); self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stream dihentikan.")

    def change_stream(self, stream_name):
        global CURRENT_STREAM_URL
        url = STREAM_URLS.get(stream_name)
        if not url or url == CURRENT_STREAM_URL: return
        write_log(f"Pindah stream → {stream_name}")
        self.stop_stream()
        self.lbl_total.setText("0"); self.lbl_overspeed.setText("0")
        self.lbl_crossings.setText("0"); self.lbl_line_detail.setText("Belum ada")
        self.known_ids.clear(); self.overspeed_cnt = 0
        self.log_table.setRowCount(0)
        if hasattr(self,'video_thread'):     self.video_thread.wait()
        if hasattr(self,'detection_thread'): self.detection_thread.wait()
        # Reset KalmanBoxTracker counter
        KalmanBoxTracker.count = 0
        CURRENT_STREAM_URL = url
        self.btn_pause.setEnabled(True); self.btn_stop.setEnabled(True)
        self.init_stream()

    def closeEvent(self, event):
        self.stop_stream()
        DB_CONN.close()
        event.accept()


# ===============================================================
#  MAIN
# ===============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = SIGAPWindow()
    win.show()
    sys.exit(app.exec())