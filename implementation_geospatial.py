import subprocess
import numpy as np
import cv2
import threading
import queue
import time
import os
import sys
import json
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStatusBar, 
                            QFrame, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QComboBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor
import torch
import torchvision.transforms as transforms
import math
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter

# ================= KONFIGURASI =================
STREAM_URLS = {
    "Sugeng Jeroni 2": "http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8",
    "Simpang Wirosaban Barat": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/playlist.m3u8",
    "Wirobrajan": "https://cctvjss.jogjakota.go.id/atcs/ATCS_wirobrajan.stream/playlist.m3u8"
}
CURRENT_STREAM_URL = STREAM_URLS["Sugeng Jeroni 2"]

WIDTH = 1920
HEIGHT = 1080
LOG_FILE = "stable_stream_log.txt"
BUFFER_SECONDS = 60
FALLBACK_FPS = 25.0

# ================= LOGGING =================
def write_log(text):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {text}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ================= DETECT FPS =================
def detect_stream_fps(url, timeout=15):
    """Deteksi FPS asli dengan parsing robust."""
    write_log("Mendeteksi FPS asli stream...")
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
        '-of', 'csv=p=0',
        url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
        write_log(f"ffprobe raw output: '{output}'")

        parts = output.replace('\n', ',').split(',')
        for part in parts:
            part = part.strip()
            if '/' in part:
                try:
                    num, den = part.split('/')
                    fps = float(num) / float(den)
                    if 1.0 < fps < 120.0:
                        write_log(f"FPS terdeteksi: {fps:.3f}")
                        return fps
                except Exception:
                    continue
            elif part:
                try:
                    fps = float(part)
                    if 1.0 < fps < 120.0:
                        write_log(f"FPS terdeteksi (plain): {fps:.3f}")
                        return fps
                except Exception:
                    continue

    except Exception as e:
        write_log(f"ffprobe error: {e}")

    write_log(f"Pakai fallback FPS: {FALLBACK_FPS}")
    return FALLBACK_FPS

# ================= STREAMER =================
class StableStreamer:
    def __init__(self, src, width, height, fps):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = int(BUFFER_SECONDS * fps)
        write_log(f"Buffer: {self.buffer_size} frames ({BUFFER_SECONDS}s @ {fps:.2f} FPS)")

        self.q = queue.Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.proc = None
        self._lock = threading.Lock()

        self.cmd = [
            'ffmpeg',
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '10',
            '-i', self.src,
            '-vsync', 'passthrough',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-loglevel', 'error',
            'pipe:1'
        ]

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        frame_size = self.width * self.height * 3
        while not self.stopped:
            raw = self.proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                write_log("Koneksi putus, mencoba reconnect...")
                with self._lock:
                    self.proc.kill()
                time.sleep(2)
                with self._lock:
                    self.proc = subprocess.Popen(
                        self.cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10**8
                    )
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

            try:
                self.q.put(frame, timeout=5.0)
            except queue.Full:
                write_log("Queue penuh timeout — frame di-skip")

    def read(self, timeout=10.0):
        return self.q.get(timeout=timeout)

    def queue_size(self):
        return self.q.qsize()

    def stop(self):
        self.stopped = True
        with self._lock:
            if self.proc:
                self.proc.kill()

# ================= KALMAN FILTER LAYER =================
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
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.0
        self.kf.R *= 1.0
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
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    def update(self, bbox: tuple):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        measurement = np.array([cx, cy, w, h]).reshape(4, 1)
        self.kf.update(measurement)
    
    def get_velocity(self):
        vx, vy = self.kf.x[4, 0], self.kf.x[5, 0]
        return float(vx), float(vy)

class SpeedEstimator:
    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.speed_smoothing = defaultdict(lambda: deque(maxlen=10))
        self.calibration_data = None
        self.load_calibration()
        
    def load_calibration(self):
        calib_file = "geospatial_calibration.json"
        if os.path.exists(calib_file):
            try:
                with open(calib_file, "r") as f:
                    self.calibration_data = json.load(f)
                write_log(f"Berhasil memuat data kalibrasi geospasial: {calib_file}")
            except Exception as e:
                write_log(f"Gagal memuat {calib_file}: {e}")
        else:
            write_log("Peringatan: File kalibrasi geospasial tidak ditemukan. Menggunakan estimasi default empiris.")

    def estimate_speed(self, track_id: int, velocity: tuple, cy: int, max_y: int) -> float:
        vx, vy = velocity
        
        # JIKA KALIBRASI TERSEDIA: Gunakan Matrix Transformasi Homography Perspektif Asli
        if self.calibration_data and self.calibration_data.get("calibrated"):
            # Ambil Transformation Matrix (M) & PPM
            M_list = self.calibration_data.get("transform_matrix")
            M = np.array(M_list, dtype=np.float32)
            ppm_y = self.calibration_data.get("ppm_y", 50.0) # meter per pixel pada sumbu Y Bird's Eye
            ppm_x = self.calibration_data.get("ppm_x", 50.0)
            
            # Koordinat pusat saat ini (P1)
            # Karena vx, vy adalah vektor per frame, P2 adalah pusat frame berikutnya
            # Kita bebas memilih koordinat dasar simulasi di resolusi 1920x1080
            p1 = np.array([[[1920/2, cy]]], dtype=np.float32)
            p2 = np.array([[[1920/2 + vx, cy + vy]]], dtype=np.float32)
            
            # Transformasikan P1 dan P2 ke koordinat Bird's Eye View
            p1_warp = cv2.perspectiveTransform(p1, M)
            p2_warp = cv2.perspectiveTransform(p2, M)
            
            # Jarak di Bird's Eye View (dalam px)
            dx_warp = p2_warp[0][0][0] - p1_warp[0][0][0]
            dy_warp = p2_warp[0][0][1] - p1_warp[0][0][1]
            
            # Konversi ke Meter nyata
            dist_x_m = dx_warp / ppm_x
            dist_y_m = dy_warp / ppm_y
            
            # Teorema Pythagoras untuk jarak murni dalam meter
            speed_ms = math.sqrt(dist_x_m**2 + dist_y_m**2) * self.fps
            speed_kmh = speed_ms * 3.6
            
        else:
            # JIKA TIDAK ADA KALIBRASI: Fallback ke kode estimasi maya versi lama (inovation.py)
            horizon_y = 200     
            t = max(0.0, min(1.0, (cy - horizon_y) / (max_y - horizon_y)))
            pixels_per_meter = 1.0 / (0.4 * (1 - t) + 0.02 * t) 
            
            speed_pixels = math.sqrt(vx**2 + vy**2)
            speed_ms = speed_pixels * self.fps / pixels_per_meter
            speed_kmh = speed_ms * 3.6
        
        self.speed_smoothing[track_id].append(speed_kmh)
        return np.mean(list(self.speed_smoothing[track_id]))

# ================= DETECT THREAD =================
class DetectionThread(QThread):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.running = True
        self.frame_to_process = None
        self.lock = threading.Lock()
        self.detections = []
        
        # State tracking via Kalman Filter Manager 
        self.kf_trackers = {}
        self.speed_estimator = SpeedEstimator(fps=FALLBACK_FPS)
        
    def run(self):
        try:
            from ultralytics import YOLO
            import urllib.request
            
            # Load YOLO
            write_log(f"Loading YOLO model {self.model_path}...")
            self.model = YOLO(self.model_path)
            write_log("YOLO model loaded successfully. (MiDaS depth removed for performance)")
            
        except Exception as e:
            write_log(f"Error loading YOLO model: {e}")
            return
            
        while self.running:
            frame = None
            with self.lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process
                    self.frame_to_process = None
            
            if frame is not None:
                try:
                    current_time = time.time()
                    
                    # predict instead of track because we skip frames to keep true FPS
                    # NOW SHIFTED TO TRACK to assign persistent IDs
                    results = self.model.track(frame, persist=True, verbose=False)
                    
                    # MiDaS Depth inference removed to save resources and improve tracking FPS
                    
                    dets = []
                    current_frame_ids = []
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            if box.id is None:
                                continue # ensure tracked id exists
                                
                            obj_id = int(box.id[0].cpu().numpy())
                            current_frame_ids.append(obj_id)
                            
                            b = box.xyxy[0].cpu().numpy().astype(int)
                            cls = int(box.cls[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            name = self.model.names[cls]
                            
                            x1, y1, x2, y2 = b
                            bbox_tuple = (x1, y1, x2, y2)
                            
                            if obj_id not in self.kf_trackers:
                                self.kf_trackers[obj_id] = KalmanBoxTracker(bbox_tuple, name, conf)
                                
                            tracker = self.kf_trackers[obj_id]
                            tracker.predict()
                            tracker.update(bbox_tuple)
                            vx, vy = tracker.get_velocity()
                            
                            # Estimate velocity and combine with perspective
                            cy = int((y1 + y2) / 2)
                            cx = int((x1 + x2) / 2)
                            max_y = frame.shape[0]
                            speed_kmh = self.speed_estimator.estimate_speed(obj_id, (vx, vy), cy, max_y)
                            
                            # Cap unrealistic speeds
                            if speed_kmh > 140.0:
                                speed_kmh = 140.0
                                
                            dets.append((b, name, conf, speed_kmh, obj_id, vx, vy))
                    
                    # Cleanup old tracks that are gone from frame to prevent memory leak
                    stale_ids = [k for k in self.kf_trackers.keys() if k not in current_frame_ids]
                    for k in stale_ids:
                        del self.kf_trackers[k]
                        if k in self.speed_estimator.speed_smoothing:
                            del self.speed_estimator.speed_smoothing[k]
                    
                    with self.lock:
                        self.detections = dets
                except Exception as e:
                    write_log(f"Inference error: {e}")
            else:
                time.sleep(0.01)

    def update_frame(self, frame):
        with self.lock:
            self.frame_to_process = frame

    def get_detections(self):
        with self.lock:
            return self.detections

    def stop(self):
        self.running = False
        self.wait()

# ================= VIDEO THREAD =================
class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    stats_ready = pyqtSignal(dict)
    
    def __init__(self, streamer, fps, detection_thread=None):
        super().__init__()
        self.streamer = streamer
        self.fps = fps
        self.detection_thread = detection_thread
        self.frame_duration = 1.0 / fps
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.health_timer = time.time()
        self.next_frame_time = time.time()
        
        # Trajectory Background Capture
        self.trajectory_mask = None
        self.background_frame = None
        self.last_centroids = {}
        
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
                
                # Ensure frame is editable before drawing anything!
                frame = frame.copy()
                
                h_frame, w_frame = frame.shape[:2]
                
                if self.trajectory_mask is None:
                    self.trajectory_mask = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
                if self.background_frame is None:
                    self.background_frame = frame.copy()
                    
                if not detections:
                    # Capture clean background when no vehicles are present
                    self.background_frame = frame.copy()
                
                if detections:
                    # Arrow drawing variables
                    arrow_color = (255, 0, 255) # Magenta for movement vector
                    current_ids = []
                    
                    for b, name, conf, speed, obj_id, vx, vy in detections:
                        x1, y1, x2, y2 = b
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        current_ids.append(obj_id)
                        
                        # Add presence to internal trajectory mask
                        val = sum(ord(c) for c in name)
                        traj_color = ((val * 45) % 255, (val * 89 + 100) % 255, (val * 123 + 150) % 255)
                        if obj_id in self.last_centroids:
                            cv2.line(self.trajectory_mask, self.last_centroids[obj_id], (cx, cy), traj_color, 2)
                        self.last_centroids[obj_id] = (cx, cy)
                        
                        # Core color based on class name
                        val = sum(ord(c) for c in name)
                        color = ((val * 45) % 255, (val * 89) % 255, (val * 123) % 255)
                        
                        # Highlight speeders in RED
                        if speed > 60.0:
                            color = (0, 0, 255) # BGR Red
                            
                        # Draw high-tech bounding box corners (Crosshairs) instead of full rect
                        length = 15
                        cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
                        cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)
                        cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
                        cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)
                        cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2)
                        cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2)
                        cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2)
                        cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2)
                        
                        # Draw velocity arrow (Kalman Prediction)
                        end_point = (int(cx + vx * 5), int(cy + vy * 5))
                        cv2.arrowedLine(frame, (cx, cy), end_point, arrow_color, 2)
                        
                        # Draw center
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        
                        # Add ID and Speed to text
                        text = f"[{obj_id}] {name.upper()} | {speed:.1f} km/h"
                        (txt_w, txt_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(frame, (x1, y1 - 22), (x1 + txt_w + 4, y1), color, -1)
                        
                        # Draw text in white
                        cv2.putText(frame, text, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    stale = [k for k in self.last_centroids.keys() if k not in current_ids]
                    for k in stale:
                        del self.last_centroids[k]
                else:
                    self.last_centroids.clear()

            # Timing control
            now = time.time()
            wait = self.next_frame_time - now
            if wait > 0:
                time.sleep(wait)

            lag = time.time() - self.next_frame_time
            if lag > 1.0:
                self.next_frame_time = time.time() + self.frame_duration
            else:
                self.next_frame_time += self.frame_duration

            self.frame_ready.emit(frame)
            self.frame_count += 1

            # Health stats every 5 seconds
            now = time.time()
            if now - self.health_timer >= 5.0:
                elapsed = now - self.health_timer
                fps_actual = self.frame_count / elapsed
                q = self.streamer.queue_size()
                delay_s = q / self.fps
                
                stats = {
                    'fps': fps_actual,
                    'buffer': q,
                    'delay': delay_s,
                    'target_fps': self.fps
                }
                self.stats_ready.emit(stats)
                
                self.frame_count = 0
                self.health_timer = now

    def pause(self):
        self.paused = not self.paused
        return self.paused
        
    def stop(self):
        self.running = False
        self.save_trajectory_map()
        self.wait()
        
    def save_trajectory_map(self):
        if self.trajectory_mask is None or self.background_frame is None:
            return
            
        # Combine background and trajectory lines
        result_img = cv2.add(self.background_frame, self.trajectory_mask)
        
        stream_name = "Unknown Camera"
        for name, url in STREAM_URLS.items():
            if url == CURRENT_STREAM_URL:
                stream_name = name
                break
                
        # Draw Output Labels
        cv2.putText(result_img, f"ATCS TRAFFIC TRAJECTORY MAP", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_img, f"Location: {stream_name}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result_img, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_output_{timestamp}.png"
        cv2.imwrite(filename, result_img)
        write_log(f"Trajectory Map saved to {filename}")

# ================= MAIN WINDOW =================
class ModernCCTVWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 ATCS Intelligence Command Center")
        self.setMinimumSize(1400, 900)
        
        # Setup modern dark theme
        self.setup_theme()
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        
        # === LEFT PANEL (Video & Controls) ===
        left_panel = QVBoxLayout()
        
        # Video display
        self.video_frame = QFrame()
        self.video_frame.setObjectName("VideoFrame")
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        video_layout.addWidget(self.video_label)
        left_panel.addWidget(self.video_frame, stretch=1)
        
        # Control panel
        control_frame = QFrame()
        control_frame.setObjectName("ControlFrame")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(20, 15, 20, 15)
        
        # Status indicators
        self.status_labels = {}
        status_items = ['FPS', 'Buffer', 'Delay', 'Status']
        
        for item in status_items:
            container = QFrame()
            container.setObjectName("MetricCard")
            vbox = QVBoxLayout(container)
            vbox.setSpacing(2)
            vbox.setContentsMargins(15, 10, 15, 10)
            
            title = QLabel(item)
            title.setStyleSheet("color: #00ffcc; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
            value = QLabel("--")
            value.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
            value.setFont(QFont("Consolas", 14))
            
            vbox.addWidget(title)
            vbox.addWidget(value)
            control_layout.addWidget(container)
            self.status_labels[item.lower()] = value
            
        # Stream Selection Dropdown
        self.stream_combo = QComboBox()
        self.stream_combo.setObjectName("StreamCombo")
        for name in STREAM_URLS.keys():
            self.stream_combo.addItem(name)
        self.stream_combo.setFixedHeight(35)
        self.stream_combo.currentTextChanged.connect(self.change_stream)
        
        control_layout.addWidget(QLabel("📹 Camera:"))
        control_layout.addWidget(self.stream_combo)
        
        control_layout.addStretch()
        
        # Control buttons
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setObjectName("BtnWarning")
        self.btn_pause.setFixedSize(120, 35)
        self.btn_pause.clicked.connect(self.toggle_pause)
        
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("BtnDanger")
        self.btn_stop.setFixedSize(120, 35)
        self.btn_stop.clicked.connect(self.stop_stream)
        
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_stop)
        left_panel.addWidget(control_frame)
        
        main_layout.addLayout(left_panel, stretch=7)
        
        # === RIGHT PANEL (Live Logs & Analytics) ===
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        # Analytics Header
        analytics_header = QLabel("LIVE TRAFFIC ANALYTICS")
        analytics_header.setStyleSheet("color: #00ffcc; font-size: 16px; font-weight: 800; letter-spacing: 2px; padding: 10px;")
        analytics_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(analytics_header)
        
        # Top right stats
        stats_layout = QHBoxLayout()
        self.total_vehicles_lbl = QLabel("0")
        self.speeding_lbl = QLabel("0")
        
        for title, lbl, color in [("TOTAL TRACKED", self.total_vehicles_lbl, "#3498db"), ("SPEEDING (>60)", self.speeding_lbl, "#ff4757")]:
            card = QFrame()
            card.setObjectName("MetricCard")
            vbox = QVBoxLayout(card)
            t = QLabel(title)
            t.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
            lbl.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
            vbox.addWidget(t)
            vbox.addWidget(lbl)
            stats_layout.addWidget(card)
            
        right_panel.addLayout(stats_layout)
        
        # Speed Calibration Panel
        calib_frame = QFrame()
        calib_frame.setObjectName("MetricCard")
        calib_layout = QVBoxLayout(calib_frame)
        
        calib_header_layout = QHBoxLayout()
        calib_title = QLabel("⚙️ STATUS GEOSPATIAL")
        calib_title.setStyleSheet("color: #00ffcc; font-size: 12px; font-weight: bold;")
        
        self.calib_value_lbl = QLabel("PURE")
        self.calib_value_lbl.setStyleSheet("color: #2ecc71; font-size: 14px; font-weight: bold;")
        
        calib_header_layout.addWidget(calib_title)
        calib_header_layout.addStretch()
        calib_header_layout.addWidget(self.calib_value_lbl)
        
        info_calib = QLabel("Kecepatan dihitung menggunakan matriks\nhomografi geospasial aktual.")
        info_calib.setStyleSheet("color: #95a5a6; font-size: 11px;")
        
        calib_layout.addLayout(calib_header_layout)
        calib_layout.addWidget(info_calib)
        right_panel.addWidget(calib_frame)
        
        # Table Log
        self.log_table = QTableWidget(0, 4)
        self.log_table.setHorizontalHeaderLabels(["ID", "Class", "Spd (km/h)", "Warn"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setObjectName("LogTable")
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        right_panel.addWidget(self.log_table, stretch=1)
        
        main_layout.addLayout(right_panel, stretch=3)
        
        # Tracking states for UI
        self.known_ids = set()
        self.speeding_count = 0
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System initialized...")
        
        # Initialize stream
        self.init_stream()
        
    def setup_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0e17;
            }
            QLabel {
                color: #e2e8f0;
            }
            QFrame#VideoFrame {
                background-color: #000000;
                border: 2px solid #1e293b;
                border-radius: 10px;
            }
            QFrame#ControlFrame {
                background-color: #0f172a;
                border-radius: 10px;
                border: 1px solid #1e293b;
            }
            QFrame#MetricCard {
                background-color: #1e293b;
                border-radius: 8px;
                border: 1px solid #334155;
            }
            QPushButton {
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                background-color: #3b82f6;
            }
            QPushButton:hover { background-color: #60a5fa; }
            QPushButton#BtnWarning { background-color: #f59e0b; }
            QPushButton#BtnWarning:hover { background-color: #fbbf24; }
            QPushButton#BtnDanger { background-color: #ef4444; }
            QPushButton#BtnDanger:hover { background-color: #f87171; }
            
            QComboBox {
                background-color: #1e293b;
                color: white;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
                min-width: 180px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: white;
                selection-background-color: #3b82f6;
            }
            
            QTableWidget#LogTable {
                background-color: #0f172a;
                alternate-background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                gridline-color: #334155;
                font-size: 12px;
            }
            QTableWidget#LogTable::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                font-weight: bold;
                border: none;
                border-bottom: 2px solid #334155;
                padding: 5px;
            }
            QStatusBar {
                background-color: #0a0e17;
                color: #94a3b8;
                border-top: 1px solid #1e293b;
            }
        """)
        
    def init_stream(self):
        # Cleanup old log
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        write_log("=== STABLE STREAM v2 - Modern GUI ===")
        
        # Detect FPS
        self.stream_fps = detect_stream_fps(CURRENT_STREAM_URL)
        
        # Start streamer
        self.streamer = StableStreamer(CURRENT_STREAM_URL, WIDTH, HEIGHT, self.stream_fps).start()
        
        # Initialize YOLO Detection Thread
        model_file = "/home/reynboo/YOLO26/ModelTest/best_traffic_model.pt"
        self.detection_thread = DetectionThread(model_file)
        self.detection_thread.start()
        
        # Warm up dialog
        self.status_bar.showMessage(f"Starting video stream...")
        self.warmup_timer = QTimer()
        self.warmup_timer.timeout.connect(self.check_warmup)
        self.warmup_timer.start(200)
        
        # Mulai video begitu sudah ada minimal 15 frame (kurang dari 1 detik)
        self.target_fill = min(15, self.streamer.buffer_size)
        
    def check_warmup(self):
        q = self.streamer.queue_size()
        self.status_labels['buffer'].setText(f"{q}")
        self.status_labels['delay'].setText(f"{q/self.stream_fps:.1f}s")
        self.status_labels['status'].setText("Warming up...")
        
        if q >= self.target_fill:
            self.warmup_timer.stop()
            self.start_video_thread()
    
    def start_video_thread(self):
        self.video_thread = VideoThread(self.streamer, self.stream_fps, self.detection_thread)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.stats_ready.connect(self.update_stats)
        self.video_thread.start()
        
        self.status_labels['status'].setText("● LIVE")
        self.status_labels['status'].setStyleSheet("color: #10b981; font-size: 18px; font-weight: bold;")
        self.status_bar.showMessage("Stream started successfully [Cyberpunk Dashboard Active]")
        
    def update_frame(self, frame):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Convert to QPixmap
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update Table Log Data from detections if available
        if hasattr(self, 'detection_thread'):
            dets = self.detection_thread.get_detections()
            # Only process up to 30 visible items to prevent lag
            visible_dets = sorted(dets, key=lambda x: x[3], reverse=True)[:30] # Sort by speed descending
            
            self.log_table.setRowCount(len(visible_dets))
            for i, data in enumerate(visible_dets):
                _, name, conf, speed, obj_id, vx, vy = data
                
                # Update Aggregate Stats
                if obj_id not in self.known_ids:
                    self.known_ids.add(obj_id)
                    self.total_vehicles_lbl.setText(str(len(self.known_ids)))
                    
                    if speed > 60.0:
                        self.speeding_count += 1
                        self.speeding_lbl.setText(str(self.speeding_count))
                
                # Populate Table rows
                id_item = QTableWidgetItem(f"#{obj_id}")
                class_item = QTableWidgetItem(name.upper())
                speed_item = QTableWidgetItem(f"{speed:.1f}")
                
                # Speed threshold logic
                warn = ""
                if speed > 60.0:
                    warn = "⚠️ OVERSPEED"
                    for item in [id_item, class_item, speed_item]:
                        item.setForeground(QColor("#ff4757"))
                
                warn_item = QTableWidgetItem(warn)
                if warn:
                    warn_item.setForeground(QColor("#ff4757"))
                    
                # Centers alignment
                for item in [id_item, class_item, speed_item, warn_item]:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                self.log_table.setItem(i, 0, id_item)
                self.log_table.setItem(i, 1, class_item)
                self.log_table.setItem(i, 2, speed_item)
                self.log_table.setItem(i, 3, warn_item)
        
    def update_stats(self, stats):
        self.status_labels['fps'].setText(f"{stats['fps']:.1f}")
        self.status_labels['buffer'].setText(f"{stats['buffer']}")
        self.status_labels['delay'].setText(f"{stats['delay']:.1f}s")
        
    def toggle_pause(self):
        if hasattr(self, 'video_thread'):
            paused = self.video_thread.pause()
            self.btn_pause.setText("▶ Resume" if paused else "⏸ Pause")
            self.status_labels['status'].setText("⏸ PAUSED" if paused else "● LIVE")
            self.status_labels['status'].setStyleSheet(
                "color: #f0883e; font-size: 16px; font-weight: bold;" if paused 
                else "color: #3fb950; font-size: 16px; font-weight: bold;"
            )
            
    def stop_stream(self):
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()
        self.streamer.stop()
        self.status_labels['status'].setText("⏹ STOPPED")
        self.status_labels['status'].setStyleSheet("color: #da3633; font-size: 16px; font-weight: bold;")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stream stopped")
        
    def change_stream(self, stream_name):
        global CURRENT_STREAM_URL
        url = STREAM_URLS.get(stream_name)
        if url and url != CURRENT_STREAM_URL:
            write_log(f"Switching stream to {stream_name}...")
            self.stop_stream()
            
            # Reset UI States
            self.total_vehicles_lbl.setText("0")
            self.speeding_lbl.setText("0")
            self.known_ids.clear()
            self.speeding_count = 0
            self.log_table.setRowCount(0)
            
            # Wait for threads to actual stop
            if hasattr(self, 'video_thread'):
                self.video_thread.wait()
            if hasattr(self, 'detection_thread'):
                self.detection_thread.wait()
                
            CURRENT_STREAM_URL = url
            self.init_stream()
        
    def closeEvent(self, event):
        self.stop_stream()
        event.accept()

# ================= MAIN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ModernCCTVWindow()
    window.show()
    
    sys.exit(app.exec())
