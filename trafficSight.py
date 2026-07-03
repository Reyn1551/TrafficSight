import os
import sys
import time
import math
import queue
import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

from trafficsight.config import (
    STREAM_URLS, DEFAULT_STREAM_URL, WIDTH, HEIGHT,
    FALLBACK_FPS, OVERSPEED_KMH, SPEED_CAP_KMH,
)
from trafficsight.logger import write_log
from trafficsight.adapters.streamer import StableStreamer
from trafficsight.services.tracking import DetectionThread
from trafficsight.services.line_counter import VirtualLineCounter


def detect_stream_fps(url, timeout=15):
    write_log("Mendeteksi FPS asli stream...")
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
        '-of', 'csv=p=0',
        url,
    ]

    try:
        result = __import__('subprocess').run(cmd, capture_output=True, text=True, timeout=timeout)
        parts = result.stdout.replace('\n', ',').split(',')
        for part in parts:
            part = part.strip()
            if '/' in part:
                try:
                    numerator, denominator = part.split('/')
                    fps = float(numerator) / float(denominator)
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
    except Exception as exc:
        write_log(f"ffprobe error: {exc}")

    write_log(f"Pakai fallback FPS: {FALLBACK_FPS}")
    return FALLBACK_FPS


class VideoLabel(QLabel):
    def __init__(self, counting_lines, parent=None):
        super().__init__(parent)
        self.counting_lines = counting_lines
        self.active_arm = None
        self.active_attr = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def get_frame_coords(self, x_label, y_label):
        if not self.pixmap() or self.pixmap().isNull():
            return 0, 0

        lbl_w, lbl_h = self.width(), self.height()
        frame_w, frame_h = WIDTH, HEIGHT
        scale = min(lbl_w / frame_w, lbl_h / frame_h)

        pix_w = int(frame_w * scale)
        pix_h = int(frame_h * scale)
        pad_x = (lbl_w - pix_w) / 2
        pad_y = (lbl_h - pix_h) / 2

        x_frame = int((x_label - pad_x) / scale)
        y_frame = int((y_label - pad_y) / scale)
        return x_frame, y_frame

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x_f, y_f = self.get_frame_coords(event.pos().x(), event.pos().y())
        min_dist = 60
        self.active_arm = None
        self.active_attr = None

        for arm, cfg in self.counting_lines.items():
            pts = [
                ("start", cfg["x1"], cfg["y"]),
                ("end", cfg["x2"], cfg["y"]),
            ] if cfg["type"] == "H" else [
                ("start", cfg["x"], cfg["y1"]),
                ("end", cfg["x"], cfg["y2"]),
            ]
            for ptype, px, py in pts:
                dist = math.hypot(x_f - px, y_f - py)
                if dist < min_dist:
                    min_dist = dist
                    self.active_arm = arm
                    self.active_attr = ptype

    def mouseMoveEvent(self, event):
        if self.active_arm is None:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return

        x_f, y_f = self.get_frame_coords(event.pos().x(), event.pos().y())
        x_f = max(0, min(WIDTH, x_f))
        y_f = max(0, min(HEIGHT, y_f))
        cfg = self.counting_lines[self.active_arm]

        if cfg["type"] == "H":
            cfg["y"] = y_f
            if self.active_attr == "start":
                cfg["x1"] = x_f
            else:
                cfg["x2"] = x_f
        else:
            cfg["x"] = x_f
            if self.active_attr == "start":
                cfg["y1"] = y_f
            else:
                cfg["y2"] = y_f

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.active_arm = None


class VideoThread(__import__('PyQt6.QtCore').QtCore.QThread):
    frame_ready = __import__('PyQt6.QtCore').QtCore.pyqtSignal(np.ndarray)
    stats_ready = __import__('PyQt6.QtCore').QtCore.pyqtSignal(dict)

    def __init__(self, streamer, fps, detection_thread=None, counting_lines=None):
        super().__init__()
        self.streamer = streamer
        self.fps = fps
        self.detection_thread = detection_thread
        self.counting_lines = counting_lines or {}
        self.frame_duration = 1.0 / fps
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.health_timer = time.time()
        self.next_frame_time = time.time()
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
            else:
                detections = []

            frame = frame.copy()
            h_frame, w_frame = frame.shape[:2]

            if self.trajectory_mask is None:
                self.trajectory_mask = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
            if self.background_frame is None:
                self.background_frame = frame.copy()
            if not detections:
                self.background_frame = frame.copy()

            for arm, cfg in self.counting_lines.items():
                if cfg["type"] == "H":
                    cv2.line(frame, (cfg["x1"], cfg["y"]), (cfg["x2"], cfg["y"]), (0, 255, 255), 2)
                    cv2.putText(frame, arm, (cfg["x1"], cfg["y"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    cv2.line(frame, (cfg["x"], cfg["y1"]), (cfg["x"], cfg["y2"]), (0, 255, 255), 2)
                    cv2.putText(frame, arm, (cfg["x"] + 5, cfg["y1"] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if detections:
                current_ids = []
                for det in detections:
                    b, name, conf, speed, obj_id, vx, vy, direction = (
                        det.bbox, det.class_name, det.confidence, det.speed_kmh,
                        det.track_id, det.vx, det.vy, det.direction
                    )
                    x1, y1, x2, y2 = b
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    current_ids.append(obj_id)

                    val = sum(ord(c) for c in name)
                    color = (0, 0, 255) if speed > OVERSPEED_KMH else ((val * 45) % 255, (val * 89) % 255, (val * 123) % 255)
                    length = 15

                    cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
                    cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)
                    cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
                    cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)
                    cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2)
                    cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2)
                    cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2)
                    cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2)

                    if obj_id in self.last_centroids:
                        cv2.line(self.trajectory_mask, self.last_centroids[obj_id], (cx, cy), color, 2)
                    self.last_centroids[obj_id] = (cx, cy)

                    cv2.arrowedLine(frame, (cx, cy), (int(cx + vx * 5), int(cy + vy * 5)), (255, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    text = f"[{obj_id}] {name.upper()} {speed:.1f}km/h {direction}"
                    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(frame, (x1, y1 - 22), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, text, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

                stale_ids = [k for k in self.last_centroids if k not in current_ids]
                for k in stale_ids:
                    del self.last_centroids[k]
            else:
                self.last_centroids.clear()

            now = time.time()
            wait = self.next_frame_time - now
            if wait > 0:
                time.sleep(wait)
            lag = time.time() - self.next_frame_time
            self.next_frame_time = time.time() + self.frame_duration if lag > 1.0 else self.next_frame_time + self.frame_duration

            self.frame_ready.emit(frame)
            self.frame_count += 1

            if time.time() - self.health_timer >= 5.0:
                elapsed = time.time() - self.health_timer
                fps_actual = self.frame_count / elapsed if elapsed > 0 else 0.0
                self.stats_ready.emit({
                    'fps': fps_actual,
                    'buffer': self.streamer.queue_size(),
                    'delay': self.streamer.queue_size() / self.fps,
                    'target_fps': self.fps,
                })
                self.frame_count = 0
                self.health_timer = time.time()

    def pause(self):
        self.paused = not self.paused
        return self.paused

    def stop(self):
        self.running = False
        self.wait()
        self._save_trajectory()

    def _save_trajectory(self):
        if self.trajectory_mask is None or self.background_frame is None:
            return
        result = cv2.add(self.background_frame, self.trajectory_mask)
        cam_name = next((name for name, url in STREAM_URLS.items() if url == DEFAULT_STREAM_URL), "Unknown")
        cv2.putText(result, "TrafficSight — TRAJECTORY MAP", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result, f"Location: {cam_name}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(result, time.strftime("%Y-%m-%d %H:%M:%S"), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        filename = f"trajectory_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, result)
        write_log(f"Trajectory saved: {filename}")


class TrafficSightWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚦 TrafficSight — Sistem Analisis dan Pemantauan Lalu Lintas")
        self.setMinimumSize(1500, 900)
        self._setup_theme()
        self.current_stream_url = DEFAULT_STREAM_URL
        self.line_counter = VirtualLineCounter(self.current_stream_url)
        self.counting_lines = self.line_counter.counting_lines

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(15)

        left = QVBoxLayout()
        self.video_label = VideoLabel(counting_lines=self.counting_lines)
        self.video_label.setMinimumSize(960, 540)
        vf = QFrame()
        vf.setObjectName("VideoFrame")
        vl = QVBoxLayout(vf)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.video_label)
        left.addWidget(vf, stretch=1)

        ctrl = QFrame()
        ctrl.setObjectName("ControlFrame")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(15, 10, 15, 10)

        self.status_labels = {}
        for item in ['FPS', 'Buffer', 'Delay', 'Status']:
            card = QFrame(); card.setObjectName("MetricCard")
            vb = QVBoxLayout(card); vb.setSpacing(2); vb.setContentsMargins(12, 8, 12, 8)
            t = QLabel(item); t.setStyleSheet("color:#00ffcc;font-size:10px;font-weight:bold;")
            v = QLabel("--"); v.setStyleSheet("color:#fff;font-size:16px;font-weight:bold;")
            v.setFont(QFont("Consolas", 12))
            vb.addWidget(t); vb.addWidget(v)
            cl.addWidget(card)
            self.status_labels[item.lower()] = v

        self.stream_combo = QComboBox()
        self.stream_combo.setObjectName("StreamCombo")
        for name in STREAM_URLS:
            self.stream_combo.addItem(name)
        self.stream_combo.setFixedHeight(32)
        self.stream_combo.currentTextChanged.connect(self.change_stream)
        cl.addWidget(QLabel("📹")); cl.addWidget(self.stream_combo)
        cl.addStretch()

        self.btn_edit_lines = QPushButton("⚙️ Edit Garis")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_edit_lines.setFixedSize(110, 32)
        self.btn_pause.setFixedSize(110, 32)
        self.btn_stop.setFixedSize(110, 32)
        self.btn_edit_lines.clicked.connect(self.open_edit_lines)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_stream)

        cl.addWidget(self.btn_edit_lines); cl.addWidget(self.btn_pause); cl.addWidget(self.btn_stop)
        left.addWidget(ctrl)
        root.addLayout(left, stretch=7)

        right = QVBoxLayout(); right.setSpacing(10)
        hdr = QLabel("TrafficSight ANALYTICS")
        hdr.setStyleSheet("color:#00ffcc;font-size:15px;font-weight:800;letter-spacing:2px;padding:8px;")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(hdr)

        mrow = QHBoxLayout()
        self.lbl_total = QLabel("0")
        self.lbl_overspeed = QLabel("0")
        self.lbl_crossings = QLabel("0")
        for title, lbl, color in [
            ("TERDETEKSI", self.lbl_total, "#3498db"),
            ("OVERSPEED", self.lbl_overspeed, "#ff4757"),
            ("LINE CROSS", self.lbl_crossings, "#2ecc71"),
        ]:
            card = QFrame(); card.setObjectName("MetricCard")
            vb = QVBoxLayout(card)
            t = QLabel(title); t.setStyleSheet(f"color:{color};font-size:9px;font-weight:bold;")
            lbl.setStyleSheet("color:white;font-size:22px;font-weight:bold;")
            vb.addWidget(t); vb.addWidget(lbl)
            mrow.addWidget(card)
        right.addLayout(mrow)

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

        lc_frame = QFrame(); lc_frame.setObjectName("MetricCard")
        lc_layout = QVBoxLayout(lc_frame)
        lc_title = QLabel("📊 KENDARAAN LEWAT GARIS")
        lc_title.setStyleSheet("color:#00ffcc;font-size:11px;font-weight:bold;")
        lc_layout.addWidget(lc_title)
        self.lbl_line_detail = QLabel("Belum ada")
        self.lbl_line_detail.setStyleSheet("color:#e2e8f0;font-size:11px;")
        self.lbl_line_detail.setWordWrap(True)
        lc_layout.addWidget(self.lbl_line_detail)
        right.addWidget(lc_frame)

        self.log_table = QTableWidget(0, 5)
        self.log_table.setHorizontalHeaderLabels(["ID", "Kelas", "Spd", "Arah", "Status"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setObjectName("LogTable")
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        right.addWidget(self.log_table, stretch=1)

        root.addLayout(right, stretch=3)

        self.known_ids = set()
        self.overspeed_cnt = 0
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("TrafficSight siap.")

        self._line_timer = QTimer()
        self._line_timer.timeout.connect(self._update_line_counter_label)
        self._line_timer.start(2000)

        self.init_stream()

    def _setup_theme(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#0a0e17;}
            QLabel{color:#e2e8f0;}
            QFrame#VideoFrame{background:#000;border:2px solid #1e293b;border-radius:8px;}
            QFrame#ControlFrame{background:#0f172a;border-radius:8px;border:1px solid #1e293b;}
            QFrame#MetricCard{background:#1e293b;border-radius:6px;border:1px solid #334155;}
            QPushButton{color:#fff;border:none;border-radius:5px;font-weight:bold;background:#3b82f6;}
            QPushButton:hover{background:#60a5fa;}
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

    def init_stream(self):
        if os.path.exists("trafficSight_log.txt"):
            os.remove("trafficSight_log.txt")
        write_log("=== TrafficSight START ===")
        self.stream_fps = detect_stream_fps(self.current_stream_url)
        self.streamer = StableStreamer(self.current_stream_url, WIDTH, HEIGHT, self.stream_fps).start()

        cam_name = next((name for name, url in STREAM_URLS.items() if url == self.current_stream_url), "Unknown")
        self.detection_thread = DetectionThread(
            model_path="/home/reynboo/YOLO26/ModelTest/best_traffic_model.pt",
            camera_name=cam_name,
            line_counter=self.line_counter,
        )
        self.detection_thread.speed_estimator.fps = self.stream_fps
        self.detection_thread.start()

        self.status_bar.showMessage("Menghubungkan ke stream...")
        self.warmup_timer = QTimer()
        self.warmup_timer.timeout.connect(self._check_warmup)
        self.warmup_timer.start(200)
        self.target_fill = min(15, int(self.streamer.buffer_size))

    def _check_warmup(self):
        q = self.streamer.queue_size()
        self.status_labels['buffer'].setText(str(q))
        self.status_labels['delay'].setText(f"{q / self.stream_fps:.1f}s")
        self.status_labels['status'].setText("Warming up...")
        if q >= self.target_fill:
            self.warmup_timer.stop()
            self._start_video()

    def _start_video(self):
        self.video_thread = VideoThread(
            self.streamer,
            self.stream_fps,
            detection_thread=self.detection_thread,
            counting_lines=self.counting_lines,
        )
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.stats_ready.connect(self.update_stats)
        self.video_thread.start()
        self.status_labels['status'].setText("● LIVE")
        self.status_labels['status'].setStyleSheet("color:#10b981;font-size:16px;font-weight:bold;")
        self.status_bar.showMessage(f"Stream aktif — TrafficSight | PostgreSQL: {self.current_stream_url}")

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

        detections = self.detection_thread.get_detections()
        visible = sorted(detections, key=lambda d: d.speed_kmh, reverse=True)[:30]
        self.log_table.setRowCount(len(visible))

        for i, det in enumerate(visible):
            is_over = det.speed_kmh > OVERSPEED_KMH
            color = QColor("#ff4757") if is_over else QColor("#e2e8f0")
            status = "⚠️ OVERSPEED" if is_over else "✅ Normal"
            cols = [f"#{det.track_id}", det.class_name.upper(), f"{det.speed_kmh:.1f}", det.direction, status]
            for j, text in enumerate(cols):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(color)
                self.log_table.setItem(i, j, item)

            if det.track_id not in self.known_ids:
                self.known_ids.add(det.track_id)
                self.lbl_total.setText(str(len(self.known_ids)))
                if is_over:
                    self.overspeed_cnt += 1
                    self.lbl_overspeed.setText(str(self.overspeed_cnt))

    def _update_line_counter_label(self):
        summary = self.line_counter.get_summary()
        self.lbl_crossings.setText(str(summary.get('unique_total', 0)))
        detail_lines = []
        for arm in ["Utara", "Selatan", "Barat", "Timur"]:
            counts = summary.get('per_arm', {}).get(arm, {})
            m = counts.get('masuk', 0); k = counts.get('keluar', 0)
            if m > 0 or k > 0:
                detail_lines.append(f"{arm} → in:{m} out:{k}")
        self.lbl_line_detail.setText("\n".join(detail_lines) if detail_lines else "Belum ada")

    def update_stats(self, stats):
        self.status_labels['fps'].setText(f"{stats['fps']:.1f}")
        self.status_labels['buffer'].setText(str(stats['buffer']))
        self.status_labels['delay'].setText(f"{stats['delay']:.1f}s")

    def toggle_pause(self):
        if not hasattr(self, 'video_thread'):
            return
        paused = self.video_thread.pause()
        self.btn_pause.setText("▶ Resume" if paused else "⏸ Pause")
        self.status_labels['status'].setText("⏸ PAUSED" if paused else "● LIVE")
        self.status_labels['status'].setStyleSheet(
            "color:#f0883e;font-size:16px;font-weight:bold;" if paused else
            "color:#3fb950;font-size:16px;font-weight:bold;"
        )

    def open_edit_lines(self):
        self.video_label.setCursor(Qt.CursorShape.OpenHandCursor)
        self.btn_edit_lines.setText("💾 Simpan Garis")
        self.btn_edit_lines.setStyleSheet("background:#10b981;")
        self.status_bar.showMessage("EDIT MODE: Tarik titik merah di video untuk mengubah garis.")
        self.video_label.setCursor(Qt.CursorShape.PointingHandCursor)

    def stop_stream(self):
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()
        self.streamer.stop()
        self.status_labels['status'].setText("⏹ STOPPED")
        self.status_labels['status'].setStyleSheet("color:#da3633;font-size:16px;font-weight:bold;")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stream dihentikan.")
        self.line_counter.save_config()

    def change_stream(self, stream_name):
        url = STREAM_URLS.get(stream_name)
        if not url or url == self.current_stream_url:
            return
        write_log(f"Pindah stream → {stream_name}")
        self.stop_stream()
        if hasattr(self, 'video_thread'):
            self.video_thread.wait()
        if hasattr(self, 'detection_thread'):
            self.detection_thread.wait()

        self.current_stream_url = url
        self.line_counter = VirtualLineCounter(self.current_stream_url)
        self.counting_lines = self.line_counter.counting_lines
        self.video_label.counting_lines = self.counting_lines
        self.video_label.setCursor(Qt.CursorShape.CrossCursor)

        self.known_ids.clear()
        self.overspeed_cnt = 0
        self.lbl_total.setText("0")
        self.lbl_overspeed.setText("0")
        self.lbl_crossings.setText("0")
        self.lbl_line_detail.setText("Belum ada")
        self.log_table.setRowCount(0)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.init_stream()

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = TrafficSightWindow()
    win.show()
    sys.exit(app.exec())
