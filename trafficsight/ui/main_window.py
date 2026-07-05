"""Top-level PyQt6 dashboard.

Composes the video label, control bar (FPS / Buffer / Delay / Status
metric cards + camera selector + action buttons), and the right-hand
analytics panel (detections / overspeed / crossings / per-arm detail /
log table). Wires streamer → detection thread → video thread together
and exposes lifecycle methods (pause, stop, switch stream).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QPushButton, QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from ..adapters.streamer import StableStreamer
from ..config import cfg
from ..logger import write_log
from ..services.line_counter import VirtualLineCounter
from ..services.tracking import DetectionThread
from .fps_detector import detect_stream_fps
from .theme import THEME_QSS, STATUS_LIVE, STATUS_PAUSED, STATUS_STOPPED
from .video_label import VideoLabel
from .video_thread import VideoThread


class TrafficSightWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TrafficSight — Sistem Analisis dan Pemantauan Lalu Lintas")
        self.setMinimumSize(1500, 900)
        self.setStyleSheet(THEME_QSS)

        # Runtime state
        self.current_stream_url: str = cfg.default_stream_url
        self.line_counter = VirtualLineCounter(self.current_stream_url)
        self.counting_lines = self.line_counter.counting_lines
        self.known_ids: set[int] = set()
        self.overspeed_cnt = 0

        self.streamer: Optional[StableStreamer] = None
        self.detection_thread: Optional[DetectionThread] = None
        self.video_thread: Optional[VideoThread] = None
        self.stream_fps: float = cfg.fallback_fps

        # Build UI
        self._build_layout()
        self._build_status_bar()
        self._start_line_summary_timer()

        # Kick off the stream
        self.init_stream()

    # ------------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(15)

        root.addLayout(self._build_left_panel(), stretch=7)
        root.addLayout(self._build_right_panel(), stretch=3)

    def _build_left_panel(self) -> QVBoxLayout:
        left = QVBoxLayout()

        # Video frame
        self.video_label = VideoLabel(counting_lines=self.counting_lines)
        self.video_label.setMinimumSize(960, 540)
        vf = QFrame()
        vf.setObjectName("VideoFrame")
        vl = QVBoxLayout(vf)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.video_label)
        left.addWidget(vf, stretch=1)

        # Control bar
        left.addWidget(self._build_control_bar())
        return left

    def _build_control_bar(self) -> QFrame:
        ctrl = QFrame()
        ctrl.setObjectName("ControlFrame")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(15, 10, 15, 10)

        # Metric cards
        self.status_labels: dict[str, QLabel] = {}
        for item in ("FPS", "Buffer", "Delay", "Status"):
            card = QFrame()
            card.setObjectName("MetricCard")
            vb = QVBoxLayout(card)
            vb.setSpacing(2)
            vb.setContentsMargins(12, 8, 12, 8)
            title = QLabel(item)
            title.setStyleSheet("color:#00ffcc;font-size:10px;font-weight:bold;")
            value = QLabel("--")
            value.setStyleSheet("color:#fff;font-size:16px;font-weight:bold;")
            value.setFont(QFont("Consolas", 12))
            vb.addWidget(title)
            vb.addWidget(value)
            cl.addWidget(card)
            self.status_labels[item.lower()] = value

        # Camera selector
        self.stream_combo = QComboBox()
        self.stream_combo.setObjectName("StreamCombo")
        for name in cfg.stream_urls:
            self.stream_combo.addItem(name)
        self.stream_combo.setFixedHeight(32)
        self.stream_combo.currentTextChanged.connect(self.change_stream)
        cl.addWidget(QLabel("📹"))
        cl.addWidget(self.stream_combo)
        cl.addStretch()

        # Action buttons
        self.btn_edit_lines = QPushButton("⚙️ Edit Garis")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_stop = QPushButton("⏹ Stop")
        for btn in (self.btn_edit_lines, self.btn_pause, self.btn_stop):
            btn.setFixedSize(110, 32)
        self.btn_edit_lines.clicked.connect(self.open_edit_lines)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_stream)
        cl.addWidget(self.btn_edit_lines)
        cl.addWidget(self.btn_pause)
        cl.addWidget(self.btn_stop)

        return ctrl

    def _build_right_panel(self) -> QVBoxLayout:
        right = QVBoxLayout()
        right.setSpacing(10)

        hdr = QLabel("TrafficSight ANALYTICS")
        hdr.setStyleSheet(
            "color:#00ffcc;font-size:15px;font-weight:800;letter-spacing:2px;padding:8px;"
        )
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(hdr)

        right.addLayout(self._build_metric_row())
        right.addWidget(self._build_geospatial_card())
        right.addWidget(self._build_line_detail_card())
        right.addWidget(self._build_log_table(), stretch=1)
        return right

    def _build_metric_row(self) -> QHBoxLayout:
        mrow = QHBoxLayout()
        self.lbl_total = QLabel("0")
        self.lbl_overspeed = QLabel("0")
        self.lbl_crossings = QLabel("0")
        for title, lbl, color in [
            ("TERDETEKSI", self.lbl_total, "#3498db"),
            ("OVERSPEED",  self.lbl_overspeed, "#ff4757"),
            ("LINE CROSS", self.lbl_crossings, "#2ecc71"),
        ]:
            card = QFrame()
            card.setObjectName("MetricCard")
            vb = QVBoxLayout(card)
            t = QLabel(title)
            t.setStyleSheet(f"color:{color};font-size:9px;font-weight:bold;")
            lbl.setStyleSheet("color:white;font-size:22px;font-weight:bold;")
            vb.addWidget(t)
            vb.addWidget(lbl)
            mrow.addWidget(card)
        return mrow

    def _build_geospatial_card(self) -> QFrame:
        gs = QFrame()
        gs.setObjectName("MetricCard")
        gl = QVBoxLayout(gs)
        ghr = QHBoxLayout()
        ghr.addWidget(QLabel("⚙️ GEOSPATIAL"))
        self.lbl_calib = QLabel("✅ AKTIF")
        self.lbl_calib.setStyleSheet("color:#2ecc71;font-size:13px;font-weight:bold;")
        ghr.addStretch()
        ghr.addWidget(self.lbl_calib)
        gl.addLayout(ghr)
        info = QLabel(
            "Homografi + cx per-kendaraan\nppm_x & ppm_y dihitung dinamis"
        )
        info.setStyleSheet("color:#718096;font-size:10px;")
        gl.addWidget(info)
        return gs

    def _build_line_detail_card(self) -> QFrame:
        lc_frame = QFrame()
        lc_frame.setObjectName("MetricCard")
        lc_layout = QVBoxLayout(lc_frame)
        title = QLabel("📊 KENDARAAN LEWAT GARIS")
        title.setStyleSheet("color:#00ffcc;font-size:11px;font-weight:bold;")
        lc_layout.addWidget(title)
        self.lbl_line_detail = QLabel("Belum ada")
        self.lbl_line_detail.setStyleSheet("color:#e2e8f0;font-size:11px;")
        self.lbl_line_detail.setWordWrap(True)
        lc_layout.addWidget(self.lbl_line_detail)
        return lc_frame

    def _build_log_table(self) -> QTableWidget:
        self.log_table = QTableWidget(0, 5)
        self.log_table.setHorizontalHeaderLabels(["ID", "Kelas", "Spd", "Arah", "Status"])
        self.log_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setObjectName("LogTable")
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        return self.log_table

    def _build_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("TrafficSight siap.")

    def _start_line_summary_timer(self) -> None:
        self._line_timer = QTimer()
        self._line_timer.timeout.connect(self._update_line_counter_label)
        self._line_timer.start(2000)

    # ------------------------------------------------------------------ stream
    def init_stream(self) -> None:
        if os.path.exists("trafficSight_log.txt"):
            os.remove("trafficSight_log.txt")
        write_log("=== TrafficSight START ===")

        self.stream_fps = detect_stream_fps(self.current_stream_url)
        self.streamer = StableStreamer(
            self.current_stream_url, cfg.width, cfg.height,
            self.stream_fps, buffer_seconds=cfg.buffer_seconds,
        ).start()

        cam_name = cfg.camera_name_for(self.current_stream_url)
        self.detection_thread = DetectionThread(
            model_path=str(cfg.model_path),
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

    def _check_warmup(self) -> None:
        assert self.streamer is not None
        q = self.streamer.queue_size()
        self.status_labels["buffer"].setText(str(q))
        self.status_labels["delay"].setText(f"{q / self.stream_fps:.1f}s")
        self.status_labels["status"].setText("Warming up...")
        if q >= self.target_fill:
            self.warmup_timer.stop()
            self._start_video()

    def _start_video(self) -> None:
        assert self.streamer is not None and self.detection_thread is not None
        self.video_thread = VideoThread(
            self.streamer,
            self.stream_fps,
            detection_thread=self.detection_thread,
            counting_lines=self.counting_lines,
        )
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.stats_ready.connect(self.update_stats)
        self.video_thread.start()

        self.status_labels["status"].setText("● LIVE")
        self.status_labels["status"].setStyleSheet(STATUS_LIVE)
        self.status_bar.showMessage(
            f"Stream aktif — TrafficSight | PostgreSQL: {self.current_stream_url}"
        )

    # ------------------------------------------------------------------ slots
    def update_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

        assert self.detection_thread is not None
        detections = self.detection_thread.get_detections()
        visible = sorted(detections, key=lambda d: d.speed_kmh, reverse=True)[:30]
        self.log_table.setRowCount(len(visible))

        for i, det in enumerate(visible):
            is_over = det.speed_kmh > cfg.overspeed_kmh
            color = QColor("#ff4757") if is_over else QColor("#e2e8f0")
            status = "⚠️ OVERSPEED" if is_over else "✅ Normal"
            cols = [
                f"#{det.track_id}",
                det.class_name.upper(),
                f"{det.speed_kmh:.1f}",
                det.direction,
                status,
            ]
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

    def _update_line_counter_label(self) -> None:
        summary = self.line_counter.get_summary()
        self.lbl_crossings.setText(str(summary.get("unique_total", 0)))
        detail_lines: list[str] = []
        for arm in ("Utara", "Selatan", "Barat", "Timur"):
            counts = summary.get("per_arm", {}).get(arm, {})
            masuk = counts.get("masuk", 0)
            keluar = counts.get("keluar", 0)
            if masuk > 0 or keluar > 0:
                detail_lines.append(f"{arm} → in:{masuk} out:{keluar}")
        self.lbl_line_detail.setText(
            "\n".join(detail_lines) if detail_lines else "Belum ada"
        )

    def update_stats(self, stats: dict) -> None:
        self.status_labels["fps"].setText(f"{stats['fps']:.1f}")
        self.status_labels["buffer"].setText(str(stats["buffer"]))
        self.status_labels["delay"].setText(f"{stats['delay']:.1f}s")

    # ------------------------------------------------------------------ actions
    def toggle_pause(self) -> None:
        if self.video_thread is None:
            return
        paused = self.video_thread.pause()
        self.btn_pause.setText("▶ Resume" if paused else "⏸ Pause")
        self.status_labels["status"].setText("⏸ PAUSED" if paused else "● LIVE")
        self.status_labels["status"].setStyleSheet(
            STATUS_PAUSED if paused else STATUS_LIVE
        )

    def open_edit_lines(self) -> None:
        self.video_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_lines.setText("💾 Simpan Garis")
        self.btn_edit_lines.setStyleSheet("background:#10b981;")
        self.status_bar.showMessage(
            "EDIT MODE: Tarik titik merah di video untuk mengubah garis."
        )

    def stop_stream(self) -> None:
        if self.video_thread is not None:
            self.video_thread.stop()
        if self.detection_thread is not None:
            self.detection_thread.stop()
        if self.streamer is not None:
            self.streamer.stop()

        self.status_labels["status"].setText("⏹ STOPPED")
        self.status_labels["status"].setStyleSheet(STATUS_STOPPED)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stream dihentikan.")
        self.line_counter.save_config()

    def change_stream(self, stream_name: str) -> None:
        url = cfg.stream_urls.get(stream_name)
        if not url or url == self.current_stream_url:
            return
        write_log(f"Pindah stream → {stream_name}")
        self.stop_stream()
        if self.video_thread is not None:
            self.video_thread.wait()
        if self.detection_thread is not None:
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

    # ------------------------------------------------------------------ close
    def closeEvent(self, event) -> None:
        self.stop_stream()
        event.accept()
