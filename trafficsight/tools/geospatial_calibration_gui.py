"""Interactive 4-point homography calibration GUI.

Workflow:
1. Grab one frame from the chosen stream.
2. Drag the 4 red corners so the rectangle aligns with a flat road.
3. Enter the real-world width (1→2) and length (1→4) in meters.
4. Click "Hitung Kalibrasi (PPM) & Simpan".

The script writes ``geospatial_calibration.json`` next to the package
data dir (override with ``TRAFFICSIGHT_GEOSPATIAL_CALIB_FILE``).
"""

from __future__ import annotations

import json
import sys
import webbrowser

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QFrame, QVBoxLayout, QWidget,
)

from ..config import cfg

_PPM_SCALE = 50  # 50 px per meter in the warped bird's-eye preview


class ImageLabel(QLabel):
    """QLabel with 4 draggable corners overlaid on a video frame."""

    pointsChanged = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points: list[tuple[int, int]] = []
        self.original_image: np.ndarray | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(400, 300)
        self.dragging_idx = -1
        self.drag_threshold = 15

    # ----- image load -------------------------------------------------------
    def set_cv_image(self, cv_img: np.ndarray) -> None:
        self.original_image = cv_img.copy()
        if not self.points:
            self._reset_points_to_center()
        self.update_view()
        self.pointsChanged.emit(self.points)

    def _reset_points_to_center(self) -> None:
        assert self.original_image is not None
        h, w = self.original_image.shape[:2]
        cx, cy = w // 2, h // 2
        qw, qh = w // 4, h // 4
        self.points = [
            (cx - qw, cy - qh),  # top-left
            (cx + qw, cy - qh),  # top-right
            (cx + qw, cy + qh),  # bottom-right
            (cx - qw, cy + qh),  # bottom-left
        ]

    # ----- coordinate mapping ----------------------------------------------
    def _get_pixel_mapping(self, click_x: int, click_y: int):
        if not self.pixmap() or self.original_image is None:
            return None
        label_w = self.width()
        label_h = self.height()
        pixmap_w = self.pixmap().width()
        pixmap_h = self.pixmap().height()
        x_offset = (label_w - pixmap_w) // 2
        y_offset = (label_h - pixmap_h) // 2
        adj_x = click_x - x_offset
        adj_y = click_y - y_offset
        if not (0 <= adj_x <= pixmap_w and 0 <= adj_y <= pixmap_h):
            return None
        orig_h, orig_w = self.original_image.shape[:2]
        return (
            int((adj_x / pixmap_w) * orig_w),
            int((adj_y / pixmap_h) * orig_h),
        )

    # ----- drawing ----------------------------------------------------------
    def update_view(self) -> None:
        if self.original_image is None:
            return
        display_img = self.original_image.copy()

        if len(self.points) == 4:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
            cv2.polylines(display_img, [pts], True, (255, 255, 0), 2)

        for i, pt in enumerate(self.points):
            color = (0, 0, 255) if i == self.dragging_idx else (0, 255, 0)
            radius = 8 if i == self.dragging_idx else 6
            cv2.circle(display_img, pt, radius, color, -1)
            cv2.circle(display_img, pt, radius, (255, 255, 255), 1)
            cv2.putText(
                display_img, str(i + 1), (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(display_img_rgb.data, w, h, bytes_per_line,
                      QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    # ----- mouse ------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        real_pos = self._get_pixel_mapping(event.pos().x(), event.pos().y())
        if real_pos is None:
            return
        rx, ry = real_pos
        min_dist = float("inf")
        closest_idx = -1
        for i, (px, py) in enumerate(self.points):
            dist = float(np.sqrt((rx - px) ** 2 + (ry - py) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        orig_h = self.original_image.shape[0] if self.original_image is not None else 1080
        scaled_threshold = self.drag_threshold * (orig_h / min(self.height(), 1))
        if min_dist < max(scaled_threshold, 150):
            self.dragging_idx = closest_idx
            self.update_view()

    def mouseMoveEvent(self, event) -> None:
        if self.dragging_idx == -1:
            return
        real_pos = self._get_pixel_mapping(event.pos().x(), event.pos().y())
        if real_pos is None:
            return
        assert self.original_image is not None
        rx, ry = real_pos
        h, w = self.original_image.shape[:2]
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))
        self.points[self.dragging_idx] = (rx, ry)
        self.update_view()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.dragging_idx != -1:
            self.dragging_idx = -1
            self.update_view()
            self.pointsChanged.emit(self.points)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_view()

    def clear(self) -> None:
        if self.original_image is not None:
            self._reset_points_to_center()
        self.update_view()
        self.pointsChanged.emit(self.points)


class CalibrationDashboard(QMainWindow):
    """Top-level window: image label + measurement inputs + save button."""

    def __init__(self, stream_url: str | None = None) -> None:
        super().__init__()
        self.stream_url = stream_url or cfg.default_stream_url
        self.setWindowTitle("TrafficSight — Kalibrasi Geospasial")
        self.resize(1100, 700)
        self.setStyleSheet("background-color: #f0f2f5;")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        main_layout.addWidget(self._build_left_panel(), stretch=6)
        main_layout.addWidget(self._build_right_panel(), stretch=3)

        self._load_frame()

    # ----- panels -----------------------------------------------------------
    def _build_left_panel(self) -> QFrame:
        left = QFrame()
        layout = QVBoxLayout(left)
        self.image_label = ImageLabel()
        self.image_label.setStyleSheet(
            "background-color: #000; border: 2px solid #bdc3c7; border-radius: 5px;"
        )

        step1 = QLabel(
            "<b>Langkah 1: Sesuaikan Area Kalibrasi (Lengan Jalan)</b><br>"
            "<i>Tarik (Drag) ke-4 titik merah untuk membentuk area persegi "
            "panjang searah ruas jalan.</i>"
        )
        step1.setStyleSheet("font-size: 14px; color: #2c3e50;")
        layout.addWidget(step1)
        layout.addWidget(self.image_label, stretch=1)

        btn_reset = QPushButton("Reset Posisi Titik")
        btn_reset.clicked.connect(self.image_label.clear)
        btn_reset.setStyleSheet(
            "padding: 10px; background-color: #e74c3c; color: white; "
            "font-weight: bold; font-size: 13px; border-radius: 4px;"
        )
        layout.addWidget(btn_reset)
        return left

    def _build_right_panel(self) -> QFrame:
        right = QFrame()
        right.setMinimumWidth(400)
        right.setStyleSheet(
            "background-color: white; border-radius: 10px; padding: 15px; "
            "border: 1px solid #dcdde1;"
        )
        layout = QVBoxLayout(right)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Menu Kalibrasi Lanjutan")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        layout.addSpacing(15)
        layout.addWidget(self._build_guide_group())
        layout.addSpacing(15)
        layout.addWidget(self._build_input_group())
        layout.addSpacing(25)

        self.btn_calc = QPushButton("Hitung Kalibrasi (PPM) & Simpan")
        self.btn_calc.setStyleSheet(
            "padding: 15px; background-color: #2ecc71; color: white; "
            "font-weight: bold; font-size: 15px; border-radius: 6px;"
        )
        self.btn_calc.clicked.connect(self.calculate_homography)
        layout.addWidget(self.btn_calc)

        self.lbl_status = QLabel("Status: Menunggu 4 Titik...")
        self.lbl_status.setStyleSheet(
            "color: #e67e22; font-weight: bold; font-size: 14px; "
            "margin-top: 10px;"
        )
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.image_label.pointsChanged.connect(self.on_points_changed)
        return right

    def _build_guide_group(self) -> QGroupBox:
        group = QGroupBox("Langkah 2: Ukur di Google Earth")
        group.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        layout = QVBoxLayout(group)
        guide = QLabel(
            "<b>PENTING! Jangan gunakan tool 'Luas Area/Polygon'.</b><br><br>"
            "Gunakan tool berlambang penggaris (Measure -> Jarak/Line):<br><br>"
            "1. Tarik garis searah sumbu X (kiri ke kanan jalan) = <b>Lebar</b>.<br>"
            "2. Tarik garis searah sumbu Y (atas ke bawah jalan) = <b>Panjang</b>."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet("font-weight: normal; font-size: 13px; color: #555;")
        layout.addWidget(guide)

        btn_earth = QPushButton("🌍 Buka Google Earth Web")
        btn_earth.setStyleSheet(
            "padding: 12px; background-color: #3498db; color: white; "
            "font-weight: bold; font-size: 14px; border-radius: 4px;"
        )
        btn_earth.clicked.connect(lambda: webbrowser.open("https://earth.google.com/web/"))
        layout.addWidget(btn_earth)
        return group

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Langkah 3: Masukkan Jarak Asli")
        group.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        form = QFormLayout(group)

        self.input_width = QLineEdit()
        self.input_width.setPlaceholderText("Contoh: 6.5")
        self.input_length = QLineEdit()
        self.input_length.setPlaceholderText("Contoh: 15.2")
        for w in (self.input_width, self.input_length):
            w.setStyleSheet(
                "padding: 8px; font-weight: normal; font-size: 13px; "
                "border: 1px solid #ccc; border-radius: 4px; background: white;"
            )

        lbl_w = QLabel("Lebar Jalan (Titik 1 ke 2) [Meter]:")
        lbl_h = QLabel("Panjang Jalan (Titik 1 ke 4) [Meter]:")
        for lbl in (lbl_w, lbl_h):
            lbl.setStyleSheet("font-weight: normal;")

        form.addRow(lbl_w, self.input_width)
        form.addRow(lbl_h, self.input_length)
        return group

    # ----- frame load -------------------------------------------------------
    def _load_frame(self) -> None:
        self.lbl_status.setText("Status: Mengakses Server CCTV...")
        self.lbl_status.setStyleSheet("color: #3498db; font-weight: bold;")
        QApplication.processEvents()

        cap = cv2.VideoCapture(self.stream_url)
        ret, frame = cap.read()
        cap.release()

        if ret:
            self.image_label.set_cv_image(frame)
            self.lbl_status.setText("Status: OK! Silakan klik 4 titik.")
            self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            return

        QMessageBox.warning(self, "Koneksi Gagal", "Gagal mengambil koneksi m3u8. Mode Demo aktif.")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            frame, "CCTV KONEKSI TERPUTUS", (300, 360),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 4,
        )
        self.image_label.set_cv_image(frame)

    # ----- callbacks --------------------------------------------------------
    def on_points_changed(self, points: list) -> None:
        if len(points) == 4:
            self.lbl_status.setText("Status: 4 Titik Terkunci, Input Jarak.")
            self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self.lbl_status.setText(f"Status: {len(points)}/4 Titik Dipilih.")
            self.lbl_status.setStyleSheet("color: #e67e22; font-weight: bold;")

    def calculate_homography(self) -> None:
        points = self.image_label.points
        if len(points) != 4:
            QMessageBox.warning(self, "Peringatan",
                                "Anda belum memilih 4 titik koordinat di layar CCTV!")
            return

        try:
            real_w = float(self.input_width.text().replace(",", "."))
            real_h = float(self.input_length.text().replace(",", "."))
        except ValueError:
            QMessageBox.warning(self, "Peringatan",
                                "Masukan angka jarak asli salah.\n"
                                "Pastikan anda mengisi dengan angka (contoh: 5.5).")
            return

        dst_w = int(real_w * _PPM_SCALE)
        dst_h = int(real_h * _PPM_SCALE)
        pts_src = np.array(points, dtype=np.float32)
        pts_dst = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        ppm_x = dst_w / real_w
        ppm_y = dst_h / real_h

        calib_data = {
            "calibrated": True,
            "points_cctv": points,
            "real_world_m": {"width": real_w, "length": real_h},
            "warped_resolution": {"width": dst_w, "height": dst_h},
            "transform_matrix": matrix.tolist(),
            "ppm_x": ppm_x,
            "ppm_y": ppm_y,
        }

        cfg.geospatial_calib_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.geospatial_calib_file, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=4)

        warped = cv2.warpPerspective(
            self.image_label.original_image, matrix, (dst_w, dst_h)
        )
        QMessageBox.information(
            self, "Kalibrasi Berhasil!",
            f"Setelan Kalibrasi Geospasial sudah disimpan.\n\n"
            f"Pixel-Per-Meter Sumbu X: {ppm_x:.2f} px/m\n"
            f"Pixel-Per-Meter Sumbu Y: {ppm_y:.2f} px/m\n\n"
            f"Jendela Pratinjau Bird's Eye View akan terbuka saat anda menekan OK.",
        )
        cv2.imshow("Hasil Transformasi Bird's Eye View", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> int:
    app = QApplication(sys.argv)
    window = CalibrationDashboard()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
