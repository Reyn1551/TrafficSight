import sys
import cv2
import numpy as np
import json
import webbrowser
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QMessageBox, QFrame, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

STREAM_URL = "http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8"

class ImageLabel(QLabel):
    pointsChanged = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.original_image = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(400, 300)
        self.dragging_idx = -1
        self.drag_threshold = 15 # radius in pixels to grab a point

    def set_cv_image(self, cv_img):
        self.original_image = cv_img.copy()
        h, w = cv_img.shape[:2]
        
        # Initialize default rectangle in the center if no points
        if not self.points:
            cx, cy = w // 2, h // 2
            qw, qh = w // 4, h // 4
            self.points = [
                (cx - qw, cy - qh), # Top-Left
                (cx + qw, cy - qh), # Top-Right
                (cx + qw, cy + qh), # Bottom-Right
                (cx - qw, cy + qh)  # Bottom-Left
            ]
        self.update_view()
        self.pointsChanged.emit(self.points)

    def _get_pixel_mapping(self, click_x, click_y):
        """Convert relative widget click coordinates to original image coordinates."""
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
        
        if 0 <= adj_x <= pixmap_w and 0 <= adj_y <= pixmap_h:
            orig_h, orig_w = self.original_image.shape[:2]
            real_x = int((adj_x / pixmap_w) * orig_w)
            real_y = int((adj_y / pixmap_h) * orig_h)
            return (real_x, real_y)
        return None

    def update_view(self):
        if self.original_image is None: return
        
        display_img = self.original_image.copy()
        
        # Draw translucent polygon overlay
        if len(self.points) == 4:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            
            # Create overlay for alpha blending
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
            
            # Draw boundary lines
            cv2.polylines(display_img, [pts], True, (255, 255, 0), 2)
            
        # Draw draggable points
        for i, pt in enumerate(self.points):
            color = (0, 0, 255) if i == self.dragging_idx else (0, 255, 0)
            radius = 8 if i == self.dragging_idx else 6
            cv2.circle(display_img, pt, radius, color, -1)
            cv2.circle(display_img, pt, radius, (255, 255, 255), 1)
            cv2.putText(display_img, str(i+1), (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Convert to QPixmap
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(display_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            real_pos = self._get_pixel_mapping(event.pos().x(), event.pos().y())
            if real_pos:
                rx, ry = real_pos
                # Find closest point
                min_dist = float('inf')
                closest_idx = -1
                
                for i, (px, py) in enumerate(self.points):
                    dist = np.sqrt((rx - px)**2 + (ry - py)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                        
                # Check against threshold but scaled for high-res images
                orig_h = self.original_image.shape[0] if self.original_image is not None else 1080
                scaled_threshold = self.drag_threshold * (orig_h / min(self.height(), 1)) 
                
                # Make grabbing easier
                if min_dist < max(scaled_threshold, 150): 
                    self.dragging_idx = closest_idx
                    self.update_view()

    def mouseMoveEvent(self, event):
        if self.dragging_idx != -1:
            real_pos = self._get_pixel_mapping(event.pos().x(), event.pos().y())
            if real_pos:
                rx, ry = real_pos
                # Ensure it stays within bounds
                h, w = self.original_image.shape[:2]
                rx = max(0, min(w - 1, rx))
                ry = max(0, min(h - 1, ry))
                
                self.points[self.dragging_idx] = (rx, ry)
                self.update_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging_idx != -1:
                self.dragging_idx = -1
                self.update_view()
                self.pointsChanged.emit(self.points)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_view()

    def clear(self):
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            cx, cy = w // 2, h // 2
            qw, qh = w // 4, h // 4
            self.points = [
                (cx - qw, cy - qh),
                (cx + qw, cy - qh),
                (cx + qw, cy + qh),
                (cx - qw, cy + qh)
            ]
        self.update_view()
        self.pointsChanged.emit(self.points)

class CalibrationDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 Dashboard Kalibrasi CCTV & Google Earth")
        self.resize(1100, 700)
        self.setStyleSheet("background-color: #f0f2f5;")
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # ================= KIRI - LAYAR CCTV =================
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: #000; border: 2px solid #bdc3c7; border-radius: 5px;")
        
        step1_title = QLabel("<b>Langkah 1: Sesuaikan Area Kalibrasi (Lengan Jalan)</b><br><i>Tarik (Drag) ke-4 titik merah untuk membentuk area persegi panjang searah ruas jalan.</i>")
        step1_title.setStyleSheet("font-size: 14px; color: #2c3e50;")
        
        left_layout.addWidget(step1_title)
        left_layout.addWidget(self.image_label, stretch=1)
        
        btn_reset = QPushButton("Reset Posisi Titik")
        btn_reset.clicked.connect(self.image_label.clear)
        btn_reset.setStyleSheet("padding: 10px; background-color: #e74c3c; color: white; font-weight: bold; font-size: 13px; border-radius: 4px;")
        left_layout.addWidget(btn_reset)
        
        main_layout.addWidget(left_panel, stretch=6)
        
        # ================= KANAN - KONTROL KALIBRASI =================
        right_panel = QFrame()
        right_panel.setMinimumWidth(400)
        right_panel.setStyleSheet("background-color: white; border-radius: 10px; padding: 15px; border: 1px solid #dcdde1;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        title = QLabel("Menu Kalibrasi Lanjutan")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; border: none;")
        right_layout.addWidget(title)
        right_layout.addSpacing(15)
        
        # --- LANGKAH 2: Buka Earth ---
        guide_group = QGroupBox("Langkah 2: Ukur di Google Earth")
        guide_group.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        guide_layout = QVBoxLayout(guide_group)
        
        guide_text = QLabel(
            "<b>PENTING! Jangan gunakan tool 'Luas Area/Polygon'.</b><br><br>"
            "Gunakan tool berlambang penggaris (Measure -> Jarak/Line):<br><br>"
            "1. Tarik garis searah sumbu X (kiri ke kanan jalan) = <b>Lebar</b>.<br>"
            "2. Tarik garis searah sumbu Y (atas ke bawah jalan) = <b>Panjang</b>."
        )
        guide_text.setWordWrap(True)
        guide_text.setStyleSheet("font-weight: normal; font-size: 13px; color: #555; border: none;")
        guide_layout.addWidget(guide_text)
        
        btn_earth = QPushButton("🌍 Buka Google Earth Web")
        btn_earth.setStyleSheet("padding: 12px; background-color: #3498db; color: white; font-weight: bold; font-size: 14px; border-radius: 4px;")
        btn_earth.clicked.connect(lambda: webbrowser.open("https://earth.google.com/web/"))
        guide_layout.addWidget(btn_earth)
        
        right_layout.addWidget(guide_group)
        right_layout.addSpacing(15)
        
        # --- LANGKAH 3: Input ---
        input_group = QGroupBox("Langkah 3: Masukkan Jarak Asli")
        input_group.setStyleSheet("font-weight: bold; font-size: 14px; color: #34495e;")
        form_layout = QFormLayout(input_group)
        
        self.input_width = QLineEdit()
        self.input_width.setPlaceholderText("Contoh: 6.5")
        self.input_width.setStyleSheet("padding: 8px; font-weight: normal; font-size: 13px; border: 1px solid #ccc; border-radius: 4px; background: white;")
        
        self.input_length = QLineEdit()
        self.input_length.setPlaceholderText("Contoh: 15.2")
        self.input_length.setStyleSheet("padding: 8px; font-weight: normal; font-size: 13px; border: 1px solid #ccc; border-radius: 4px; background: white;")
        
        lbl_w = QLabel("Lebar Jalan (Titik 1 ke 2) [Meter]:")
        lbl_h = QLabel("Panjang Jalan (Titik 1 ke 4) [Meter]:")
        lbl_w.setStyleSheet("font-weight: normal; border: none;")
        lbl_h.setStyleSheet("font-weight: normal; border: none;")
        
        form_layout.addRow(lbl_w, self.input_width)
        form_layout.addRow(lbl_h, self.input_length)
        
        right_layout.addWidget(input_group)
        right_layout.addSpacing(25)
        
        # --- Ekusi ---
        self.btn_calc = QPushButton("Hitung Kalibrasi (PPM) & Simpan")
        self.btn_calc.setStyleSheet("padding: 15px; background-color: #2ecc71; color: white; font-weight: bold; font-size: 15px; border-radius: 6px;")
        self.btn_calc.clicked.connect(self.calculate_homography)
        right_layout.addWidget(self.btn_calc)
        
        self.lbl_status = QLabel("Status: Menunggu 4 Titik...")
        self.lbl_status.setStyleSheet("color: #e67e22; font-weight: bold; font-size: 14px; margin-top: 10px; border: none;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.lbl_status)
        
        self.image_label.pointsChanged.connect(self.on_points_changed)
        
        main_layout.addWidget(right_panel, stretch=3)
        
        self.load_frame()

    def load_frame(self):
        self.lbl_status.setText("Status: Mengakses Server CCTV...")
        self.lbl_status.setStyleSheet("color: #3498db; font-weight: bold; border: none;")
        QApplication.processEvents()
        
        cap = cv2.VideoCapture(STREAM_URL)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.image_label.set_cv_image(frame)
            self.lbl_status.setText("Status: OK! Silakan klik 4 titik.")
            self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold; border: none;")
        else:
            QMessageBox.warning(self, "Koneksi Gagal", "Gagal mengambil koneksi m3u8. Mode Demo aktif.")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "CCTV KONEKSI TERPUTUS", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (100,100,100), 4)
            self.image_label.set_cv_image(frame)

    def on_points_changed(self, points):
        if len(points) == 4:
            self.lbl_status.setText("Status: 4 Titik Terkunci, Input Jarak.")
            self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold; border: none;")
        else:
            self.lbl_status.setText(f"Status: {len(points)}/4 Titik Dipilih.")
            self.lbl_status.setStyleSheet("color: #e67e22; font-weight: bold; border: none;")

    def calculate_homography(self):
        points = self.image_label.points
        if len(points) != 4:
            QMessageBox.warning(self, "Peringatan", "Anda belum memilih 4 titik koordinat di layar CCTV!")
            return
            
        try:
            val_w = self.input_width.text().replace(',', '.')
            val_h = self.input_length.text().replace(',', '.')
            real_w = float(val_w)
            real_h = float(val_h)
        except ValueError:
            QMessageBox.warning(self, "Peringatan", "Masukan angka jarak asli salah.\nPastikan anda mengisi dengan angka (contoh: 5.5).")
            return

        # Skala resolusi Bird's Eye (50 pixel = 1 meter)
        ppm_scale = 50 
        dst_w = int(real_w * ppm_scale)
        dst_h = int(real_h * ppm_scale)
        
        pts_src = np.array(points, dtype=np.float32)
        pts_dst = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        ppm_x = dst_w / real_w
        ppm_y = dst_h / real_h
        
        calib_data = {
            "calibrated": True,
            "points_cctv": points,
            "real_world_m": {"width": real_w, "length": real_h},
            "warped_resolution": {"width": dst_w, "height": dst_h},
            "transform_matrix": M.tolist(),
            "ppm_x": ppm_x,
            "ppm_y": ppm_y
        }
        
        # Simpan supaya bisa dipakai di inovation.py
        with open("geospatial_calibration.json", "w") as f:
            json.dump(calib_data, f, indent=4)
            
        # Potong & Tampilkan Preview Homography
        warped = cv2.warpPerspective(self.image_label.original_image, M, (dst_w, dst_h))
        
        QMessageBox.information(self, "Kalibrasi Berhasil!", 
            f"Setelan Kalibrasi Geospasial sudah disimpan.\n\n"
            f"Pixel-Per-Meter Sumbu X: {ppm_x:.2f} px/m\n"
            f"Pixel-Per-Meter Sumbu Y: {ppm_y:.2f} px/m\n\n"
            f"Jendela Pratinjau Bird's Eye View akan terbuka saat anda menekan OK.")
            
        cv2.imshow("Hasil Transformasi Bird's Eye View", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    app = QApplication(sys.argv)
    window = CalibrationDashboard()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
