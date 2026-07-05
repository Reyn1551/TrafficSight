"""``QLabel`` subclass that displays video and lets the user drag counting lines.

Mouse coordinates from the widget are mapped back to native frame
coordinates (1920×1080 by default) using the letterbox transform applied
by Qt's ``KeepAspectRatio`` scaling. Dragging a line endpoint updates
the shared ``counting_lines`` dict in place.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from ..config import cfg


class VideoLabel(QLabel):
    """Video display + draggable counting-line endpoints."""

    def __init__(self, counting_lines: Dict[str, Dict[str, Any]],
                 parent=None) -> None:
        super().__init__(parent)
        self.counting_lines = counting_lines
        self.active_arm: Optional[str] = None
        self.active_attr: Optional[str] = None

        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ----- coordinate mapping ----------------------------------------------
    def get_frame_coords(self, x_label: int, y_label: int) -> Tuple[int, int]:
        if not self.pixmap() or self.pixmap().isNull():
            return 0, 0

        lbl_w, lbl_h = self.width(), self.height()
        frame_w, frame_h = cfg.width, cfg.height
        scale = min(lbl_w / frame_w, lbl_h / frame_h)

        pix_w = int(frame_w * scale)
        pix_h = int(frame_h * scale)
        pad_x = (lbl_w - pix_w) / 2
        pad_y = (lbl_h - pix_h) / 2

        return (
            int((x_label - pad_x) / scale),
            int((y_label - pad_y) / scale),
        )

    # ----- mouse handling ---------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x_f, y_f = self.get_frame_coords(event.pos().x(), event.pos().y())
        min_dist = 60
        self.active_arm = None
        self.active_attr = None

        for arm, cfg_line in self.counting_lines.items():
            pts = self._line_endpoints(cfg_line)
            for ptype, px, py in pts:
                dist = math.hypot(x_f - px, y_f - py)
                if dist < min_dist:
                    min_dist = dist
                    self.active_arm = arm
                    self.active_attr = ptype

    def mouseMoveEvent(self, event) -> None:
        if self.active_arm is None:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return

        x_f, y_f = self.get_frame_coords(event.pos().x(), event.pos().y())
        x_f = max(0, min(cfg.width, x_f))
        y_f = max(0, min(cfg.height, y_f))
        line_cfg = self.counting_lines[self.active_arm]

        if line_cfg["type"] == "H":
            line_cfg["y"] = y_f
            if self.active_attr == "start":
                line_cfg["x1"] = x_f
            else:
                line_cfg["x2"] = x_f
        else:
            line_cfg["x"] = x_f
            if self.active_attr == "start":
                line_cfg["y1"] = y_f
            else:
                line_cfg["y2"] = y_f

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.active_arm = None

    # ----- helpers ----------------------------------------------------------
    @staticmethod
    def _line_endpoints(cfg_line: Dict[str, Any]):
        if cfg_line["type"] == "H":
            return [
                ("start", cfg_line["x1"], cfg_line["y"]),
                ("end",   cfg_line["x2"], cfg_line["y"]),
            ]
        return [
            ("start", cfg_line["x"], cfg_line["y1"]),
            ("end",   cfg_line["x"], cfg_line["y2"]),
        ]
