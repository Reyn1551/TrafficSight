"""Paced render loop that pulls frames from the streamer, overlays
detection results, and emits Qt signals to the main window.

The thread paces itself at ``1 / fps`` seconds per frame. If the consumer
falls behind by more than 1 s, it resynchronises to ``now`` to avoid
runaway latency.

A persistent trajectory mask accumulates line segments between
consecutive centroids per track ID, producing a heat-map-like trail.
On stop, the mask is composited with the last empty frame and saved as
``trajectory_<timestamp>.png``.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..adapters.streamer import StableStreamer
from ..config import cfg
from ..logger import write_log
from ..services.tracking import DetectionThread


class VideoThread(QThread):
    """Render loop: streamer → overlays → Qt signals."""

    frame_ready = pyqtSignal(np.ndarray)
    stats_ready = pyqtSignal(dict)

    def __init__(self, streamer: StableStreamer, fps: float,
                 detection_thread: Optional[DetectionThread] = None,
                 counting_lines: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        super().__init__()
        self.streamer = streamer
        self.fps = fps
        self.detection_thread = detection_thread
        self.counting_lines = counting_lines or {}

        self.frame_duration = 1.0 / fps
        self.running = True
        self.paused = False

        self._frame_count = 0
        self._health_timer = time.time()
        self._next_frame_time = time.time()

        self._trajectory_mask: Optional[np.ndarray] = None
        self._background_frame: Optional[np.ndarray] = None
        self._last_centroids: Dict[int, tuple[int, int]] = {}

    # ----- QThread ----------------------------------------------------------
    def run(self) -> None:  # noqa: D401 - QThread contract
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            self._tick()

    # ----- public API -------------------------------------------------------
    def pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def stop(self) -> None:
        self.running = False
        self.wait()
        self._save_trajectory()

    # ----- per-frame work ---------------------------------------------------
    def _tick(self) -> None:
        try:
            frame = self.streamer.read(timeout=10.0)
        except queue.Empty:
            self._next_frame_time = time.time()
            time.sleep(0.5)
            return

        detections = self._collect_detections(frame)
        frame = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        self._ensure_buffers(frame)
        if not detections:
            self._background_frame = frame.copy()

        self._draw_counting_lines(frame)
        self._draw_detections(frame, detections)
        self._pace()
        self._emit(frame)

    def _collect_detections(self, frame: np.ndarray) -> list:
        if self.detection_thread is None:
            return []
        self.detection_thread.update_frame(frame.copy())
        return self.detection_thread.get_detections()

    def _ensure_buffers(self, frame: np.ndarray) -> None:
        if self._trajectory_mask is None:
            self._trajectory_mask = np.zeros_like(frame)
        if self._background_frame is None:
            self._background_frame = frame.copy()

    def _draw_counting_lines(self, frame: np.ndarray) -> None:
        for arm, line_cfg in self.counting_lines.items():
            if line_cfg["type"] == "H":
                cv2.line(
                    frame,
                    (line_cfg["x1"], line_cfg["y"]),
                    (line_cfg["x2"], line_cfg["y"]),
                    (0, 255, 255), 2,
                )
                cv2.putText(
                    frame, arm,
                    (line_cfg["x1"], line_cfg["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )
            else:
                cv2.line(
                    frame,
                    (line_cfg["x"], line_cfg["y1"]),
                    (line_cfg["x"], line_cfg["y2"]),
                    (0, 255, 255), 2,
                )
                cv2.putText(
                    frame, arm,
                    (line_cfg["x"] + 5, line_cfg["y1"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )

    def _draw_detections(self, frame: np.ndarray, detections: list) -> None:
        if not detections:
            self._last_centroids.clear()
            return

        current_ids: list[int] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            current_ids.append(det.track_id)

            color = self._color_for(det.class_name, det.speed_kmh)
            self._draw_corner_brackets(frame, (x1, y1, x2, y2), color)
            self._update_trajectory(det.track_id, (cx, cy), color)
            self._draw_velocity_arrow(frame, (cx, cy), det.vx, det.vy)
            self._draw_label(frame, det, (x1, y1), color)

        self._forget_stale(current_ids)

    def _color_for(self, class_name: str, speed_kmh: float) -> tuple[int, int, int]:
        if speed_kmh > cfg.overspeed_kmh:
            return (0, 0, 255)
        val = sum(ord(c) for c in class_name)
        return ((val * 45) % 255, (val * 89) % 255, (val * 123) % 255)

    @staticmethod
    def _draw_corner_brackets(frame, box, color) -> None:
        x1, y1, x2, y2 = box
        length = 15
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2)

    def _update_trajectory(self, track_id: int, centroid, color) -> None:
        if track_id in self._last_centroids:
            cv2.line(
                self._trajectory_mask,  # type: ignore[arg-type]
                self._last_centroids[track_id],
                centroid,
                color, 2,
            )
        self._last_centroids[track_id] = centroid

    @staticmethod
    def _draw_velocity_arrow(frame, centroid, vx, vy) -> None:
        cx, cy = centroid
        cv2.arrowedLine(
            frame,
            (cx, cy),
            (int(cx + vx * 5), int(cy + vy * 5)),
            (255, 0, 255), 2,
        )
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    @staticmethod
    def _draw_label(frame, det, origin, color) -> None:
        x1, y1 = origin
        text = f"[{det.track_id}] {det.class_name.upper()} {det.speed_kmh:.1f}km/h {det.direction}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
        )

    def _forget_stale(self, current_ids: list[int]) -> None:
        stale = [k for k in self._last_centroids if k not in current_ids]
        for k in stale:
            del self._last_centroids[k]

    # ----- pacing + signals -------------------------------------------------
    def _pace(self) -> None:
        now = time.time()
        wait = self._next_frame_time - now
        if wait > 0:
            time.sleep(wait)
        lag = time.time() - self._next_frame_time
        if lag > 1.0:
            self._next_frame_time = time.time() + self.frame_duration
        else:
            self._next_frame_time += self.frame_duration

    def _emit(self, frame: np.ndarray) -> None:
        self.frame_ready.emit(frame)
        self._frame_count += 1

        now = time.time()
        if now - self._health_timer >= 5.0:
            elapsed = now - self._health_timer
            fps_actual = self._frame_count / elapsed if elapsed > 0 else 0.0
            self.stats_ready.emit({
                "fps": fps_actual,
                "buffer": self.streamer.queue_size(),
                "delay": self.streamer.queue_size() / self.fps,
                "target_fps": self.fps,
            })
            self._frame_count = 0
            self._health_timer = now

    # ----- shutdown ---------------------------------------------------------
    def _save_trajectory(self) -> None:
        if self._trajectory_mask is None or self._background_frame is None:
            return
        result = cv2.add(self._background_frame, self._trajectory_mask)
        cam_name = cfg.camera_name_for(self.streamer.src)
        cv2.putText(
            result, "TrafficSight — TRAJECTORY MAP",
            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
        )
        cv2.putText(
            result, f"Location: {cam_name}",
            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            result, time.strftime("%Y-%m-%d %H:%M:%S"),
            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        filename = f"trajectory_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, result)
        write_log(f"Trajectory saved: {filename}")
