"""Virtual line counter.

Maintains four configurable counting arms (Utara / Selatan / Barat /
Timur) per stream URL and detects when a track's centroid crosses one.
Each ``(track_id, arm)`` pair is counted at most once per session.
Crossings are classified as ``masuk`` (in) or ``keluar`` (out) based on
the arm's orientation and the direction of motion.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Set

from ..adapters.line_config import load_lines_config, save_lines_config
from ..domain.entities import LineCrossingEvent
from ..domain.ports import DetectionRepository
from ..logger import write_log


class VirtualLineCounter:
    """Thread-safe per-arm directional line counter."""

    def __init__(self, current_stream_url: str,
                 repo: Optional[DetectionRepository] = None) -> None:
        self.current_stream_url = current_stream_url
        self.repo = repo

        self._lock = threading.Lock()
        self._prev_pos: Dict[int, tuple[int, int]] = {}
        self._counted: Dict[str, Set[int]] = defaultdict(set)
        self._counts_arm: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"masuk": 0, "keluar": 0}
        )
        self._unique_total = 0

        # Loaded JSON config — mutable in-place when the user drags lines.
        self.counting_lines = load_lines_config(current_stream_url)

    # ----- main API ---------------------------------------------------------
    def update(self, track_id: int, cx: int, cy: int, class_name: str,
               speed_kmh: float, camera: str) -> Optional[str]:
        """Process a new centroid; return a ``"{arm}-{event}"`` label on crossing."""
        with self._lock:
            prev = self._prev_pos.get(track_id)
            self._prev_pos[track_id] = (cx, cy)
            if prev is None:
                return None

            px, py = prev
            for arm, cfg in self.counting_lines.items():
                if track_id in self._counted[arm]:
                    continue
                if not self._did_cross(cfg, px, py, cx, cy):
                    continue

                event = self._resolve_event(arm, px, py, cx, cy)
                if self._is_new_unique(track_id):
                    self._unique_total += 1
                self._counted[arm].add(track_id)
                self._counts_arm[arm][event] += 1
                label = f"{arm}-{event}"
                write_log(
                    f"[LINE] {class_name} #{track_id} {label} "
                    f"{speed_kmh:.1f} km/h"
                )

                if self.repo is not None:
                    self.repo.insert_line_crossing(
                        LineCrossingEvent(
                            timestamp=datetime.now(),
                            camera=camera,
                            track_id=track_id,
                            class_name=class_name,
                            speed_kmh=speed_kmh,
                            direction=label,
                        )
                    )
                return label

            return None

    # ----- queries ----------------------------------------------------------
    def get_summary(self) -> dict:
        with self._lock:
            return {
                "unique_total": self._unique_total,
                "per_arm": {arm: dict(v) for arm, v in self._counts_arm.items()},
            }

    # ----- maintenance ------------------------------------------------------
    def remove(self, track_id: int) -> None:
        """Drop all state for a track (e.g. after it disappears)."""
        with self._lock:
            self._prev_pos.pop(track_id, None)
            for track_set in self._counted.values():
                track_set.discard(track_id)

    def save_config(self) -> None:
        save_lines_config(self.current_stream_url, self.counting_lines)

    # ----- geometry ---------------------------------------------------------
    @staticmethod
    def _did_cross(cfg: dict, px: int, py: int, cx: int, cy: int) -> bool:
        if cfg["type"] == "H":
            in_range = cfg["x1"] <= cx <= cfg["x2"]
            return in_range and (
                (py < cfg["y"] <= cy) or (py > cfg["y"] >= cy)
            )
        in_range = cfg["y1"] <= cy <= cfg["y2"]
        return in_range and (
            (px < cfg["x"] <= cx) or (px > cfg["x"] >= cx)
        )

    @staticmethod
    def _resolve_event(arm: str, px: int, py: int,
                       cx: int, cy: int) -> str:
        if arm == "Utara":
            return "masuk" if cy > py else "keluar"
        if arm == "Selatan":
            return "masuk" if cy < py else "keluar"
        if arm == "Barat":
            return "masuk" if cx > px else "keluar"
        if arm == "Timur":
            return "masuk" if cx < px else "keluar"
        return "lintas"

    def _is_new_unique(self, track_id: int) -> bool:
        return track_id not in {
            tid for s in self._counted.values() for tid in s
        }
