import threading
from collections import defaultdict
from datetime import datetime
from trafficsight.logger import write_log
from trafficsight.adapters.line_config import load_lines_config, save_lines_config
from trafficsight.domain.entities import LineCrossingEvent


class VirtualLineCounter:
    def __init__(self, current_stream_url: str, repo=None):
        self.prev_pos = {}
        self.counted = defaultdict(set)
        self.counts_arm = defaultdict(lambda: defaultdict(int))
        self.unique_total = 0
        self._lock = threading.Lock()
        self.current_stream_url = current_stream_url
        self.counting_lines = load_lines_config(current_stream_url)
        self.repo = repo

    def update(self, track_id: int, cx: int, cy: int, class_name: str,
               speed_kmh: float, camera: str):
        with self._lock:
            prev = self.prev_pos.get(track_id)
            self.prev_pos[track_id] = (cx, cy)
            if prev is None:
                return None

            px, py = prev
            crossed_any = None
            for arm, cfg in self.counting_lines.items():
                if track_id in self.counted[arm]:
                    continue

                if cfg["type"] == "H":
                    in_range = cfg["x1"] <= cx <= cfg["x2"]
                    crossed = in_range and ((py < cfg["y"] <= cy) or (py > cfg["y"] >= cy))
                else:
                    in_range = cfg["y1"] <= cy <= cfg["y2"]
                    crossed = in_range and ((px < cfg["x"] <= cx) or (px > cfg["x"] >= cx))

                if not crossed:
                    continue

                event = self._resolve_event(arm, px, py, cx, cy)
                if self._increment_unique(track_id):
                    self.unique_total += 1

                self.counted[arm].add(track_id)
                self.counts_arm[arm][event] += 1
                direction_label = f"{arm}-{event}"
                crossed_any = direction_label
                write_log(f"[LINE] {class_name} #{track_id} {direction_label} {speed_kmh:.1f} km/h")

                if self.repo is not None:
                    self.repo.insert_line_crossing(
                        LineCrossingEvent(
                            timestamp=datetime.now(),
                            camera=camera,
                            track_id=track_id,
                            class_name=class_name,
                            speed_kmh=speed_kmh,
                            direction=direction_label,
                        )
                    )
                break

            return crossed_any

    def _resolve_event(self, arm, px, py, cx, cy):
        if arm == "Utara":
            return "masuk" if cy > py else "keluar"
        if arm == "Selatan":
            return "masuk" if cy < py else "keluar"
        if arm == "Barat":
            return "masuk" if cx > px else "keluar"
        if arm == "Timur":
            return "masuk" if cx < px else "keluar"
        return "lintas"

    def _increment_unique(self, track_id):
        for track_set in self.counted.values():
            if track_id in track_set:
                return False
        return True

    def get_summary(self):
        with self._lock:
            return {
                "unique_total": self.unique_total,
                "per_arm": {k: dict(v) for k, v in self.counts_arm.items()},
            }

    def remove(self, track_id: int):
        with self._lock:
            self.prev_pos.pop(track_id, None)
            for track_set in self.counted.values():
                track_set.discard(track_id)

    def save_config(self):
        save_lines_config(self.current_stream_url, self.counting_lines)
