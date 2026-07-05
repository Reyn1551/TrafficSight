"""Domain entities.

These dataclasses cross every layer boundary:
- ``DetectionEvent`` and ``LineCrossingEvent`` are the persistence
  records stored in PostgreSQL.
- ``DetectionResult`` is an in-process value object that carries
  per-frame detection state from ``DetectionThread`` to ``VideoThread``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple


@dataclass(frozen=True)
class DetectionEvent:
    """A sampled row in the ``detections`` table (one per ~2s per track)."""

    timestamp: datetime
    camera: str
    track_id: int
    class_name: str
    speed_kmh: float
    cx: int
    cy: int
    direction: str
    is_overspeed: bool


@dataclass(frozen=True)
class LineCrossingEvent:
    """A row in the ``line_crossings`` table (one per actual crossing)."""

    timestamp: datetime
    camera: str
    track_id: int
    class_name: str
    speed_kmh: float
    direction: str


@dataclass(frozen=True)
class DetectionResult:
    """Per-frame detection snapshot exchanged between threads."""

    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    speed_kmh: float
    track_id: int
    vx: float
    vy: float
    direction: str
