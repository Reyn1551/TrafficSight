from dataclasses import dataclass
from datetime import datetime
from typing import Tuple


@dataclass
class DetectionEvent:
    timestamp: datetime
    camera: str
    track_id: int
    class_name: str
    speed_kmh: float
    cx: int
    cy: int
    direction: str
    is_overspeed: bool


@dataclass
class LineCrossingEvent:
    timestamp: datetime
    camera: str
    track_id: int
    class_name: str
    speed_kmh: float
    direction: str


@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    speed_kmh: float
    track_id: int
    vx: float
    vy: float
    direction: str
