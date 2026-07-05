"""PostgreSQL implementation of :class:`DetectionRepository`.

The schema (``detections`` and ``line_crossings`` tables) is created
idempotently on first construction. All public methods are guarded by a
``threading.Lock`` so the repository is safe to share between the
detection thread and the line-counter thread.
"""

from __future__ import annotations

import threading
from typing import Optional

import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import DictCursor

from ..config import cfg
from ..domain.entities import DetectionEvent, LineCrossingEvent
from ..domain.ports import DetectionRepository
from ..logger import write_log


_SCHEMA_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMP(3) NOT NULL,
    camera       TEXT NOT NULL,
    track_id     INTEGER NOT NULL,
    class_name   TEXT NOT NULL,
    speed_kmh    REAL NOT NULL,
    cx           INTEGER,
    cy           INTEGER,
    direction    TEXT,
    is_overspeed BOOLEAN DEFAULT FALSE
)
"""

_SCHEMA_LINE_CROSSINGS = """
CREATE TABLE IF NOT EXISTS line_crossings (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMP(3) NOT NULL,
    camera       TEXT NOT NULL,
    track_id     INTEGER NOT NULL,
    class_name   TEXT NOT NULL,
    speed_kmh    REAL NOT NULL,
    direction    TEXT
)
"""

_INSERT_DETECTION = """
INSERT INTO detections (
    timestamp, camera, track_id, class_name,
    speed_kmh, cx, cy, direction, is_overspeed
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_LINE_CROSSING = """
INSERT INTO line_crossings (
    timestamp, camera, track_id, class_name,
    speed_kmh, direction
) VALUES (%s, %s, %s, %s, %s, %s)
"""


class PostgresDetectionRepository(DetectionRepository):
    """Thread-safe PostgreSQL persistence for detection and crossing events."""

    def __init__(self, dsn: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._conn: Optional[Connection] = psycopg2.connect(dsn or cfg.database_url)
        self._conn.autocommit = False
        self._ensure_schema()

    # ----- schema -----------------------------------------------------------
    def _ensure_schema(self) -> None:
        assert self._conn is not None
        with self._conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(_SCHEMA_DETECTIONS)
            cursor.execute(_SCHEMA_LINE_CROSSINGS)
        self._conn.commit()
        write_log(f"Database PostgreSQL siap: {cfg.database_url}")

    # ----- DetectionRepository ---------------------------------------------
    def insert_detection(self, event: DetectionEvent) -> None:
        assert self._conn is not None
        with self._lock, self._conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                _INSERT_DETECTION,
                (
                    event.timestamp,
                    event.camera,
                    event.track_id,
                    event.class_name,
                    event.speed_kmh,
                    event.cx,
                    event.cy,
                    event.direction,
                    event.is_overspeed,
                ),
            )
            self._conn.commit()

    def insert_line_crossing(self, event: LineCrossingEvent) -> None:
        assert self._conn is not None
        with self._lock, self._conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                _INSERT_LINE_CROSSING,
                (
                    event.timestamp,
                    event.camera,
                    event.track_id,
                    event.class_name,
                    event.speed_kmh,
                    event.direction,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                write_log("Koneksi PostgreSQL ditutup.")
