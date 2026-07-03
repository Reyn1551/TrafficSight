import threading
from datetime import datetime
import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import DictCursor
from trafficsight.domain.ports import DetectionRepository
from trafficsight.domain.entities import DetectionEvent, LineCrossingEvent
from trafficsight.config import DATABASE_URL
from trafficsight.logger import write_log


class PostgresDetectionRepository(DetectionRepository):
    def __init__(self, dsn: str = DATABASE_URL):
        self._lock = threading.Lock()
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP(3) NOT NULL,
                    camera TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    speed_kmh REAL NOT NULL,
                    cx INTEGER,
                    cy INTEGER,
                    direction TEXT,
                    is_overspeed BOOLEAN DEFAULT FALSE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS line_crossings (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP(3) NOT NULL,
                    camera TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    speed_kmh REAL NOT NULL,
                    direction TEXT
                )
                """
            )
            self._conn.commit()
        write_log(f"Database PostgreSQL siap: {DATABASE_URL}")

    def insert_detection(self, event: DetectionEvent) -> None:
        with self._lock, self._conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                """
                INSERT INTO detections (
                    timestamp, camera, track_id, class_name,
                    speed_kmh, cx, cy, direction, is_overspeed
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
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
        with self._lock, self._conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                """
                INSERT INTO line_crossings (
                    timestamp, camera, track_id, class_name,
                    speed_kmh, direction
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
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
