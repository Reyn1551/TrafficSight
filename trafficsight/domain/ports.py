"""Repository ports — abstract interfaces the domain exposes to adapters.

Any persistence backend (PostgreSQL, SQLite, in-memory, TimescaleDB)
implements :class:`DetectionRepository`. The application wires a concrete
implementation at startup; tests inject a fake.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .entities import DetectionEvent, LineCrossingEvent


class DetectionRepository(ABC):
    """Persistence port for detection and crossing events."""

    @abstractmethod
    def insert_detection(self, event: DetectionEvent) -> None:
        """Persist a sampled detection row."""

    @abstractmethod
    def insert_line_crossing(self, event: LineCrossingEvent) -> None:
        """Persist a line-crossing event row."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (DB connection, file handles, ...)."""


class InMemoryDetectionRepository(DetectionRepository):
    """Thread-safe in-memory fake, useful for tests and headless runs."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.detections: list[DetectionEvent] = []
        self.crossings: list[LineCrossingEvent] = []

    def insert_detection(self, event: DetectionEvent) -> None:
        with self._lock:
            self.detections.append(event)

    def insert_line_crossing(self, event: LineCrossingEvent) -> None:
        with self._lock:
            self.crossings.append(event)

    def close(self) -> None:
        pass
