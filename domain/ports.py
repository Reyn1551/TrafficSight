from abc import ABC, abstractmethod
from .entities import DetectionEvent, LineCrossingEvent


class DetectionRepository(ABC):
    @abstractmethod
    def insert_detection(self, event: DetectionEvent) -> None:
        raise NotImplementedError()

    @abstractmethod
    def insert_line_crossing(self, event: LineCrossingEvent) -> None:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()
