"""TrafficSight — real-time traffic analytics for municipal CCTV streams.

Public surface:
    - ``trafficsight.config.cfg`` — validated runtime configuration
    - ``trafficsight.domain``     — pure entities + repository ports
    - ``trafficsight.adapters``   — streamer / Postgres / line-config
    - ``trafficsight.services``   — YOLO + Kalman + speed + line counter
    - ``trafficsight.ui``         — PyQt6 dashboard
"""

from __future__ import annotations

from .config import Config, ConfigError, cfg

__all__ = ["Config", "ConfigError", "cfg"]
__version__ = "2.0.0"
