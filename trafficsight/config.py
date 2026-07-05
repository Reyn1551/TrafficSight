"""Validated runtime configuration for TrafficSight.

All settings are read from environment variables with safe defaults.
The module exposes a single ``cfg`` instance that other modules import.
Validation runs at import time and raises :class:`ConfigError` on bad
input so that the app fails fast rather than misbehaving at runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when the environment configuration is invalid."""


# ----- helpers --------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float, got {raw!r}") from exc


def _env_path(name: str, default: Path | str) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else Path(default)


def _env_json_dict(name: str, default: dict[str, str]) -> dict[str, str]:
    raw = os.getenv(name)
    if not raw:
        return dict(default)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must be a JSON object, got {raw!r}") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ConfigError(f"{name} must be a dict[str, str], got {parsed!r}")
    return parsed


# ----- defaults -------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

DEFAULT_STREAM_URLS: dict[str, str] = {
    "Sugeng Jeroni 2": "http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8",
    "Simpang Wirosaban Barat": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/playlist.m3u8",
    "Simpang Pingit": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_Pingit2.stream/playlist.m3u8",
}

DEFAULT_COUNTING_LINES: dict[str, dict[str, Any]] = {
    "Utara":   {"type": "H", "y": 310, "x1": 350, "x2": 700},
    "Selatan": {"type": "H", "y": 580, "x1": 350, "x2": 750},
    "Barat":   {"type": "V", "x": 150, "y1": 310, "y2": 580},
    "Timur":   {"type": "V", "x": 970, "y1": 260, "y2": 520},
}


# ----- config dataclass -----------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Immutable, validated runtime configuration."""

    base_dir: Path
    package_dir: Path
    log_file: Path
    lines_file: Path
    geospatial_calib_file: Path

    database_url: str
    model_path: Path

    stream_urls: dict[str, str]
    default_stream_name: str

    width: int
    height: int
    fallback_fps: float
    buffer_seconds: int

    overspeed_kmh: float
    speed_cap_kmh: float

    default_counting_lines: dict[str, dict[str, Any]] = field(
        default_factory=lambda: _deep_copy_lines(DEFAULT_COUNTING_LINES)
    )

    # ----- derived helpers --------------------------------------------------
    @property
    def default_stream_url(self) -> str:
        return self.stream_urls[self.default_stream_name]

    def camera_name_for(self, url: str) -> str:
        return next(
            (name for name, u in self.stream_urls.items() if u == url),
            "Unknown",
        )

    # ----- validation -------------------------------------------------------
    def validate(self) -> None:
        if not (self.database_url.startswith("postgresql://")
                or self.database_url.startswith("postgres://")):
            raise ConfigError(
                "DATABASE_URL must start with 'postgresql://' or 'postgres://', "
                f"got {self.database_url!r}"
            )
        if self.width <= 0 or self.width > 7680:
            raise ConfigError(f"width must be in (0, 7680], got {self.width}")
        if self.height <= 0 or self.height > 7680:
            raise ConfigError(f"height must be in (0, 7680], got {self.height}")
        if not (1.0 < self.fallback_fps < 120.0):
            raise ConfigError(
                f"fallback_fps must be in (1.0, 120.0), got {self.fallback_fps}"
            )
        if self.overspeed_kmh <= 0:
            raise ConfigError(
                f"overspeed_kmh must be positive, got {self.overspeed_kmh}"
            )
        if self.speed_cap_kmh <= self.overspeed_kmh:
            raise ConfigError(
                f"speed_cap_kmh ({self.speed_cap_kmh}) must be greater than "
                f"overspeed_kmh ({self.overspeed_kmh})"
            )
        if not self.stream_urls:
            raise ConfigError("stream_urls must not be empty")
        if self.default_stream_name not in self.stream_urls:
            raise ConfigError(
                f"default_stream_name {self.default_stream_name!r} is not in "
                f"stream_urls"
            )
        if self.buffer_seconds <= 0:
            raise ConfigError(
                f"buffer_seconds must be positive, got {self.buffer_seconds}"
            )


def _deep_copy_lines(lines: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {arm: dict(cfg) for arm, cfg in lines.items()}


# ----- builder --------------------------------------------------------------

def _build() -> Config:
    package_dir = PACKAGE_DIR
    base_dir = BASE_DIR

    stream_urls = _env_json_dict(
        "TRAFFICSIGHT_STREAM_URLS", DEFAULT_STREAM_URLS
    )

    cfg_ = Config(
        base_dir=base_dir,
        package_dir=package_dir,
        log_file=_env_path(
            "TRAFFICSIGHT_LOG_FILE",
            base_dir / "trafficSight.log",
        ),
        lines_file=_env_path(
            "TRAFFICSIGHT_LINES_FILE",
            package_dir / "data" / "counting_lines.json",
        ),
        geospatial_calib_file=_env_path(
            "TRAFFICSIGHT_GEOSPATIAL_CALIB_FILE",
            package_dir / "data" / "geospatial_calibration.json",
        ),
        database_url=_env(
            "DATABASE_URL",
            "postgresql://trafficsight:trafficsight@localhost:5432/trafficsight",
        ),
        model_path=_env_path(
            "TRAFFICSIGHT_MODEL_PATH",
            base_dir / "models" / "best_traffic_model.pt",
        ),
        stream_urls=stream_urls,
        default_stream_name=_env(
            "TRAFFICSIGHT_DEFAULT_STREAM_NAME",
            "Sugeng Jeroni 2",
        ),
        width=_env_int("TRAFFICSIGHT_WIDTH", 1920),
        height=_env_int("TRAFFICSIGHT_HEIGHT", 1080),
        fallback_fps=_env_float("TRAFFICSIGHT_FALLBACK_FPS", 25.0),
        buffer_seconds=_env_int("TRAFFICSIGHT_BUFFER_SECONDS", 60),
        overspeed_kmh=_env_float("TRAFFICSIGHT_OVERSPEED_KMH", 60.0),
        speed_cap_kmh=_env_float("TRAFFICSIGHT_SPEED_CAP_KMH", 140.0),
    )
    cfg_.validate()

    if not cfg_.model_path.exists():
        # Not fatal — UI can boot in preview-only mode without YOLO.
        import sys
        print(
            f"[Config] WARNING: model_path {cfg_.model_path} does not exist; "
            "detection thread will fail to load YOLO. Set TRAFFICSIGHT_MODEL_PATH.",
            file=sys.stderr,
        )

    return cfg_


cfg: Config = _build()
