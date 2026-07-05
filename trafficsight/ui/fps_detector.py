"""Stream FPS detection via ``ffprobe``.

Used once at stream startup to lock onto the native FPS so the render
loop can pace frames accurately. Falls back to
:data:`cfg.fallback_fps` when ``ffprobe`` is unavailable or returns an
unparseable value.
"""

from __future__ import annotations

import subprocess

from ..config import cfg
from ..logger import write_log

_FPS_MIN = 1.0
_FPS_MAX = 120.0


def detect_stream_fps(url: str, timeout: int = 15) -> float:
    """Probe ``url`` with ``ffprobe`` and return its frame rate."""
    write_log("Mendeteksi FPS asli stream...")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "csv=p=0",
        url,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.SubprocessError, OSError) as exc:
        write_log(f"ffprobe error: {exc}")
        return _fallback()

    for part in result.stdout.replace("\n", ",").split(","):
        fps = _parse_part(part.strip())
        if fps is not None:
            write_log(f"FPS terdeteksi: {fps:.3f}")
            return fps

    return _fallback()


def _parse_part(part: str) -> float | None:
    if not part:
        return None
    if "/" in part:
        try:
            num, den = part.split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    else:
        try:
            fps = float(part)
        except ValueError:
            return None
    if _FPS_MIN < fps < _FPS_MAX:
        return fps
    return None


def _fallback() -> float:
    write_log(f"Pakai fallback FPS: {cfg.fallback_fps}")
    return cfg.fallback_fps
