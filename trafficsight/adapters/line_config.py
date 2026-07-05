"""JSON-backed counting-line configuration.

The on-disk format is a single JSON object keyed by stream URL, so each
camera keeps its own set of four arms. Missing entries fall back to the
``default_counting_lines`` baked into :mod:`trafficsight.config`.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ..config import cfg
from ..logger import write_log


def load_lines_config(current_stream_url: str) -> Dict[str, Dict[str, Any]]:
    """Return the counting-line config for ``current_stream_url``.

    Falls back to :data:`cfg.default_counting_lines` if no saved config
    exists for this stream.
    """
    lines: Dict[str, Dict[str, Any]] = {
        arm: dict(values) for arm, values in cfg.default_counting_lines.items()
    }
    if not cfg.lines_file.exists():
        return lines
    try:
        with open(cfg.lines_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict) and current_stream_url in saved:
            loaded = saved[current_stream_url]
            if isinstance(loaded, dict):
                return {arm: dict(values) for arm, values in loaded.items()}
    except (OSError, json.JSONDecodeError) as exc:
        write_log(f"Gagal memuat {cfg.lines_file}: {exc}")
    return lines


def save_lines_config(current_stream_url: str,
                      lines: Dict[str, Dict[str, Any]]) -> None:
    """Persist the counting-line config for ``current_stream_url``."""
    saved: Dict[str, Any] = {}
    if cfg.lines_file.exists():
        try:
            with open(cfg.lines_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                saved = loaded
        except (OSError, json.JSONDecodeError):
            saved = {}

    saved[current_stream_url] = lines
    cfg.lines_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.lines_file, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=4)
    write_log(f"Konfigurasi garis tersimpan untuk stream: {current_stream_url}")
