"""Tiny timestamped logger used across the package.

Each call writes a single ``[HH:MM:SS.mmm] message`` line to stdout and
appends it to the configured log file. The logger is intentionally
process-local and not thread-safe across processes; for multi-process
setups, route stdout through journald or a log shipper.
"""

from __future__ import annotations

from datetime import datetime

from .config import cfg


def write_log(message: str) -> None:
    """Print and append ``message`` with a millisecond timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.log_file, "a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")
