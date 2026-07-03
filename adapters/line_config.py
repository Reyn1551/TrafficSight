import json
from typing import Any, Dict
from config import LINES_FILE, DEFAULT_COUNTING_LINES
from logger import write_log


def load_lines_config(current_stream_url: str) -> Dict[str, Any]:
    lines = DEFAULT_COUNTING_LINES.copy()
    if LINES_FILE.exists():
        try:
            with open(LINES_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict) and current_stream_url in saved:
                    lines = saved[current_stream_url]
        except Exception as exc:
            write_log(f"Gagal memuat {LINES_FILE}: {exc}")
    return lines


def save_lines_config(current_stream_url: str, lines: Dict[str, Any]) -> None:
    saved = {}
    if LINES_FILE.exists():
        try:
            with open(LINES_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            saved = {}

    saved[current_stream_url] = lines
    with open(LINES_FILE, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=4)
    write_log(f"Konfigurasi garis tersimpan untuk stream: {current_stream_url}")
