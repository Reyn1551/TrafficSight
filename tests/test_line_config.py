"""Unit tests for :mod:`trafficsight.adapters.line_config`."""

from __future__ import annotations

import json

from trafficsight.adapters.line_config import load_lines_config, save_lines_config


def test_load_returns_defaults_when_file_missing():
    lines = load_lines_config("http://no-such/stream")
    assert "Utara" in lines
    assert "Selatan" in lines
    assert "Barat" in lines
    assert "Timur" in lines


def test_save_then_load_roundtrip():
    stream_url = "http://example/test-stream"
    lines = {
        "Utara":   {"type": "H", "y": 100, "x1": 50, "x2": 200},
        "Selatan": {"type": "H", "y": 500, "x1": 50, "x2": 200},
        "Barat":   {"type": "V", "x": 30, "y1": 100, "y2": 500},
        "Timur":   {"type": "V", "x": 800, "y1": 100, "y2": 500},
    }
    save_lines_config(stream_url, lines)
    loaded = load_lines_config(stream_url)
    assert loaded["Utara"]["y"] == 100
    assert loaded["Timur"]["x"] == 800


def test_save_preserves_other_streams():
    stream_a = "http://example/a"
    stream_b = "http://example/b"
    save_lines_config(stream_a, {"Utara": {"type": "H", "y": 10, "x1": 0, "x2": 100}})
    save_lines_config(stream_b, {"Utara": {"type": "H", "y": 999, "x1": 0, "x2": 100}})
    a = load_lines_config(stream_a)
    b = load_lines_config(stream_b)
    assert a["Utara"]["y"] == 10
    assert b["Utara"]["y"] == 999
