"""Unit tests for :mod:`trafficsight.config` validation.

These tests exercise the validation logic in :class:`Config.validate` by
directly constructing ``Config`` instances, rather than rebuilding the
module-level ``cfg``. This avoids the classic "reload creates a new class"
pitfall where ``ConfigError`` from a re-imported module is a different
class object than the one the test imported at collection time.
"""

from __future__ import annotations

import pytest

from trafficsight.config import Config, ConfigError


def _valid_kwargs(**overrides):
    """Return kwargs that build a valid Config, with optional overrides."""
    base = dict(
        base_dir="/tmp",
        package_dir="/tmp",
        log_file="/tmp/test.log",
        lines_file="/tmp/lines.json",
        geospatial_calib_file="/tmp/calib.json",
        database_url="postgresql://u:p@localhost:5432/db",
        model_path="/tmp/model.pt",
        stream_urls={"cam": "http://cam"},
        default_stream_name="cam",
        width=1920,
        height=1080,
        fallback_fps=25.0,
        buffer_seconds=60,
        overspeed_kmh=60.0,
        speed_cap_kmh=140.0,
    )
    base.update(overrides)
    return base


def test_default_config_is_valid():
    """The module-level cfg should pass validation on a clean env."""
    from trafficsight.config import cfg
    cfg.validate()  # does not raise


def test_invalid_database_url_raises():
    with pytest.raises(ConfigError):
        Config(**_valid_kwargs(database_url="mysql://nope")).validate()


def test_overspeed_must_be_below_speed_cap():
    with pytest.raises(ConfigError):
        Config(
            **_valid_kwargs(overspeed_kmh=200.0, speed_cap_kmh=100.0)
        ).validate()


def test_invalid_width_raises():
    with pytest.raises(ConfigError):
        Config(**_valid_kwargs(width=0)).validate()


def test_invalid_stream_name_raises():
    with pytest.raises(ConfigError):
        Config(
            **_valid_kwargs(default_stream_name="missing")
        ).validate()


def test_camera_name_for_returns_unknown_when_missing():
    cfg = Config(**_valid_kwargs())
    assert cfg.camera_name_for("http://not-in-map") == "Unknown"
    assert cfg.camera_name_for("http://cam") == "cam"


def test_default_stream_url_derives_from_name():
    cfg = Config(**_valid_kwargs())
    assert cfg.default_stream_url == "http://cam"
