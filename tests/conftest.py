"""Pytest configuration.

This file is imported by pytest *before* test collection begins. We use
that ordering to set environment variables that ``trafficsight.config``
reads at import time, so tests run with a known config regardless of
the host environment.

Per-test isolation of filesystem paths is handled by the ``_isolated_env``
fixture below.
"""

from __future__ import annotations

import os
import sys
import tempfile

# ----- module-level env setup (runs before test collection) ----------------
# These MUST be set before any ``import trafficsight.*`` happens. We force
# (not setdefault) because the host environment may have its own DATABASE_URL
# that the tests must not pick up.
os.environ["DATABASE_URL"] = (
    "postgresql://trafficsight:trafficsight@localhost:5432/trafficsight"
)
os.environ["TRAFFICSIGHT_MODEL_PATH"] = "/tmp/test-model.pt"

_TMP_ROOT = tempfile.mkdtemp(prefix="trafficsight-tests-")
os.environ["TRAFFICSIGHT_LOG_FILE"] = f"{_TMP_ROOT}/test.log"
os.environ["TRAFFICSIGHT_LINES_FILE"] = f"{_TMP_ROOT}/counting_lines.json"
os.environ["TRAFFICSIGHT_GEOSPATIAL_CALIB_FILE"] = (
    f"{_TMP_ROOT}/geospatial_calibration.json"
)

import pytest  # noqa: E402  (must come after env setup)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Redirect all runtime file paths into a per-test tmp dir.

    Forces a clean re-import of the config module so each test sees the
    fresh paths, even if a previous test mutated the global ``cfg``.
    """
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://trafficsight:trafficsight@localhost:5432/trafficsight",
    )
    monkeypatch.setenv("TRAFFICSIGHT_LOG_FILE", str(tmp_path / "test.log"))
    monkeypatch.setenv(
        "TRAFFICSIGHT_LINES_FILE", str(tmp_path / "counting_lines.json")
    )
    monkeypatch.setenv(
        "TRAFFICSIGHT_GEOSPATIAL_CALIB_FILE",
        str(tmp_path / "geospatial_calibration.json"),
    )
    monkeypatch.setenv("TRAFFICSIGHT_MODEL_PATH", str(tmp_path / "model.pt"))

    for mod in list(sys.modules):
        if mod.startswith("trafficsight"):
            del sys.modules[mod]

    yield

    # Tear down: drop trafficsight modules so the next test re-imports cleanly.
    for mod in list(sys.modules):
        if mod.startswith("trafficsight"):
            del sys.modules[mod]
