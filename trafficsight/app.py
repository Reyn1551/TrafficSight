"""QApplication bootstrap for the TrafficSight dashboard."""

from __future__ import annotations

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from .ui.main_window import TrafficSightWindow


def run() -> int:
    """Create the QApplication, show the main window, and enter the event loop."""
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = TrafficSightWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
