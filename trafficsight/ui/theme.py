"""Centralised QSS stylesheet for the TrafficSight dashboard."""

from __future__ import annotations


THEME_QSS = """
QMainWindow, QWidget {
    background: #0a0e17;
}
QLabel {
    color: #e2e8f0;
}
QFrame#VideoFrame {
    background: #000;
    border: 2px solid #1e293b;
    border-radius: 8px;
}
QFrame#ControlFrame {
    background: #0f172a;
    border-radius: 8px;
    border: 1px solid #1e293b;
}
QFrame#MetricCard {
    background: #1e293b;
    border-radius: 6px;
    border: 1px solid #334155;
}
QPushButton {
    color: #fff;
    border: none;
    border-radius: 5px;
    font-weight: bold;
    background: #3b82f6;
}
QPushButton:hover {
    background: #60a5fa;
}
QPushButton#BtnWarning {
    background: #f59e0b;
}
QPushButton#BtnWarning:hover {
    background: #fbbf24;
}
QPushButton#BtnDanger {
    background: #ef4444;
}
QPushButton#BtnDanger:hover {
    background: #f87171;
}
QComboBox {
    background: #1e293b;
    color: #fff;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
    min-width: 160px;
}
QComboBox QAbstractItemView {
    background: #0f172a;
    color: #fff;
    selection-background-color: #3b82f6;
}
QTableWidget#LogTable {
    background: #0f172a;
    alternate-background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
    gridline-color: #334155;
    font-size: 11px;
}
QHeaderView::section {
    background: #1e293b;
    color: #94a3b8;
    font-weight: bold;
    border: none;
    border-bottom: 2px solid #334155;
    padding: 4px;
}
QStatusBar {
    background: #0a0e17;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
}
"""

# Reusable inline styles for metric cards and status pills.
METRIC_TITLE_STYLE = "color:{color};font-size:{size};font-weight:bold;"
METRIC_VALUE_STYLE = "color:white;font-size:{size};font-weight:bold;"
STATUS_LIVE = "color:#3fb950;font-size:16px;font-weight:bold;"
STATUS_PAUSED = "color:#f0883e;font-size:16px;font-weight:bold;"
STATUS_STOPPED = "color:#da3633;font-size:16px;font-weight:bold;"
