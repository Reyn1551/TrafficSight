# TrafficSight

> Real-time traffic analytics dashboard for municipal CCTV streams.
> Detect vehicles, track them across frames, estimate speed via geospatial
> homography, count line crossings per arm, and persist everything to
> PostgreSQL — all from a PyQt6 desktop UI.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/UI-PyQt6-41cd52.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![YOLO](https://img.shields.io/badge/detector-YOLO-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Calibration Tool](#calibration-tool)
- [Database](#database)
- [Testing](#testing)
- [Contributing](#contributing)
- [Further Reading](#further-reading)
- [License](#license)

---

## Overview

TrafficSight turns public ATCS (Area Traffic Control System) CCTV streams —
specifically those published by the City of Yogyakarta — into structured,
queryable traffic data. It runs entirely on a single workstation, ingests
HLS streams via `ffmpeg`, runs a YOLO detector on each frame, smooths
tracking with a Kalman filter, converts pixel velocities into km/h through
a homographic (bird's-eye) calibration, and writes both raw detections and
virtual line-crossing events to PostgreSQL.

The application ships with a dark-themed PyQt6 dashboard that overlays
bounding boxes, velocity arrows, trajectory trails, and the four
configurable counting arms (`Utara` / `Selatan` / `Barat` / `Timur`) on
top of the live video. Users can drag counting-line endpoints directly on
the video, switch between cameras at runtime, and export a trajectory heat
map on shutdown.

The project is organised as a small hexagonal architecture: a pure
`domain` layer (entities + repository port), `adapters` for I/O (streamer,
PostgreSQL, JSON config), `services` for computer-vision logic, and a `ui`
package for the PyQt6 frontend. The boundary means the core analytics can
be exercised from a CLI, a future REST API, or an AI agent without
touching the GUI code.

## Key Features

- **Multi-camera HLS ingest** with auto-FPS detection via `ffprobe` and
  seamless reconnect on stream drop.
- **YOLO + ByteTrack** object detection with persistent IDs across frames.
- **Kalman-filtered box tracking** for stable centroids and velocity
  vectors even under partial occlusion.
- **Geospatial speed estimation** — a 4-point homography warps the camera
  view to a bird's-eye plane, then `ppm_x` / `ppm_y` (pixels-per-meter)
  convert pixel deltas to km/h. Falls back to a perspective-aware heuristic
  when calibration is absent.
- **Virtual line counter** — four draggable arms (N/S/E/W) detect
  directional crossings (`masuk` / `keluar`) per unique track ID, with
  per-stream line configs persisted to `counting_lines.json`.
- **Overspeed detection** with a configurable threshold (default 60 km/h)
  and a hard speed cap (140 km/h) to suppress absurd outliers.
- **PostgreSQL persistence** — `detections` (sampled per track every 2s)
  and `line_crossings` (one row per crossing event) tables are created
  idempotently on startup.
- **Interactive UI** — drag counting lines on the video, switch cameras,
  pause / resume, save trajectory PNG on exit.
- **Pluggable repository** — the `DetectionRepository` port makes it
  trivial to swap PostgreSQL for SQLite, TimescaleDB, or an in-memory
  store for tests.

## How It Works

```
 HLS stream ──▶ StableStreamer ──▶ VideoThread ──▶ DetectionThread ──▶ PostgreSQL
 (ffmpeg)         (queue,           (pace +          (YOLO + Kalman +
                  reconnect)         draw)            homography + counter)
```

1. `StableStreamer` runs `ffmpeg` as a subprocess, decodes raw BGR24
   frames, and pushes them onto a bounded `queue.Queue` with automatic
   reconnect on EOF.
2. `VideoThread` paces frames at the stream's real FPS, hands each frame
   to `DetectionThread`, retrieves the latest detections, draws overlays
   (boxes, arrows, trajectory mask, counting lines), and emits the
   rendered frame to the UI.
3. `DetectionThread` runs `YOLO.track()`, maintains a `KalmanBoxTracker`
   per active ID, computes speed via `SpeedEstimator`, classifies
   direction from the velocity vector, and persists a `DetectionEvent`
   to PostgreSQL every 2 seconds per track.
4. `VirtualLineCounter` receives `(track_id, cx, cy)` updates and detects
   H/V line crossings; each new crossing produces a `LineCrossingEvent`
   row.
5. The UI subscribes to `frame_ready` and `stats_ready` Qt signals to
   paint the video and update metric cards (FPS, buffer, delay,
   detections, overspeed, crossings).

## Project Layout

```
trafficsight/
├── __init__.py
├── __main__.py            # `python -m trafficsight`
├── app.py                 # QApplication bootstrap
├── config.py              # validated settings (env + defaults)
├── logger.py
├── domain/                # pure entities + ports (no I/O)
│   ├── entities.py
│   └── ports.py
├── adapters/              # infra implementations
│   ├── streamer.py
│   ├── postgres_repository.py
│   └── line_config.py
├── services/              # CV + business logic
│   ├── tracking.py
│   ├── geospatial.py
│   ├── kalman.py
│   └── line_counter.py
├── ui/                    # PyQt6 frontend
│   ├── main_window.py
│   ├── video_label.py
│   ├── video_thread.py
│   ├── theme.py
│   └── fps_detector.py
├── tools/                 # standalone utilities
│   ├── geospatial_calibration_gui.py
│   └── migrate_sqlite_to_postgres.py
└── data/                  # runtime JSON configs
    ├── counting_lines.json
    └── geospatial_calibration.json
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full architecture walk-through.

## Prerequisites

| Dependency   | Version  | Notes                                              |
|--------------|----------|----------------------------------------------------|
| Python       | 3.10+    | Tested on 3.11 / 3.12                              |
| ffmpeg       | 6.0+     | Must be on `PATH`; used for HLS ingest             |
| ffprobe      | 6.0+     | Bundled with ffmpeg; used for FPS detection        |
| PostgreSQL   | 13+      | Local or remote; schema auto-created on first run  |
| CUDA (opt.)  | 11.8+    | YOLO inference is ~5× faster on GPU                |

A YOLO model trained on traffic classes is **required**. Point
`TRAFFICSIGHT_MODEL_PATH` at a `*.pt` file (any Ultralytics YOLOv8 / YOLO11
checkpoint will load).

## Installation

```bash
# 1. Clone
git clone https://github.com/your-org/TrafficSight.git
cd TrafficSight

# 2. Create venv
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install Python deps
pip install -r requirements.txt
# or, for an editable install with all extras:
pip install -e ".[dev]"

# 4. System deps (Debian/Ubuntu example)
sudo apt-get install -y ffmpeg postgresql postgresql-contrib
```

### PostgreSQL setup

```bash
sudo -u postgres psql <<'SQL'
CREATE USER trafficsight WITH PASSWORD 'trafficsight';
CREATE DATABASE trafficsight OWNER trafficsight;
GRANT ALL PRIVILEGES ON DATABASE trafficsight TO trafficsight;
SQL
```

The schema (`detections`, `line_crossings`) is created automatically on
first launch — no manual migration step is needed for new installs.

## Configuration

All runtime settings are read from environment variables with sensible
defaults baked into [`trafficsight/config.py`](trafficsight/config.py).
The config module validates types and ranges at import time and raises a
clear `ConfigError` if anything is off.

| Variable                    | Default                                          | Description                              |
|-----------------------------|--------------------------------------------------|------------------------------------------|
| `DATABASE_URL`              | `postgresql://trafficsight:trafficsight@...`     | psycopg2 connection DSN                  |
| `TRAFFICSIGHT_MODEL_PATH`   | `models/best_traffic_model.pt`                   | Path to YOLO weights                     |
| `TRAFFICSIGHT_STREAM_URLS`  | (built-in Jogja ATCS map)                        | JSON string of `name -> url` overrides   |
| `TRAFFICSIGHT_WIDTH`        | `1920`                                           | Decode width                             |
| `TRAFFICSIGHT_HEIGHT`       | `1080`                                           | Decode height                            |
| `TRAFFICSIGHT_FALLBACK_FPS` | `25.0`                                           | Used when `ffprobe` fails                |
| `TRAFFICSIGHT_OVERSPEED_KMH`| `60.0`                                           | Overspeed threshold                      |
| `TRAFFICSIGHT_SPEED_CAP_KMH`| `140.0`                                          | Hard ceiling on reported speed           |
| `TRAFFICSIGHT_BUFFER_SECONDS`| `60`                                            | Streamer queue depth in seconds          |
| `TRAFFICSIGHT_LOG_FILE`     | `trafficSight.log`                               | Log file path                            |

Example:

```bash
export DATABASE_URL='postgresql://trafficsight:secret@db.local:5432/trafficsight'
export TRAFFICSIGHT_MODEL_PATH='/opt/models/best_traffic_model.pt'
export TRAFFICSIGHT_OVERSPEED_KMH=50
```

## Running the App

```bash
# Activate venv, set env vars, then:
python -m trafficsight
```

The first launch will:

1. Probe the stream with `ffprobe` and lock onto its real FPS.
2. Open a warm-up period (until the buffer reaches ~15 frames).
3. Start the detection thread (loads YOLO into memory).
4. Show the live dashboard with metric cards, log table, and per-arm
   crossing counts.

Use the camera dropdown to switch streams at runtime. Use **⚙️ Edit Garis**
to drag counting-line endpoints with the mouse — press **💾 Simpan Garis**
to persist the new geometry to `counting_lines.json`.

## Calibration Tool

Speed estimation requires a per-camera homography. Launch the calibration
GUI once per camera:

```bash
python -m trafficsight.tools.geospatial_calibration_gui
```

Workflow:

1. The GUI grabs one frame from the current stream.
2. Drag the 4 red corners so the rectangle aligns with a flat, visible
   road segment.
3. Open Google Earth (link in the GUI), measure the **real-world width**
   (corner 1 → 2) and **length** (corner 1 → 4) in meters.
4. Enter the two values, click **Hitung Kalibrasi (PPM) & Simpan**.
5. A bird's-eye preview window opens for sanity check; the calibration is
   written to `trafficsight/data/geospatial_calibration.json`.

The `SpeedEstimator` reads this file at startup. Without it, speeds are
approximated using a perspective heuristic — accurate enough for relative
comparison but not for absolute km/h.

## Database

### Schema

```sql
CREATE TABLE detections (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMP(3) NOT NULL,
    camera       TEXT NOT NULL,
    track_id     INTEGER NOT NULL,
    class_name   TEXT NOT NULL,
    speed_kmh    REAL NOT NULL,
    cx           INTEGER,
    cy           INTEGER,
    direction    TEXT,
    is_overspeed BOOLEAN DEFAULT FALSE
);

CREATE TABLE line_crossings (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMP(3) NOT NULL,
    camera       TEXT NOT NULL,
    track_id     INTEGER NOT NULL,
    class_name   TEXT NOT NULL,
    speed_kmh    REAL NOT NULL,
    direction    TEXT
);
```

### Migrating from SQLite

If you have an older `trafficSight_traffic.db` from a pre-PostgreSQL
install, one-shot migration is supported:

```bash
python -m trafficsight.tools.migrate_sqlite_to_postgres
```

The script is idempotent and skips rows that already exist by primary key
(checkpointing is left as a future improvement — see
[`docs/GOALS.md`](docs/GOALS.md)).

### Quick queries

```sql
-- Top 10 fastest vehicles in the last hour
SELECT camera, track_id, class_name, speed_kmh, timestamp
FROM detections
WHERE timestamp > now() - interval '1 hour'
ORDER BY speed_kmh DESC
LIMIT 10;

-- Per-arm crossing counts today
SELECT direction,
       split_part(direction, '-', 1) AS arm,
       split_part(direction, '-', 2) AS movement,
       count(*) AS crossings
FROM line_crossings
WHERE timestamp::date = current_date
GROUP BY direction
ORDER BY arm, movement;
```

## Testing

The repository ships with pytest stubs covering the pure logic
(`VirtualLineCounter`, `KalmanBoxTracker`, `SpeedEstimator`, line config
I/O). The UI and live-stream paths are not unit-tested because they
require a display and a network camera — they are validated manually.

```bash
pytest -q
```

To extend coverage, mock the `DetectionRepository` port and exercise the
`DetectionThread` loop with a recorded video instead of a live stream.

## Contributing

Pull requests are welcome. Please:

1. Open an issue first to discuss non-trivial changes.
2. Follow the existing code style (`black`, `isort`, `ruff`).
3. Add or update tests for any change to `services/` or `domain/`.
4. Keep the `domain` layer free of imports from `adapters`, `services`, or
   `ui` — the dependency direction is enforced by review, not by tooling.
5. Update `docs/` if your change affects architecture, configuration, or
   the public Python API.

See [`docs/DESIGN.md`](docs/DESIGN.md) for layering rules and
[`docs/AGENT.md`](docs/AGENT.md) for the AI-agent integration surface.

## Further Reading

| Document | Audience | What's inside |
|----------|----------|---------------|
| [`docs/GOALS.md`](docs/GOALS.md)            | PMs, maintainers   | Functional / non-functional goals, success metrics, roadmap |
| [`docs/DESIGN.md`](docs/DESIGN.md)          | Engineers          | Hexagonal layers, sequence diagrams, data model, threading model |
| [`docs/AGENT.md`](docs/AGENT.md)            | AI/automation      | Spec for an LLM agent that queries the DB and controls streams |
| [`legacy/`](legacy/)                        | Curious            | Pre-refactor monolithic source kept for archaeological reference |

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
