# TrafficSight — System Goals

> What TrafficSight exists to do, how we measure success, and where it is
> headed next.

---

## 1. Vision

TrafficSight aims to be the **simplest open-source toolkit for turning
municipal CCTV into structured traffic data**. A single workstation, a
public HLS feed, and a YOLO model are enough to produce speed histograms,
directional counts, and overspeed alerts that can feed dashboards,
research, and policy decisions — without proprietary hardware or vendor
lock-in.

The project is opinionated about *realism*: it assumes jittery ATCS
streams, imperfect homographies, and noisy detectors, and it engineers
around those realities (Kalman smoothing, perspective fallback, hard
speed caps, idempotent schema creation) rather than pretending they do
not exist.

## 2. Stakeholders

| Stakeholder          | Interest in TrafficSight                                   |
|----------------------|------------------------------------------------------------|
| City traffic ops     | Live situational awareness, historical audit trail         |
| Traffic researchers  | Open dataset of per-vehicle speeds and turning movements  |
| Students / hobbyists | A reproducible CV+engineering reference project           |
| Maintainers          | A small, well-layered codebase they can keep improving    |
| AI/automation devs   | A clean Python+PostgreSQL surface to bolt agents onto     |

## 3. Functional Goals

### F1 — Real-time detection & tracking
- Ingest HLS CCTV streams at native resolution and FPS.
- Detect vehicles (car, motorcycle, bus, truck) with a YOLO model.
- Assign persistent track IDs across frames via ByteTrack.
- Smooth each track's bounding box with a per-ID Kalman filter.

### F2 — Speed estimation grounded in real-world units
- Convert pixel-space velocity to **km/h** using a per-camera homography
  calibrated against real-world measurements.
- Fall back to a perspective-aware heuristic when calibration is missing,
  so the system degrades gracefully rather than crashing.
- Cap reported speed at a sanity ceiling (default 140 km/h) to suppress
  artifacts from track loss or frame jitter.

### F3 — Directional line counting
- Maintain four configurable counting arms (Utara / Selatan / Barat /
  Timur) per stream, persisted across sessions.
- Detect line crossings and classify each as `masuk` (in) or `keluar`
  (out) per arm.
- Count each unique track ID at most once per arm.
- Allow operators to drag line endpoints in the UI at runtime.

### F4 — Persistent, queryable storage
- Persist a sampled detection row every 2 seconds per active track.
- Persist one row per crossing event.
- Auto-create the PostgreSQL schema on first launch.
- Provide a one-shot SQLite → PostgreSQL migration for legacy installs.

### F5 — Operator UI
- Dark-themed PyQt6 dashboard with live video overlay (boxes, arrows,
  trajectory trails, counting lines).
- Metric cards for FPS, buffer depth, end-to-end delay, total detections,
  overspeed count, unique crossings.
- Sortable log table of the 30 fastest current tracks.
- Camera switch, pause/resume, stop with trajectory PNG export.

### F6 — Configuration & calibration
- All runtime knobs exposed via environment variables with safe defaults.
- A standalone calibration GUI for the per-camera homography.
- JSON-persisted per-stream line configurations.

## 4. Non-Functional Goals

| ID | Category         | Target                                                                                   |
|----|------------------|------------------------------------------------------------------------------------------|
| NF1| Availability     | Auto-reconnect within 2 s of stream drop; resume without restart                        |
| NF2| Latency          | End-to-end frame delay < 5 s under 25 FPS stream with 60 s buffer                        |
| NF3| Throughput       | Sustain 25 FPS render loop on a mid-range GPU (RTX 3060 / Apple M1)                      |
| NF4| Accuracy         | Speed estimation within ±15 % of ground truth when homography is calibrated             |
| NF5| Portability      | Runs on Linux, macOS, Windows with Python 3.10+; no OS-specific deps in core            |
| NF6| Maintainability  | Clear layering; `domain` has no infra imports; public surface is < 30 modules           |
| NF7| Observability    | Every long-running action emits a timestamped log line; FPS/buffer visible in UI        |
| NF8| Security         | No credentials in source; all secrets via env vars; DB user follows least-privilege     |
| NF9| Extensibility    | Adding a new repository backend (e.g. TimescaleDB) is a single class + DI wiring         |
| NF10|Testability      | Pure services are unit-testable with no Qt/cv2/network deps; mocks for I/O ports         |

## 5. Success Metrics

| Metric                                | Source            | Target (v1)               |
|---------------------------------------|-------------------|---------------------------|
| Stream uptime (auto-reconnect count)  | log file          | ≥ 99 % over 1 h           |
| Frame render rate                     | UI metric card    | ≥ 0.8 × stream FPS        |
| Median buffer depth                   | UI metric card    | ≤ 2 s                     |
| Detection throughput                  | log file          | ≥ 20 inferences/sec (GPU) |
| Speed MAPE vs. radar ground truth     | offline eval      | ≤ 15 %                    |
| Crossing counts vs. manual count      | offline eval      | ±5 % per arm per 10 min   |
| Postgres rows / min (steady state)    | DB                | 30–120 (10–40 vehicles)   |
| Cold-start time to first frame        | log file          | ≤ 8 s                     |

## 6. Use Cases

### UC-1: Operator live monitoring
A traffic operator opens TrafficSight, selects a camera, and watches the
dashboard for overspeed vehicles. The log table highlights overspeed rows
in red; the operator can pause to inspect a frame and resume without
losing the stream.

### UC-2: After-the-fact analysis
A researcher queries the `detections` and `line_crossings` tables directly
from a notebook to compute hourly speed distributions, peak-hour
directional splits, and per-class counts. They never need to launch the
GUI.

### UC-3: Per-camera calibration
An engineer deploys TrafficSight against a new camera. They run the
calibration GUI once, enter the Google-Earth-measured road dimensions,
and the system starts producing calibrated km/h speeds within minutes.

### UC-4: AI agent integration
An LLM agent (see [`AGENT.md`](AGENT.md)) is wired up to answer natural
language questions like *"how many trucks went east through Sugeng Jeroni
between 08:00 and 09:00 today?"* by translating them into SQL against the
two tables, with no manual report writing.

### UC-5: Migration from legacy install
A site running an old SQLite-based version runs the migration script,
verifies the row counts, and switches the running app to PostgreSQL with
no downtime beyond a single restart.

## 7. Out of Scope (v1)

- **Multi-camera fusion** — each stream is processed independently; no
  cross-camera re-identification is attempted.
- **Automatic number-plate recognition (ANPR)** — plates are not captured
  or stored.
- **Web UI** — the frontend is desktop-only. A web dashboard would be a
  separate project reading from the same Postgres.
- **Real-time alerting** — overspeed events are flagged in the UI and DB
  but no SMS/email/webhook is sent.
- **Cloud deployment** — designed for a single workstation; horizontal
  scaling is out of scope.

## 8. Roadmap

| Horizon | Theme                  | Concrete items                                                             |
|---------|------------------------|----------------------------------------------------------------------------|
| v1.1    | Observability          | Structured JSON logs, Prometheus metrics endpoint, runbook                 |
| v1.2    | Calibration UX         | In-app calibration (no separate GUI), per-stream config persistence        |
| v1.3    | Accuracy               | Optional radar/lidar fusion, multi-point calibration (>= 4 corners)        |
| v1.4    | Storage                | TimescaleDB hypertables, retention policies, downsampling for long history |
| v2.0    | Decoupling             | Headless mode (no Qt), REST API for control, official AI agent bindings    |
| v2.1    | Multi-camera           | Cross-camera re-ID, intersection-wide counts                               |

## 9. Risks & Mitigations

| Risk                                          | Mitigation                                                          |
|-----------------------------------------------|---------------------------------------------------------------------|
| Stream goes dark / RTMP drops                 | `StableStreamer` auto-reconnects every 2 s; UI shows DISCONNECTED   |
| YOLO model drifts on new camera angles        | Calibration GUI + per-stream line configs; model path is env-var    |
| Homography skew inflates speeds               | Hard 140 km/h cap, EMA smoothing, fallback heuristic when no calib  |
| Postgres write contention under high load     | Per-track 2 s throttle; `commit()` per insert (batching is roadmap) |
| Track-ID reuse by ByteTrack mid-session       | `line_counter.counted[arm]` is per-ID, never per-frame              |
| Operator drags line mid-crossing              | Line geometry changes are persisted but in-flight crossings finish  |
| venv on shared workstation                    | All deps pinned in `requirements.txt`; editable install supported   |

## 10. Acceptance Criteria for v1

A v1 release is shippable when:

1. A fresh `git clone` + `pip install -e .` + `python -m trafficsight`
   launches the dashboard against the default Jogja ATCS stream with no
   manual steps other than setting `DATABASE_URL` and
   `TRAFFICSIGHT_MODEL_PATH`.
2. Switching cameras at runtime does not require an app restart.
3. The PostgreSQL schema is created on first run; no SQL file needs to
   be applied by hand.
4. Pulling the network cable for 10 s and re-plugging does not crash the
   app — it reconnects and resumes drawing within 5 s.
5. `pytest -q` is green on a clean checkout.
6. Stopping the app writes a `trajectory_*.png` next to the log file.
7. The calibration GUI can produce a `geospatial_calibration.json` that,
   when loaded, produces km/h readings within ±15 % of a hand-measured
   reference for at least one vehicle.
