# Legacy sources

These files are kept for archaeological reference only. They pre-date the
v2 refactor and are **not imported** by the `trafficsight` package.

| File                            | What it was                                                   |
|---------------------------------|---------------------------------------------------------------|
| `implementation_geospatial.py`  | Monolithic single-file version of the dashboard (PyQt6 + YOLO + Kalman + homography all in one ~1000-line file). |
| `check_database.py`             | Standalone SQLite inspector used during early development.    |

If you are starting from these files and want to migrate to the refactored
package, see [`../README.md`](../README.md) for the new layout and
[`../docs/DESIGN.md`](../docs/DESIGN.md) for the architecture.

To run the legacy monolith directly (not recommended):

```bash
cd legacy
python implementation_geospatial.py
```

To inspect an old SQLite database:

```bash
cd legacy
python check_database.py
```

These scripts have **no** test coverage and may break against newer
Python / PyQt6 / YOLO versions. Use at your own risk.
