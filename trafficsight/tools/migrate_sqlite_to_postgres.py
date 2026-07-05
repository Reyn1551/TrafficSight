"""One-shot SQLite → PostgreSQL migration for legacy TrafficSight installs.

Reads ``trafficSight_traffic.db`` (override with ``SQLITE_DB``) and
copies all rows from ``detections`` and ``line_crossings`` into the
configured PostgreSQL instance. The PostgreSQL schema is created
idempotently if it does not already exist.

The script is **not** resumable — running it twice will insert duplicate
rows. Back up the target database before re-running.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import Iterable

from ..config import cfg
from ..logger import write_log

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError as exc:
    raise SystemExit(
        "psycopg2 is required to run this migration. Install it with "
        "`pip install psycopg2-binary`."
    ) from exc


_SCHEMA_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
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
)
"""

_SCHEMA_LINE_CROSSINGS = """
CREATE TABLE IF NOT EXISTS line_crossings (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMP(3) NOT NULL,
    camera       TEXT NOT NULL,
    track_id     INTEGER NOT NULL,
    class_name   TEXT NOT NULL,
    speed_kmh    REAL NOT NULL,
    direction    TEXT
)
"""

_TABLE_COLUMNS = {
    "detections": [
        "timestamp", "camera", "track_id", "class_name",
        "speed_kmh", "cx", "cy", "direction", "is_overspeed",
    ],
    "line_crossings": [
        "timestamp", "camera", "track_id", "class_name",
        "speed_kmh", "direction",
    ],
}


def load_sqlite_rows(sqlite_conn: sqlite3.Connection, table_name: str) -> list:
    cursor = sqlite_conn.cursor()
    columns = ", ".join(_TABLE_COLUMNS[table_name])
    cursor.execute(f"SELECT {columns} FROM {table_name}")
    rows = cursor.fetchall()
    cursor.close()
    return rows


def ensure_postgres_schema(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(_SCHEMA_DETECTIONS)
        cursor.execute(_SCHEMA_LINE_CROSSINGS)
    conn.commit()


def migrate_table(sqlite_conn: sqlite3.Connection,
                  postgres_conn,
                  table_name: str) -> int:
    rows = load_sqlite_rows(sqlite_conn, table_name)
    if not rows:
        print(f"No rows to migrate for {table_name}.")
        return 0

    if table_name == "detections":
        rows = [tuple(row[:-1] + (bool(row[-1]),)) for row in rows]

    columns = ", ".join(_TABLE_COLUMNS[table_name])
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES %s"

    with postgres_conn.cursor() as cursor:
        execute_values(cursor, insert_query, rows)
    postgres_conn.commit()
    return len(rows)


def main() -> int:
    sqlite_db = os.getenv("SQLITE_DB", "trafficSight_traffic.db")
    if not os.path.exists(sqlite_db):
        raise SystemExit(f"SQLite file not found: {sqlite_db}")

    print(f"Migrating from SQLite: {sqlite_db}")
    print(f"Target PostgreSQL:     {cfg.database_url}")

    try:
        postgres_conn = psycopg2.connect(cfg.database_url)
    except psycopg2.OperationalError as exc:
        raise SystemExit(
            "Failed connecting to PostgreSQL:\n"
            f"  {exc}\n"
            "Please verify the PostgreSQL credentials and connection string.\n"
            "You can set it with:\n"
            "  export DATABASE_URL='postgresql://user:password@localhost:5432/trafficsight'\n"
        ) from exc

    sqlite_conn = sqlite3.connect(sqlite_db)
    try:
        ensure_postgres_schema(postgres_conn)
        total = 0
        for table_name in _TABLE_COLUMNS:
            count = migrate_table(sqlite_conn, postgres_conn, table_name)
            print(f"Migrated {count} rows into {table_name}.")
            total += count
        print(f"Migration complete. Total rows migrated: {total}")
    finally:
        sqlite_conn.close()
        postgres_conn.close()
    write_log(f"SQLite → PostgreSQL migration complete ({total} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
