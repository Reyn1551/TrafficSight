#!/usr/bin/env python3
"""Migrate legacy SQLite data to PostgreSQL for TrafficSight."""

import os
import sqlite3
import sys
from config import DATABASE_URL

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError as exc:
    raise SystemExit(
        "psycopg2 is required to run this migration. Install it with `pip install psycopg2-binary`."
    ) from exc

SQLITE_DB = os.getenv("SQLITE_DB", "trafficSight_traffic.db")
POSTGRES_URL = os.getenv("DATABASE_URL", os.getenv("POSTGRES_URL", DATABASE_URL))
TABLES = {
    "detections": [
        "timestamp",
        "camera",
        "track_id",
        "class_name",
        "speed_kmh",
        "cx",
        "cy",
        "direction",
        "is_overspeed",
    ],
    "line_crossings": [
        "timestamp",
        "camera",
        "track_id",
        "class_name",
        "speed_kmh",
        "direction",
    ],
}


def load_sqlite_rows(sqlite_conn, table_name):
    cursor = sqlite_conn.cursor()
    cursor.execute(f"SELECT {', '.join(TABLES[table_name])} FROM {table_name}")
    rows = cursor.fetchall()
    cursor.close()
    return rows


def ensure_postgres_schema(conn):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP(3) NOT NULL,
                camera TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                speed_kmh REAL NOT NULL,
                cx INTEGER,
                cy INTEGER,
                direction TEXT,
                is_overspeed BOOLEAN DEFAULT FALSE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS line_crossings (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP(3) NOT NULL,
                camera TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                speed_kmh REAL NOT NULL,
                direction TEXT
            )
            """
        )
    conn.commit()


def migrate_table(sqlite_conn, postgres_conn, table_name):
    rows = load_sqlite_rows(sqlite_conn, table_name)
    if not rows:
        print(f"No rows to migrate for {table_name}.")
        return 0

    if table_name == "detections":
        rows = [
            tuple(row[:-1] + (bool(row[-1]),))
            for row in rows
        ]

    columns = ", ".join(TABLES[table_name])
    insert_query = (
        f"INSERT INTO {table_name} ({columns}) VALUES %s"
    )

    with postgres_conn.cursor() as cursor:
        execute_values(cursor, insert_query, rows)
    postgres_conn.commit()
    return len(rows)


def main():
    if not os.path.exists(SQLITE_DB):
        raise SystemExit(f"SQLite file not found: {SQLITE_DB}")

    print(f"Migrating from SQLite: {SQLITE_DB}")
    print(f"Target PostgreSQL: {POSTGRES_URL}")

    try:
        postgres_conn = psycopg2.connect(POSTGRES_URL)
    except psycopg2.OperationalError as exc:
        raise SystemExit(
            "Failed connecting to PostgreSQL:\n"
            f"  {exc}\n"
            "Please verify the PostgreSQL credentials and connection string.\n"
            "You can set it with:\n"
            "  export DATABASE_URL='postgresql://user:password@localhost:5432/trafficsight'\n"
            "Or use a PGPASSWORD environment variable if password authentication is required."
        ) from exc

    sqlite_conn = sqlite3.connect(SQLITE_DB)

    try:
        ensure_postgres_schema(postgres_conn)

        total = 0
        for table_name in TABLES:
            count = migrate_table(sqlite_conn, postgres_conn, table_name)
            print(f"Migrated {count} rows into {table_name}.")
            total += count

        print(f"Migration complete. Total rows migrated: {total}")
    finally:
        sqlite_conn.close()
        postgres_conn.close()


if __name__ == "__main__":
    main()
