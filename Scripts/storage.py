"""SQLite helpers for MorningNews ingestion pipeline."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

# Path to the SQLite DB file: <project>/Scripts/data/morningnews.db
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "morningnews.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"


def ensure_db_exists() -> None:
    """
    Ensure the SQLite database file and schema exist.

    - Creates the data directory if needed.
    - If the DB file does not yet exist, it is created.
    - The schema defined in schema.sql is executed (idempotent).
    """
    # Make sure the directory for the DB exists
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # If DB already exists, we still run schema.sql because it uses
    # CREATE TABLE IF NOT EXISTS, so it's safe and keeps schema in sync.
    if not SCHEMA_PATH.exists():
        # If you want stricter behavior, you could raise here instead.
        raise FileNotFoundError(
            f"Database schema file not found at {SCHEMA_PATH}. "
            f"Please ensure schema.sql is present."
        )

    # Connect (this will create the file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    try:
        with SCHEMA_PATH.open("r", encoding="utf-8") as f:
            schema_sql = f.read()
        # Execute the schema in a single script call; it's idempotent due to
        # 'CREATE TABLE IF NOT EXISTS' in schema.sql
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect():
    """Context manager that yields a SQLite connection with row factory."""
    # Ensure DB and tables exist before providing a connection
    ensure_db_exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_articles(rows: Sequence[Sequence]) -> int:
    """Bulk insert normalized article tuples, ignoring duplicates by primary key."""
    if not rows:
        return 0
    with connect() as conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO articles (
                id, title, description, author, source, published_at,
                url, content, keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return cursor.rowcount


def log_ingestion(
    topic: str,
    fetched: int,
    inserted: int,
    duplicates: int,
    retries: int,
    rate_limit_snapshot: Mapping,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Record metadata about each ingestion run for observability."""
    payload = json.dumps(rate_limit_snapshot or {})
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_log (
                run_id, topic, fetched, inserted, duplicates, retries,
                rate_limit, started_at, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                topic,
                fetched,
                inserted,
                duplicates,
                retries,
                payload,
                _to_utc(started_at),
                _to_utc(finished_at),
            ),
        )
        conn.commit()


def latest_articles(limit: int = 20):
    """Return the most recently published articles for dashboard previews."""
    with connect() as conn:
        cursor = conn.execute(
            """
            SELECT title, source, published_at, url, keywords
            FROM articles
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def _to_utc(dt: datetime) -> str:
    """Ensure timestamps persisted to SQLite are UTC ISO strings."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()