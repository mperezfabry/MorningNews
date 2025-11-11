"""SQLite helpers for MorningNews ingestion pipeline."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DB_PATH = Path(__file__).parent / "data" / "morningnews.db"


def ensure_db_exists() -> None:
    """Raise if the expected database file is missing."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run schema.sql first.")


@contextmanager
def connect():
    """Context manager that yields a SQLite connection with row factory."""
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
