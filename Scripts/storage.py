"""SQLite helpers for MorningNews ingestion pipeline."""

import argparse
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "morningnews.db"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "schema.sql"
_SCHEMA_SQL_CACHE: str | None = None  # memoize schema text to avoid re-reading disk


def get_db_path() -> Path:
    """Return the configured SQLite path (env overrides default)."""
    override = os.getenv("MORNINGNEWS_DB_PATH")
    if override:
        return Path(override).expanduser()
    return DEFAULT_DB_PATH


def ensure_db_exists() -> None:
    """Create the SQLite file and tables if they do not already exist.

    Called before every connection so ingestion can run on a fresh checkout
    without manual schema creation.
    """
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        with sqlite3.connect(db_path) as conn:
            _apply_schema(conn)


@contextmanager
def connect():
    """Context manager that yields a SQLite connection with row factory.

    Ensures schema is present and enforces consistent row access by name.
    """
    ensure_db_exists()
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    _apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def insert_articles(rows: Sequence[Sequence]) -> int:
    """Bulk insert normalized article tuples, ignoring duplicates by primary key.

    Accepts pre-normalized sequences matching schema order; returns count inserted.
    """
    if not rows:
        return 0
    with connect() as conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO articles (
                id, title, description, author, source, provider, topic,
                published_at, url, content, keywords
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            SELECT title, source, provider, topic, published_at, url, keywords
            FROM articles
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def article_count() -> int:
    """Return total article rows."""
    with connect() as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
        return count


def ingestion_runs(limit: int = 10):
    """Return recent ingestion log entries."""
    with connect() as conn:
        cursor = conn.execute(
            """
            SELECT topic, fetched, inserted, duplicates, retries, rate_limit,
                   started_at, finished_at
            FROM ingestion_log
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()


def ingestion_count() -> int:
    """Return number of ingestion runs."""
    with connect() as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM ingestion_log").fetchone()
        return count


def _to_utc(dt: datetime) -> str:
    """Ensure timestamps persisted to SQLite are UTC ISO strings."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _apply_schema(conn: sqlite3.Connection) -> None:
    """Apply schema.sql to guarantee required tables exist."""
    conn.executescript(_load_schema_sql())
    _ensure_article_columns(conn)
    _ensure_indexes(conn)


def _ensure_article_columns(conn: sqlite3.Connection) -> None:
    """Backfill missing article columns (provider/topic) on existing DBs."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    alters = []
    if "provider" not in existing:
        alters.append("ALTER TABLE articles ADD COLUMN provider VARCHAR NOT NULL DEFAULT 'unknown'")
    if "topic" not in existing:
        alters.append("ALTER TABLE articles ADD COLUMN topic VARCHAR")
    for stmt in alters:
        conn.execute(stmt)


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """Add indexes that guard against duplicates."""
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_unique ON articles(url)"
    )


def _load_schema_sql() -> str:
    """Lazy-load the schema SQL to avoid re-reading the file.

    Supports override via MORNINGNEWS_SCHEMA_PATH for migrations/tests.
    """
    global _SCHEMA_SQL_CACHE
    if _SCHEMA_SQL_CACHE is None:
        schema_path = Path(os.getenv("MORNINGNEWS_SCHEMA_PATH", DEFAULT_SCHEMA_PATH))
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found at {schema_path}. Provide MORNINGNEWS_SCHEMA_PATH or add schema.sql."
            )
        _SCHEMA_SQL_CACHE = schema_path.read_text()
    return _SCHEMA_SQL_CACHE


def _print_inspect(limit: int) -> None:
    """Render a small dashboard of storage state."""
    db_path = get_db_path()
    print(f"DB: {db_path}")
    try:
        total_articles = article_count()
        total_runs = ingestion_count()
    except FileNotFoundError:
        print("Database not found; run an ingestion first.")
        return

    print(f"Articles: {total_articles}")
    print(f"Ingestion runs: {total_runs}")

    if total_articles:
        print(f"\nLatest {min(limit, total_articles)} articles:")
        for row in latest_articles(limit):
            published = row["published_at"]
            provider = row["provider"]
            topic = row["topic"] or "-"
            print(
                f"- {row['title']} ({row['source']}, provider={provider}, topic={topic}) "
                f"[{published}] -> {row['url']}"
            )

    if total_runs:
        print(f"\nRecent {min(limit, total_runs)} ingestion runs:")
        for row in ingestion_runs(limit):
            started = row["started_at"]
            finished = row["finished_at"]
            rate_limit = json.loads(row["rate_limit"] or "{}")
            rl_rem = rate_limit.get("rate_limit_remaining") or rate_limit.get("remaining")
            print(
                f"- {row['topic']}: fetched={row['fetched']}, inserted={row['inserted']}, "
                f"dupes={row['duplicates']}, retries={row['retries']}, "
                f"started={started}, finished={finished}, rl_remaining={rl_rem}"
            )


def cli(argv: Iterable[str] | None = None) -> None:
    """Command-line entrypoint for inspecting storage."""
    parser = argparse.ArgumentParser(description="Inspect MorningNews storage.")
    subparsers = parser.add_subparsers(dest="command")

    inspect = subparsers.add_parser("inspect", help="Show latest articles and ingestion runs.")
    inspect.add_argument("--limit", type=int, default=5, help="Items to show for articles/logs.")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "inspect":
        _print_inspect(limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
