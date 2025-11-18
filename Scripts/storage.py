"""Snowflake helpers for MorningNews ingestion pipeline."""
import snowflake.connector
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Sequence, Mapping

SNOWFLAKE_CFG = {
    "user": "JELLYFISH",
    "password": "z35y3FjgSdZR2dN",
    "account": "vfb48312",
    "warehouse": "COMPUTE_WH",
    "database": "MORNING_NEWS",
    "schema": "PUBLIC",
}


@contextmanager
def connect():
    """Context manager that yields a Snowflake connection."""
    conn = snowflake.connector.connect(**SNOWFLAKE_CFG)
    try:
        yield conn
    finally:
        conn.close()


def insert_articles(rows: Sequence[Sequence]) -> int:
    """Bulk insert normalized article tuples, ignoring duplicates."""
    if not rows:
        return 0

    sql = """
    INSERT INTO articles (
        id, title, description, author, source, published_at,
        url, content, keywords
    )
    SELECT column1, column2, column3, column4, column5,
           column6, column7, column8, PARSE_JSON(column9)
    FROM VALUES
    """

    value_blocks = []
    for r in rows:
        quoted = ",".join("%s" for _ in r)
        value_blocks.append(f"({quoted})")

    sql += ", ".join(value_blocks)
    sql += " ON ERROR = 'SKIP_FILE'"

    flat = [item for row in rows for item in row]

    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, flat)
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
):
    sql = """
    INSERT INTO ingestion_log (
        run_id, topic, fetched, inserted, duplicates, retries,
        rate_limit, started_at, finished_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, %s)
    """

    payload = json.dumps(rate_limit_snapshot or {})

    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            sql,
            (
                str(uuid.uuid4()),
                topic,
                fetched,
                inserted,
                duplicates,
                retries,
                payload,
                started_at.isoformat(),
                finished_at.isoformat(),
            ),
        )


def latest_articles(limit: int = 20):
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT title, source, published_at, url, keywords
            FROM articles
            ORDER BY published_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cursor.fetchall()
