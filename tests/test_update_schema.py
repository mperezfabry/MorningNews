import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "Scripts"))

from Scripts import update_schema


def test_update_db_applies_schema(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "schema.db"
        schema_path = ROOT / "schema.sql"

        monkeypatch.setattr(update_schema, "DB_PATH", str(db_path))
        monkeypatch.setattr(update_schema, "SCHEMA_PATH", str(schema_path))

        update_schema.update_db()

        with sqlite3.connect(db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }

    assert "articles" in tables
    assert "ingestion_log" in tables
