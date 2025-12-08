import json
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

from storage import get_db_path, ensure_db_exists


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "json_ingest.db"
        monkeypatch.setenv("MORNINGNEWS_DB_PATH", str(db_path))
        ensure_db_exists()
        yield db_path


def test_process_file_inserts_and_logs(monkeypatch, temp_db):
    import json_ingest  # Imported after env setup

    tmp_data = Path(tempfile.mkdtemp())
    sample_file = tmp_data / "20250101_business_headlines.json"
    sample = [
        {
            "title": "Sample",
            "description": "Desc",
            "author": "Author",
            "source_name": "Src",
            "publishedAt": "2025-01-01T00:00:00Z",
            "url": "https://example.com/story",
            "content": "Body",
        }
    ]
    sample_file.write_text(json.dumps(sample))

    monkeypatch.setattr(json_ingest, "DATA_DIR", str(tmp_data))
    monkeypatch.setattr(json_ingest, "FILE_PATTERN", str(tmp_data / "*_headlines.json"))

    json_ingest.process_file(str(sample_file))

    with sqlite3.connect(get_db_path()) as conn:
        article_rows = conn.execute("SELECT title, url FROM articles").fetchall()
        log_rows = conn.execute("SELECT topic, fetched, inserted FROM ingestion_log").fetchall()

    assert len(article_rows) == 1
    assert article_rows[0][0] == "Sample"
    assert len(log_rows) == 1
    assert log_rows[0][0].startswith("NewsAPI")
