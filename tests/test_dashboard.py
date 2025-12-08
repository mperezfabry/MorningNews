import importlib.util
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Skip if streamlit is unavailable in the environment
if importlib.util.find_spec("streamlit") is None:
    pytest.skip("streamlit not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "Scripts"))

from storage import ensure_db_exists, get_db_path


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "dashboard.db"
        monkeypatch.setenv("MORNINGNEWS_DB_PATH", str(db_path))
        ensure_db_exists()
        yield db_path


def test_load_data_reads_articles(monkeypatch, temp_db):
    import dashboard

    with sqlite3.connect(get_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO articles (
                id, title, description, author, source, provider, topic,
                published_at, url, content, keywords,
                sentiment_score, bias_score, is_clickbait, ai_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "a1",
                "Title",
                "Desc",
                "Author",
                "Source",
                "Provider",
                "Topic",
                "2025-01-01T00:00:00+00:00",
                "https://example.com",
                "Body",
                None,
                0.5,
                0.0,
                False,
                "AI summary",
            ),
        )
        conn.commit()

    # Force cache miss
    if hasattr(dashboard.load_data, "clear"):
        dashboard.load_data.clear()

    df = dashboard.load_data()
    assert not df.empty
    assert "ai_summary" in df.columns
    assert df["title"].iloc[0] == "Title"
