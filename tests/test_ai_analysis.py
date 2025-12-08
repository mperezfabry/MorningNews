import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Skip if optional dependency is missing
if importlib.util.find_spec("google.generativeai") is None:
    pytest.skip("google-generativeai not installed", allow_module_level=True)

os.environ.setdefault("GEMINI_API_KEY", "test-key")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "Scripts"))

from ai_analysis import analyze_article_with_llm, fetch_unprocessed_articles, update_article_in_db
from storage import ensure_db_exists, get_db_path


class FakeResponse:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload)


class FakeModel:
    def __init__(self, payload: dict):
        self.payload = payload

    def generate_content(self, prompt: str):  # noqa: ARG002
        return FakeResponse(self.payload)


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "ai_analysis.db"
        monkeypatch.setenv("MORNINGNEWS_DB_PATH", str(db_path))
        ensure_db_exists()
        yield db_path


@pytest.fixture
def seed_article(temp_db):
    with sqlite3.connect(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO articles (
                id, title, description, author, source, provider, topic,
                published_at, url, content, keywords
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "art1",
                "Title",
                "Desc",
                "Author",
                "Src",
                "Provider",
                "Topic",
                "2025-01-01T00:00:00+00:00",
                "https://example.com/1",
                "Body text",
                None,
            ),
        )
        conn.commit()


def test_fetch_and_update_article(monkeypatch, temp_db, seed_article):  # noqa: ARG001
    # Patch the model to avoid real API calls
    from ai_analysis import model as real_model

    fake_payload = {
        "sentiment_score": 0.7,
        "bias_score": 0.1,
        "is_clickbait": False,
        "ai_summary": "Summary here",
    }
    monkeypatch.setattr("ai_analysis.model", FakeModel(fake_payload))

    rows = fetch_unprocessed_articles(limit=5)
    assert len(rows) == 1

    tag = analyze_article_with_llm(rows[0])
    assert tag == fake_payload

    update_article_in_db(rows[0]["id"], tag)

    with sqlite3.connect(get_db_path()) as conn:
        row = conn.execute(
            "SELECT sentiment_score, bias_score, is_clickbait, ai_summary FROM articles WHERE id=?",
            (rows[0]["id"],),
        ).fetchone()

    assert row[0] == 0.7
    assert row[1] == 0.1
    assert row[2] == 0
    assert row[3] == "Summary here"

    # Restore original model to avoid leaking patch
    monkeypatch.setattr("ai_analysis.model", real_model)
