
"""
Unit Tests for ai_agent.py
Run with: pytest -q
"""

import sys
import os
import json
import importlib.util
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_FOLDER = os.path.join(PROJECT_ROOT, "Scripts")

sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPTS_FOLDER)

os.environ.setdefault("AI_AGENT_DRY_RUN", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from datetime import datetime

# Skip if optional dependencies are missing
if importlib.util.find_spec("sentence_transformers") is None:
    pytest.skip("sentence-transformers not installed", allow_module_level=True)

from ai_agent import (
    ArticleTag,
    compute_cluster_ids,
    write_article_tags_to_db,
    fetch_unprocessed_articles,
    run_daily_agent
)

# ----------------------------------------------------------
# FIXTURES
# ----------------------------------------------------------

@pytest.fixture
def temp_db(monkeypatch):
    """
    Creates a temporary SQLite DB mimicking morningnews.db structure.
    """

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")

        # patch DB_PATH to use temp file
        monkeypatch.setenv("MORNINGNEWS_DB_PATH", db_path)

        conn = sqlite3.connect(db_path)
        conn.execute("""
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            author TEXT,
            source TEXT,
            provider TEXT,
            topic TEXT,
            published_at TEXT,
            url TEXT,
            content TEXT,
            keywords TEXT
        );
        """)

        conn.execute("""
        CREATE TABLE article_ai_tags (
            article_id TEXT PRIMARY KEY,
            quality_score REAL,
            reliability_score REAL,
            misinformation_flag INTEGER,
            extreme_bias_flag INTEGER,
            political_bias TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            redundant_flag INTEGER,
            cluster_id TEXT,
            processed_at TEXT
        )
        """)

        conn.close()
        yield db_path


@pytest.fixture
def sample_articles(temp_db):
    conn = sqlite3.connect(temp_db)
    conn.execute("""
    INSERT INTO articles VALUES
    ('A1','Title A','Desc','Author','Source','Provider','business','2025','url1','This is article text A',NULL),
    ('A2','Title B','Desc','Author','Source','Provider','business','2025','url2','This article text is very similar to A',NULL),
    ('A3','Title C','Desc','Author','Source','Provider','business','2025','url3','Completely unrelated topic',NULL);
    """)
    conn.commit()
    conn.close()


# ----------------------------------------------------------
# TESTS
# ----------------------------------------------------------

def test_compute_cluster_ids(monkeypatch, temp_db, sample_articles):
    """
    Test that similar articles A1/A2 cluster, A3 is separate.
    """

    # Mock embeddings: first two are close, third is far
    fake_embed = MagicMock()
    fake_embed.encode.return_value = [
        [1, 0, 0],  # A1
        [0.99, 0.01, 0],  # A2
        [0, 1, 0]  # A3
    ]

    monkeypatch.setattr("ai_agent.embedder", fake_embed)

    conn = sqlite3.connect(temp_db)
    rows = conn.execute("SELECT * FROM articles").fetchall()
    conn.close()

    clusters = compute_cluster_ids(rows)

    assert clusters["A1"] == clusters["A2"]
    assert clusters["A3"] != clusters["A1"]


def test_write_article_tags_to_db(temp_db):
    """
    Ensure tags are written correctly into article_ai_tags.
    """

    tag = ArticleTag(
        quality_score=0.8,
        reliability_score=0.9,
        misinformation_flag=0,
        extreme_bias_flag=0,
        political_bias="center",
        sentiment_label="neutral",
        sentiment_score=0.0,
        redundant_flag=0,
        cluster_id="A1"
    )

    write_article_tags_to_db("A1", tag)

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT * FROM article_ai_tags WHERE article_id='A1'").fetchone()
    conn.close()

    assert row is not None
    assert row[1] == 0.8  # quality_score
    assert row[6] == "neutral"  # sentiment_label


@patch("ai_agent.client")
def test_analyze_and_run_daily_agent(mock_client, temp_db, sample_articles):
    """
    Full pipeline smoke test:
    - Mock OpenAI calls
    - Run daily agent
    - Ensure tags get inserted
    """

    # Mock responses for OpenAI chat completions
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps({
        "quality_score": 0.9,
        "reliability_score": 0.8,
        "misinformation_flag": 0,
        "extreme_bias_flag": 0,
        "political_bias": "center",
        "sentiment_label": "neutral",
        "sentiment_score": 0.0
    })

    mock_client.chat.completions.create.return_value = mock_resp

    # Mock summaries too
    mock_client.chat.completions.create.return_value = mock_resp

    # Mock embeddings simple
    fake_embed = MagicMock()
    fake_embed.encode.return_value = [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ]
    import ai_agent
    ai_agent.embedder = fake_embed

    # Run pipeline
    run_daily_agent()

    # Check tags were created
    conn = sqlite3.connect(temp_db)
    count = conn.execute("SELECT COUNT(*) FROM article_ai_tags").fetchone()[0]
    conn.close()

    assert count == 3
