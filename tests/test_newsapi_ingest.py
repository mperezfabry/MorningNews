import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "Scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from NewsAPI import newsapi_to_articles, persist_newsapi_ingestion


NEWSAPI_MISSING = importlib.util.find_spec("newsapi") is None


@unittest.skipIf(NEWSAPI_MISSING, "newsapi-python not installed")
class NewsApiIngestTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "newsapi.db"
        os.environ["MORNINGNEWS_DB_PATH"] = str(self.db_path)
        self.addCleanup(lambda: os.environ.pop("MORNINGNEWS_DB_PATH", None))

    def test_newsapi_to_articles_maps_fields(self):
        raw = [
            {
                "title": "Mapped",
                "description": "Desc",
                "author": "Byline",
                "source": {"name": "SourceName"},
                "publishedAt": "2025-01-02T00:00:00Z",
                "url": "https://example.com",
                "content": "Body",
            }
        ]
        mapped = newsapi_to_articles(raw)
        self.assertEqual(len(mapped), 1)
        article = mapped[0]
        self.assertEqual(article["source"]["name"], "SourceName")
        self.assertEqual(article["url"], "https://example.com")

    def test_persist_newsapi_ingestion_inserts(self):
        articles = [
            {
                "title": "t",
                "description": "d",
                "author": "a",
                "source": {"name": "Src"},
                "publishedAt": "2025-01-02T00:00:00Z",
                "url": "https://example.com/story",
                "content": "Body",
            }
        ]
        started = datetime(2025, 1, 1, tzinfo=timezone.utc)
        persist_newsapi_ingestion(
            topic="Tech",
            articles=articles,
            meta={"provider": "NewsAPI"},
            started_at=started,
        )

        with sqlite3.connect(self.db_path) as conn:
            article_rows = conn.execute("SELECT title, url, provider, topic FROM articles").fetchall()
            log_rows = conn.execute(
                "SELECT topic, fetched, inserted, duplicates FROM ingestion_log"
            ).fetchall()

        self.assertEqual(len(article_rows), 1)
        self.assertEqual(article_rows[0][0], "t")
        self.assertEqual(article_rows[0][1], "https://example.com/story")
        self.assertEqual(article_rows[0][2], "NewsAPI")
        self.assertEqual(article_rows[0][3], "Tech")
        self.assertEqual(len(log_rows), 1)
        self.assertEqual(log_rows[0][0], "Tech")
        self.assertEqual(log_rows[0][1], 1)
        self.assertEqual(log_rows[0][2], 1)
        self.assertEqual(log_rows[0][3], 0)


if __name__ == "__main__":
    unittest.main()
