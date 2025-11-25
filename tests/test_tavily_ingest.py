import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "Scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tavily_client import TavilyError
from tavily_ingest import (
    fetch_tavily_articles,
    normalize_articles,
    parse_published_at,
    persist_ingestion,
    tavily_results_to_articles,
)


class TavilyIngestTests(unittest.TestCase):
    def test_normalize_articles_builds_expected_tuple(self):
        articles = [
            {
                "title": "Example",
                "description": "Desc",
                "author": "Author",
                "source": {"name": "Src"},
                "publishedAt": "2025-01-01T00:00:00Z",
                "url": "https://example.com/story",
                "content": "Full text",
            }
        ]
        rows = normalize_articles(articles, provider="Tavily", topic="Topic")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row[1], "Example")
        self.assertEqual(row[4], "Src")
        self.assertEqual(row[5], "Tavily")
        self.assertEqual(row[6], "Topic")
        self.assertEqual(row[8], "https://example.com/story")

    def test_tavily_results_to_articles_maps_payload(self):
        results = [
            {
                "title": "Mapped",
                "content": "Body",
                "author": "Reporter",
                "source": "TavilyWire",
                "published_date": "2025-01-02T00:00:00Z",
                "url": "https://example.com",
            }
        ]
        mapped = tavily_results_to_articles(results)
        self.assertEqual(len(mapped), 1)
        article = mapped[0]
        self.assertEqual(article["source"]["name"], "TavilyWire")
        self.assertEqual(article["url"], "https://example.com")

    def test_parse_published_at_handles_invalid(self):
        result = parse_published_at("not-a-date")
        self.assertTrue(result.endswith("+00:00"))

    def test_fetch_tavily_articles_handles_error(self):
        mock_client = MagicMock()
        mock_client.search.side_effect = TavilyError("fail")
        articles, meta = fetch_tavily_articles("Topic", {"q": "test"}, mock_client)
        self.assertEqual(articles, [])
        self.assertEqual(meta, {})

    def test_persist_ingestion_inserts_rows(self):
        mock_insert = MagicMock(return_value=1)
        mock_log = MagicMock()
        from datetime import datetime, timezone

        with patch("tavily_ingest.insert_articles", mock_insert), patch(
            "tavily_ingest.log_ingestion", mock_log
        ):
            persist_ingestion(
                topic="Topic",
                articles=[
                    {
                        "title": "t",
                        "publishedAt": "2025-01-01T00:00:00Z",
                        "url": "https://example.com",
                        "source": {"name": "Src"},
                    }
                ],
                meta={"provider": "Tavily"},
                started_at=datetime.now(timezone.utc),
            )
        mock_insert.assert_called_once()
        mock_log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
