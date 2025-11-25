import importlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "Scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "morningnews.db"
        os.environ["MORNINGNEWS_DB_PATH"] = str(self.db_path)
        self.addCleanup(lambda: os.environ.pop("MORNINGNEWS_DB_PATH", None))

        import storage

        self.storage = importlib.reload(storage)

    def test_connect_bootstraps_schema(self):
        self.storage.ensure_db_exists()
        self.assertTrue(self.db_path.exists())
        self.assertEqual(self.storage.get_db_path(), self.db_path)

        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }

        self.assertIn("articles", tables)
        self.assertIn("ingestion_log", tables)

    def test_insert_and_latest_articles(self):
        self.storage.ensure_db_exists()
        rows = [
            (
                "a1",
                "Old",
                "desc",
                None,
                "Src",
                "Provider",
                "TopicOld",
                "2024-01-01T00:00:00+00:00",
                "https://example.com/old",
                None,
                None,
            ),
            (
                "a2",
                "Recent",
                "desc",
                None,
                "Src",
                "Provider",
                "TopicRecent",
                "2024-02-01T00:00:00+00:00",
                "https://example.com/recent",
                None,
                None,
            ),
        ]

        inserted = self.storage.insert_articles(rows)
        self.assertEqual(inserted, 2)
        latest = self.storage.latest_articles(limit=5)
        self.assertEqual(len(latest), 2)
        self.assertEqual(latest[0]["title"], "Recent")
        self.assertEqual(latest[1]["url"], "https://example.com/old")

        # Duplicate insert should be ignored (URL uniqueness)
        dup_inserted = self.storage.insert_articles(rows[:1])
        self.assertEqual(dup_inserted, 0)

    def test_log_ingestion_records_metadata(self):
        self.storage.ensure_db_exists()
        started = datetime(2025, 1, 1, tzinfo=timezone.utc)
        finished = datetime(2025, 1, 2, tzinfo=timezone.utc)
        self.storage.log_ingestion(
            topic="Topic",
            fetched=3,
            inserted=2,
            duplicates=1,
            retries=0,
            rate_limit_snapshot={"remaining": "42"},
            started_at=started,
            finished_at=finished,
        )

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT topic, fetched, inserted, duplicates, rate_limit FROM ingestion_log"
            ).fetchone()

        self.assertEqual(row[0], "Topic")
        self.assertEqual(row[1], 3)
        self.assertEqual(row[2], 2)
        self.assertEqual(row[3], 1)
        self.assertEqual(json.loads(row[4])["remaining"], "42")

    def test_counts_and_cli_inspect(self):
        self.storage.ensure_db_exists()
        # Seed articles
        rows = [
            (
                "a1",
                "One",
                None,
                None,
                "Src",
                "Provider",
                "Topic",
                "2024-01-01T00:00:00+00:00",
                "https://example.com/one",
                None,
                None,
            ),
        ]
        self.storage.insert_articles(rows)

        # Seed ingestion log
        self.storage.log_ingestion(
            topic="Demo",
            fetched=1,
            inserted=1,
            duplicates=0,
            retries=0,
            rate_limit_snapshot={},
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(self.storage.article_count(), 1)
        self.assertEqual(self.storage.ingestion_count(), 1)

        # Run CLI inspect and confirm it emits expected text
        buf = StringIO()
        with redirect_stdout(buf):
            self.storage.cli(["inspect", "--limit", "1"])
        output = buf.getvalue()
        self.assertIn("Articles: 1", output)
        self.assertIn("Ingestion runs: 1", output)


if __name__ == "__main__":
    unittest.main()
