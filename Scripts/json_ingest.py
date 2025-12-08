import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- PATH SETUP (allow running from anywhere) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import insert_articles, log_ingestion  # noqa: E402

# --- CONFIGURATION ---
# Ingest JSON snapshots from repo-level data folder
DATA_DIR = ROOT / "data"
FILE_PATTERN = str(DATA_DIR / "*_headlines.json")


# --- HELPER FUNCTIONS ---

def parse_published_at(value):
    """Return ISO timestamp strings compatible with SQLite."""
    if not value:
        return datetime.now(timezone.utc).isoformat()

    # Standardize timezones for python's fromisoformat
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.now(timezone.utc).isoformat()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def normalize_article(article):
    """
    Transforms a single article dictionary into a tuple for the DB.
    Handles the source extraction logic carefully.
    """
    url = article.get("url")
    if not url:
        return None

    # Create a unique ID from the URL
    article_id = hashlib.sha256(url.encode()).hexdigest()

    # Parse date
    published_at_iso = parse_published_at(article.get("publishedAt"))

    # --- SOURCE EXTRACTION LOGIC ---
    # 1. Try 'source_name' (added by our custom NewsAPI.py)
    source_name = article.get('source_name')

    # 2. If missing, try standard NewsAPI dict: {'id': 'cnn', 'name': 'CNN'}
    if not source_name:
        source_obj = article.get('source')
        if isinstance(source_obj, dict):
            source_name = source_obj.get('name')
        elif isinstance(source_obj, str):
            source_name = source_obj

    # 3. Fallback
    if not source_name:
        source_name = "Unknown Source"
    # -------------------------------

    keywords = ""

    # Tuple matching table schema order (storage.insert_articles expects 11 fields)
    return (
        article_id,
        article.get("title"),
        article.get("description"),
        article.get("author"),
        source_name,
        "json_import",        # provider
        None,                 # topic (optional)
        published_at_iso,
        url,
        article.get("content"),
        keywords,
    )


def process_file(filepath):
    """Reads a JSON file and inserts articles into DB."""
    start_time = datetime.now(timezone.utc)
    filename = os.path.basename(filepath)
    print(f"-> Processing file: {filename}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except Exception as e:
        print(f"!!! Error reading {filename}: {e}")
        return

    if not articles:
        print("   File empty.")
        return

    normalized_rows = []
    for article in articles:
        row = normalize_article(article)
        if row:
            normalized_rows.append(row)

    if not normalized_rows:
        return

    # Insert into DB
    inserted_count = insert_articles(normalized_rows)
    duplicates_count = len(normalized_rows) - inserted_count

    # Log ingestion
    try:
        topic = filename.split('_')[1]  # simplistic topic extraction from filename
    except:
        topic = "json_import"

    log_ingestion(
        topic=f"NewsAPI_{topic}",
        fetched=len(articles),
        inserted=inserted_count,
        duplicates=duplicates_count,
        retries=0,
        rate_limit_snapshot={},
        started_at=start_time,
        finished_at=datetime.now(timezone.utc),
    )

    print(f"   [DB] Inserted {inserted_count} new articles. Skipped {duplicates_count} duplicates.")


# --- MAIN EXECUTION ---

def run_ingestion():
    print("\n--- Starting JSON Database Ingestion ---")

    # Check if data dir exists
    if not os.path.exists(DATA_DIR):
        print(f"No data directory found at {DATA_DIR}. Run NewsAPI.py first.")
        return

    files = glob.glob(FILE_PATTERN)

    if not files:
        print(f"No JSON files found in {DATA_DIR} to ingest.")
        return

    total_inserted = 0
    for filepath in files:
        process_file(filepath)

    print("--- Ingestion Complete ---")


if __name__ == "__main__":
    run_ingestion()
