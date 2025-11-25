import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

from dotenv import load_dotenv
from newsapi import NewsApiClient

from storage import insert_articles, log_ingestion
from tavily_ingest import normalize_articles


#Loads environment variables from a .env file into the environment
load_dotenv()

# Categories we will pull for the dashboard
CATEGORIES = [
    'business',
    'technology',
    'science',
    'health',
    'sports'
]


# Pulls 30 articles per category (can be changed)
PAGE_SIZE = 30

# US headlines only for now. Also can change this aspect. Feel free to adjust this, especially if we decide pull data from outside the US. We probably have to add an additional function(s) that can translate other languages to english.
COUNTRY = 'us'


# Initializes the NewsAPI client
def get_api_client():
    api_key = os.getenv('NEWS_API_KEY')
    print(api_key)                      # If this line is still here when it gets to the repo, you can remove it. I only have it here to make sure API connect to key in my person .env
    if not api_key:
        raise ValueError('No NewsAPI key found. Set NEWS_API_KEY in .env.')
    return NewsApiClient(api_key=api_key)


# This function is a basic quality filter that maybe one of you can look at and improve it if you see any flaws or just overall issues
# It tosses incomplete articles/ just junk from the API like placeholders in titles, empty titles, empty description, broken or missing URLs
def clean_articles(articles):
    clean_list=[]
    for article in articles:
        title = article.get('title')
        description = article.get('description')
        url = article.get('url')
        if not title or title.lower() == '[removed]':       # Skips if title missing or placeholder is present
            continue

        # URL to link sources
        if not url:
            continue

        # Description for context
        if not description:
            continue


        clean_list.append(article)
    return clean_list


# This function will grab headlines for one category filter them and save a list as json
#
def fetch_and_save_category(
    client: NewsApiClient,
    category: str,
    *,
    save_json: bool = True,
    persist: bool = True,
    page_size: int = None,
    country: str = None,
):
    print(f"Start fetch for {category.upper()}...")
    started_at = datetime.now(timezone.utc)

    try:
        # Use client's top_headlines method.
        response = client.get_top_headlines(
            category=category,
            country=country or COUNTRY,
            page_size=page_size or PAGE_SIZE,
        )

    except Exception as e:
        print(f"Error fetching articles for {category}: {e}")
        return

    # This if checks the API success status
    if response.get('status') != 'ok':
        print(f"API returned ERROR!!! Error fetching articles for {category}: {response.get('code','Unknown ERROR')}")
        return


    fetched_articles =response.get('articles', [])
    print(f" Pulled/Fetched {len(fetched_articles)} articles")

    # SIMPLE cleaning filter on the fetched articles
    final_articles = clean_articles(fetched_articles)
    print(f" Kept {len(final_articles)} articles after quality check and filtering out of articles that does not meet criteria")

    if not final_articles:
        return

    if save_json:
        _save_articles_json(final_articles, category)

    if persist:
        persist_newsapi_ingestion(
            topic=category,
            articles=final_articles,
            meta={"provider": "NewsAPI", "category": category},
            started_at=started_at,
        )


# Function first gets client object
# then interates through each categories list and processes each one
def main(categories: Sequence[str] = None, *, save_json: bool = True, persist: bool = True):
    categories = list(categories or CATEGORIES)
    try:
        news_client = get_api_client()
        for category in categories:
            fetch_and_save_category(
                news_client,
                category,
                save_json=save_json,
                persist=persist,
            )
        print("\n ALL DONE! All ingestion tasks successfully finsihed \n ")

    except ValueError as e:
        print(f"\n Fatal configuration error: {e}")
    except Exception as e:
        print(f"\n Some unexpected error occurred in the main function execution: {e}")


def newsapi_to_articles(articles: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map NewsAPI article payloads to canonical article dictionaries."""
    mapped: List[Dict[str, Any]] = []
    for article in articles:
        url = article.get("url")
        if not url:
            continue
        mapped.append(
            {
                "title": article.get("title", "Untitled"),
                "description": article.get("description"),
                "author": article.get("author"),
                "source": {"name": (article.get("source") or {}).get("name", "NewsAPI")},
                "publishedAt": article.get("publishedAt"),
                "url": url,
                "content": article.get("content"),
            }
        )
    return mapped


def persist_newsapi_ingestion(
    topic: str,
    articles: List[Dict[str, Any]],
    meta: Dict[str, Any],
    started_at: datetime,
) -> None:
    """Normalize NewsAPI articles and persist both rows and ingestion log."""
    normalized_rows = normalize_articles(newsapi_to_articles(articles), provider="NewsAPI", topic=topic)
    inserted = insert_articles(normalized_rows)
    duplicates = max(len(normalized_rows) - inserted, 0)
    log_ingestion(
        topic=topic,
        fetched=len(articles),
        inserted=inserted,
        duplicates=duplicates,
        retries=0,
        rate_limit_snapshot=meta or {"provider": "NewsAPI"},
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
    )


def _save_articles_json(articles: List[Dict[str, Any]], category: str) -> None:
    """Persist cleaned articles to disk for audit or re-use."""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{category.lower()}_headlines.json"
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"SAVED {len(articles)} articles to {filepath}")


def cli(argv: Iterable[str] | None = None) -> None:
    """CLI wrapper for NewsAPI ingestion."""
    parser = argparse.ArgumentParser(description="Ingest NewsAPI articles into MorningNews DB.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
        help="NewsAPI categories to fetch.",
    )
    parser.add_argument("--no-json", action="store_true", help="Skip writing JSON snapshot files.")
    parser.add_argument("--no-db", action="store_true", help="Skip persisting to the database.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    main(categories=args.categories, save_json=not args.no_json, persist=not args.no_db)


if __name__ == "__main__":
    cli()
