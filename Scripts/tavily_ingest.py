"""MorningNews ingestion script that pulls topical articles via Tavily."""

import hashlib
import os
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from storage import insert_articles, log_ingestion
from tavily_client import TavilyClient, TavilyError

load_dotenv()

# Shared defaults for Tavily requests
DEFAULT_PAGE_SIZE = 10
DEFAULT_DEPTH = "advanced"


def demo_queries(tavily: TavilyClient):
    """Run through the topical queries that feed the ingestion pipeline."""
    today = date.today()
    queries = [
        (
            "Global markets and economy",
            {
                "q": "global markets inflation news 2025",
                "from": (today - timedelta(days=3)).isoformat(),
                "include_domains": "wsj.com,ft.com,bloomberg.com",
            },
        ),
        (
            "AI policy and regulation",
            {
                "q": '"artificial intelligence" regulation government policy 2025',
                "from": (today - timedelta(days=7)).isoformat(),
                "include_domains": "nytimes.com,politico.com,theverge.com",
            },
        ),
        (
            "Consumer tech launches",
            {
                "q": "smartphone launch review 2025",
                "from": (today - timedelta(days=14)).isoformat(),
                "include_domains": "theverge.com,engadget.com,cnet.com",
            },
        ),
        (
            "Climate and energy innovation",
            {
                "q": "climate tech energy storage breakthrough 2025",
                "from": (today - timedelta(days=14)).isoformat(),
                "include_domains": "techcrunch.com,grist.org,fastcompany.com",
            },
        ),
        (
            "Geopolitics and security",
            {
                "q": "geopolitics security alliance 2025",
                "from": (today - timedelta(days=7)).isoformat(),
                "include_domains": "reuters.com,bbc.com,aljazeera.com",
            },
        ),
    ]

    for label, params in queries:
        print(f"\n=== Example: {label} ===")
        started_at = datetime.now(timezone.utc)
        articles, meta = fetch_tavily_articles(label, params, tavily)
        print_results(articles)
        persist_ingestion(label, articles, meta, started_at)


def fetch_tavily_articles(label: str, params: Dict, tavily: TavilyClient):
    """Fetch articles via Tavily using topic-specific parameters."""
    include_domains = None
    if params.get("include_domains"):
        include_domains = [item.strip() for item in params["include_domains"].split(",")]

    query = params.get("q") or label
    try:
        results = tavily.search(
            query=query,
            max_results=params.get("pageSize", DEFAULT_PAGE_SIZE),
            include_domains=include_domains,
            depth=DEFAULT_DEPTH,
        )
    except TavilyError as exc:
        print(f"Tavily search failed for {label}: {exc}")
        return [], {}

    articles = tavily_results_to_articles(results)
    return articles, tavily.latest_response_meta()


def persist_ingestion(topic: str, articles: List[Dict], meta: Dict, started_at: datetime):
    """Normalize and store article results plus an ingestion log entry."""
    normalized_rows = normalize_articles(articles)
    inserted = insert_articles(normalized_rows)
    duplicates = max(len(normalized_rows) - inserted, 0)
    log_ingestion(
        topic=topic,
        fetched=len(articles),
        inserted=inserted,
        duplicates=duplicates,
        retries=0,
        rate_limit_snapshot=meta or {"provider": "Tavily"},
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
    )


def normalize_articles(articles: List[Dict]):
    """Convert Tavily article payloads into the DB schema order."""
    normalized: List[Tuple] = []
    for article in articles:
        url = article.get("url")
        if not url:
            continue
        article_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
        normalized.append(
            (
                article_id,
                article.get("title", "").strip(),
                article.get("description"),
                article.get("author"),
                (article.get("source") or {}).get("name", "Tavily"),
                parse_published_at(article.get("publishedAt")),
                url,
                article.get("content"),
                None,
            )
        )
    return normalized


def tavily_results_to_articles(results: List[Dict]):
    """Map Tavily results into our canonical article dictionaries."""
    mapped = []
    for result in results:
        url = result.get("url")
        if not url:
            continue
        published = result.get("published_date") or datetime.now(timezone.utc).isoformat()
        mapped.append(
            {
                "title": result.get("title", "Untitled"),
                "description": result.get("content"),
                "author": result.get("author"),
                "source": {"name": result.get("source", "Tavily")},
                "publishedAt": published,
                "url": url,
                "content": result.get("content"),
            }
        )
    return mapped


def parse_published_at(value: Optional[str]):
    """Return ISO timestamp strings compatible with SQLite."""
    if not value:
        return datetime.now(timezone.utc).isoformat()
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.now(timezone.utc).isoformat()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def format_article(article: Dict):
    """Build a printable summary of an article."""
    source = article.get("source", {}).get("name", "Unknown source")
    title = article.get("title", "Untitled")
    published = article.get("publishedAt", "Unknown date")
    url = article.get("url", "No URL provided")
    return f"Title: {title}\nSource: {source}\nDate: {published}\nURL: {url}\n"


def print_results(articles: Iterable[Dict], limit: int = 5):
    """Print the first few articles returned by a query."""
    subset = list(articles)[:limit]
    print(f"\nFound {len(subset)} articles (showing {len(subset)} of {limit} requested):")
    for article in subset:
        print(format_article(article))


def cli():
    """Entry point for manual exploration from the command line."""
    tavily = TavilyClient()
    demo_queries(tavily)


if __name__ == "__main__":
    cli()
