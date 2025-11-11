MorningNews ingestion will orchestrate three layers:

1. Collection
   - Schedule fetch jobs (apscheduler or cron) that instantiate `TavilyClient`.
   - For each topic (global markets, AI policy, consumer tech, climate & energy, geopolitics) trigger `tavily_client.search` with curated, news-focused queries (e.g., include "news", add domain whitelists like WSJ/FT/Bloomberg, and use `depth="advanced"`).
   - Capture Tavily response metadata (status, latency, rate-limit hints) after every call to decide whether to pause or switch sources.
   - Persist the raw JSON payloads for audit (e.g., in `data/raw/YYYY-MM-DD/<topic>.json`).

2. Normalization
   - Flatten each article into the canonical schema: id (URL hash), title, description, author, source, published_at, url, content.
   - Enrich with lightweight NLP (spaCy keywords) before handing off to the tagging pipeline.
   - Store canonical rows in SQLite/DuckDB with a unique index on article id to prevent duplicates. Use `storage.insert_articles` so the same logic runs locally and in production.

3. Downstream hooks
   - Emit normalized records to the LLM tagging workers (Celery queue or async tasks).
   - Update the Streamlit cache via a shared repository layer so the dashboard can refresh without re-hitting Tavily.
   - Log ingestion outcomes (count fetched, duplicates skipped, retry stats) to a daily report for observability.

Tavily integration
------------------
- Configure `TAVILY_API_KEY` in `.env` and rely on `tavily_client.TavilyClient` for queries.
- Feed every Tavily response through `normalize_articles` before saving to SQLite.
- In tests, mock `requests.post` to simulate Tavily payloads (no network dependency).
