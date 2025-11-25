CREATE TABLE IF NOT EXISTS articles (
  id           VARCHAR PRIMARY KEY,          -- SHA256(url)
  title        VARCHAR NOT NULL,
  description  VARCHAR,
  author       VARCHAR,
  source       VARCHAR NOT NULL,
  provider     VARCHAR NOT NULL DEFAULT 'unknown', -- pipeline/provider (Tavily, NewsAPI, etc.)
  topic        VARCHAR,                      -- topical query/category used during ingest
  published_at TIMESTAMP NOT NULL,
  url          VARCHAR NOT NULL,
  content      VARCHAR,
  keywords     VARCHAR,                       -- CSV of spaCy noun chunks or tf-idf terms
  ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingestion_log (
  run_id        UUID,
  topic         VARCHAR,
  fetched       INTEGER,
  inserted      INTEGER,
  duplicates    INTEGER,
  retries       INTEGER,
  rate_limit    VARCHAR,                      -- JSON of last RL headers
  started_at    TIMESTAMP,
  finished_at   TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_unique ON articles(url);
