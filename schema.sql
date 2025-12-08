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
  sentiment_score REAL,
  bias_score   REAL,
  is_clickbait BOOLEAN,
  ai_summary   TEXT,
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

CREATE TABLE IF NOT EXISTS article_ai_tags (
    article_id TEXT PRIMARY KEY REFERENCES articles(id),

    quality_score REAL,
    reliability_score REAL,
    misinformation_flag INTEGER,
    extreme_bias_flag INTEGER,
    political_bias TEXT,

    sentiment_label TEXT,
    sentiment_score REAL,

    redundant_flag INTEGER DEFAULT 0,
    cluster_id TEXT,          -- article_id of cluster centroid

    processed_at TEXT
);

-- Stores one summary per category per day
CREATE TABLE IF NOT EXISTS daily_summaries (
    summary_date TEXT,
    category TEXT,
    summary_text TEXT,
    generated_at TEXT,
    PRIMARY KEY (summary_date, category)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_unique ON articles(url);
