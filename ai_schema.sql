-- ======================================================
-- AI TAGGING & SUMMARIZATION TABLES FOR MorningNews
-- ======================================================

-- Stores per-article AI-generated metadata
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
