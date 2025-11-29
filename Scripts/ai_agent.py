"""
AI Agent Layer for MorningNews
--------------------------------
Reads unprocessed articles from SQLite,
runs AI tagging using OpenAI GPT 4o,
performs embedding-based similarity clustering using all-MiniLM-L6-v2,
generates per-category summaries,
and writes tags/results back to the database. 

Ensure to run ai_agent_schema.sql first.
"""

import os
import json
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Optional

from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from storage import connect as storage_connect
from storage import get_db_path

# -------------------------------------------------------------------
# Load environment vars
# -------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# Embedding Model
# -------------------------------------------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------------------------------------------
# Pydantic model for AI output validation
# -------------------------------------------------------------------
class ArticleTag(BaseModel):
    quality_score: float = Field(..., ge=0, le=1)
    reliability_score: float = Field(..., ge=0, le=1)

    misinformation_flag: int = Field(..., ge=0, le=1)
    extreme_bias_flag: int = Field(..., ge=0, le=1)

    political_bias: Optional[str]

    sentiment_label: str
    sentiment_score: float

    redundant_flag: int = 0
    cluster_id: Optional[str] = None


# -------------------------------------------------------------------
# Fetch unprocessed articles
# -------------------------------------------------------------------
def fetch_unprocessed_articles(limit: int = 50) -> List[sqlite3.Row]:
    with storage_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, description, content, source, url
            FROM articles
            WHERE id NOT IN (SELECT article_id FROM article_ai_tags)
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return rows


# -------------------------------------------------------------------
# OpenAI Tagging (GPT-4o-mini)
# -------------------------------------------------------------------
def analyze_article_with_openai(article: sqlite3.Row) -> ArticleTag:
    """
    Return pydantic-validated tag object.
    """

    text = article["content"] or article["description"] or article["title"]
    text = text[:4000]  # cost-saving truncation

    prompt = f"""
You are an AI assistant analyzing a news article.

ARTICLE TITLE:
{article['title']}

ARTICLE TEXT:
{text}

Return ONLY JSON with the following fields:
{{
  "quality_score": float (0–1),
  "reliability_score": float (0–1),

  "misinformation_flag": 0 or 1,
  "extreme_bias_flag": 0 or 1,

  "political_bias": string or null,

  "sentiment_label": "positive" | "neutral" | "negative",
  "sentiment_score": float (-1 to 1)
}}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = completion.choices[0].message.content
    parsed = json.loads(data)
    try:
        return ArticleTag(**parsed)
    except ValidationError as e:
        print("VALIDATION ERROR:", e)
        raise


# -------------------------------------------------------------------
# Embedding-based clustering
# -------------------------------------------------------------------
def compute_cluster_ids(articles: List[sqlite3.Row]) -> Dict[str, str]:
    """
    Returns mapping: article_id -> cluster_id
    cluster_id = article_id of centroid
    """

    if not articles:
        return {}

    texts = [
        (a["content"] or a["description"] or a["title"])[:2000] for a in articles
    ]
    ids = [a["id"] for a in articles]

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    sim = cosine_similarity(embeddings)

    cluster_ids = {}
    visited = set()

    for i, aid in enumerate(ids):
        if i in visited:
            continue

        # find all articles ≥ 0.78 similarity
        group = [j for j in range(len(ids)) if sim[i][j] >= 0.78]

        # mark visited
        for j in group:
            visited.add(j)

        # centroid is the article with highest avg similarity
        centroid_idx = max(group, key=lambda j: sim[j].mean())
        centroid_id = ids[centroid_idx]

        for j in group:
            cluster_ids[ids[j]] = centroid_id

    return cluster_ids


# -------------------------------------------------------------------
# Write tag to DB
# -------------------------------------------------------------------
def write_article_tags_to_db(article_id: str, tag: ArticleTag):
    with storage_connect() as conn:
        conn.execute(
            """
            INSERT INTO article_ai_tags (
                article_id, quality_score, reliability_score,
                misinformation_flag, extreme_bias_flag, political_bias,
                sentiment_label, sentiment_score,
                redundant_flag, cluster_id, processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                tag.quality_score,
                tag.reliability_score,
                tag.misinformation_flag,
                tag.extreme_bias_flag,
                tag.political_bias,
                tag.sentiment_label,
                tag.sentiment_score,
                tag.redundant_flag,
                tag.cluster_id,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


# -------------------------------------------------------------------
# Category Summaries (GPT-4o)
# -------------------------------------------------------------------
def generate_category_summary(category: str, article_texts: List[str]) -> str:
    """
    Take all the article text from a category and produce a daily summary.
    """

    joined_text = "\n\n".join([t[:1500] for t in article_texts])[:8000]

    prompt = f"""
Write a concise, neutral DAILY NEWS SUMMARY for the category "{category}"
covering the key events and stories below.

Be accurate, non-opinionated, and well-structured.

ARTICLES:
{joined_text}
"""

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )

    return completion.choices[0].message.content.strip()


def write_summary_to_db(category: str, summary_text: str):
    with storage_connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_summaries (
                summary_date, category, summary_text, generated_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                date.today().isoformat(),
                category,
                summary_text,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------
def run_daily_agent():
    CATEGORY_KEYWORDS = {
    "business": ["business", "markets", "economy", "finance", "stocks"],
    "technology": ["tech", "technology", "ai", "software", "hardware"],
    "science": ["science", "research", "space", "biology", "physics"],
    "health": ["health", "medicine", "medical", "covid"],
    "sports": ["sports", "sport", "nba", "nfl", "mlb"],
    "politics": ["politics", "government", "policy", "congress"],
    "international": ["world", "international", "global", "geopolitics"],
    }
    
    print("\nLoading unprocessed articles...")
    articles = fetch_unprocessed_articles(limit=200)

    if not articles:
        print("No new articles to process.")
        return

    print(f"Found {len(articles)} new articles.")

    # Step 1: Compute cluster IDs
    print("\nComputing similarity clusters...")
    cluster_map = compute_cluster_ids(articles)

    # Step 2: Tag each article with OpenAI
    print("\nTagging articles with AI...")
    for article in tqdm(articles):
        tag = analyze_article_with_openai(article)
        tag.cluster_id = cluster_map[article["id"]]
        tag.redundant_flag = int(tag.cluster_id != article["id"])
        write_article_tags_to_db(article["id"], tag)

    # Step 3: Generate category summaries
    print("\nGenerating category summaries...\n")

    # You can later expand categories using your ingestion categories
    categories = ["business", "technology", "science", "health", "sports", "politics", "international"]

    with storage_connect() as conn:
        for category in categories:
            keywords = CATEGORY_KEYWORDS.get(category, [])
            rows = conn.execute(
                f"""
                SELECT a.title, a.description, a.content
                FROM articles a
                WHERE {" OR ".join(["lower(a.topic) LIKE ?"] * len(keywords))}
                """,
                tuple([f"%{kw}%" for kw in keywords]),
            ).fetchall()

            if not rows:
                continue

            if not keywords:
                continue

            texts = [
                (r["content"] or r["description"] or r["title"]) for r in rows
            ]
            summary = generate_category_summary(category, texts)
            write_summary_to_db(category, summary)

            print(f"✓ Summary for {category}")

        print("\nDaily agent run complete.\n")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_daily_agent()
