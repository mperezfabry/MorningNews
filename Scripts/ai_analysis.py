import os
import sys
import json
import time
from dotenv import load_dotenv
from typing import Optional

# --- CONFIGURATION ---

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
   sys.path.append(parent_dir)

# Environment Variables
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)


# AI Setup / toggles
DRY_RUN = os.getenv("AI_ANALYSIS_DRY_RUN", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_ANALYSIS_MODEL = os.getenv("AI_ANALYSIS_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY and not DRY_RUN:
    print("❌ Error: OPENAI_API_KEY not found.")
    sys.exit(1)

from openai import OpenAI, RateLimitError
from storage import connect as storage_connect, ensure_db_exists, get_db_path

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

MAX_RETRIES = int(os.getenv("AI_ANALYSIS_MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("AI_ANALYSIS_BACKOFF_BASE", "1.5"))
BATCH_SIZE = int(os.getenv("AI_ANALYSIS_BATCH_SIZE", "200"))
RATE_LIMIT_SLEEP = float(os.getenv("AI_ANALYSIS_RATE_LIMIT_SLEEP", "20"))


def fetch_unprocessed_articles(limit=5):
   """Retrieves a batch of articles that haven't been analyzed yet."""
   with storage_connect() as conn:
       cursor = conn.cursor()

       cursor.execute("""
           SELECT id, title, description, content, source
           FROM articles
           WHERE sentiment_score IS NULL
           LIMIT ?
       """, (limit,))

       rows = cursor.fetchall()
       return [dict(row) for row in rows]




def _generate_with_retry(prompt: str):
   last_exc = None
   if client is None:
       raise RuntimeError("OpenAI client unavailable (missing API key).")
   for attempt in range(1, MAX_RETRIES + 1):
       try:
           completion = client.chat.completions.create(
               model=AI_ANALYSIS_MODEL,
               messages=[{"role": "user", "content": prompt}],
               response_format={"type": "json_object"},
               max_tokens=300,
               temperature=0.2,
           )
           return completion.choices[0].message.content
       except Exception as exc:  # pragma: no cover - network dependent
           last_exc = exc
           sleep_for = BACKOFF_BASE ** (attempt - 1)
           # If API returns a suggested retry delay, honor it
           import re as _re
           match = _re.search(r"retry in ([0-9\\.]+)s", str(exc))
           if match:
               sleep_for = max(sleep_for, float(match.group(1)))
           else:
               # default guardrail for 429s on free tier
               if "429" in str(exc):
                   sleep_for = max(sleep_for, RATE_LIMIT_SLEEP)
           print(f"[ai_analysis] Gemini call failed (attempt {attempt}/{MAX_RETRIES}): {exc}; retrying in {sleep_for:.1f}s")
           time.sleep(sleep_for)
   print("[ai_analysis] Gemini call failed after retries.")
   if last_exc:
       raise last_exc
   raise RuntimeError("Gemini call failed without exception")


def analyze_article_with_llm(article):
    """Sends article data to OpenAI and requests a structured JSON response."""
    # Safety check for empty content
    desc = article.get("description") or ""
    if len(desc) < 5 and not article.get("content"):
        return None  # Skip articles with no info to analyze

    if DRY_RUN:
        return {
            "sentiment_score": 0.5,
            "bias_score": 0.0,
            "is_clickbait": False,
            "ai_summary": f"[DRY RUN] {article.get('title','')}",
        }

    prompt = f"""
    Analyze this news article.
    Title: {article['title']}
    Source: {article['source']}
    Description: {desc}


    Return valid JSON (NO markdown formatting) with:
    - "sentiment_score": float 0.0 (Negative) to 1.0 (Positive).
    - "bias_score": float -1.0 (Left) to 1.0 (Right).
    - "is_clickbait": boolean.
    - "ai_summary": 2 sentence summary.
   """

    try:
        text_response = _generate_with_retry(prompt)
        if text_response.startswith("```"):
            text_response = text_response.split("```")[1]
            if text_response.startswith("json"):
                text_response = text_response[4:]
        data = json.loads(text_response.strip())
        data["sentiment_score"] = float(data.get("sentiment_score", 0.5))
        data["bias_score"] = float(data.get("bias_score", 0.0))
        data["is_clickbait"] = bool(data.get("is_clickbait", False))
        data["ai_summary"] = str(data.get("ai_summary", article.get("description") or ""))[:1000]
        return data
    except Exception as e:
        print(f"   ! Analysis Failed: {e}")
        return None




def update_article_in_db(article_id, analysis):
    """Saves the AI analysis results back to the database."""
    with storage_connect() as conn:
        conn.execute(
            """
            UPDATE articles
            SET sentiment_score = ?, bias_score = ?, is_clickbait = ?, ai_summary = ?
            WHERE id = ?
            """,
            (
                analysis.get("sentiment_score"),
                analysis.get("bias_score"),
                analysis.get("is_clickbait"),
                analysis.get("ai_summary"),
                article_id,
            ),
        )
        conn.commit()




def main():
   print("--- Starting AI Analysis Pipeline ---")

   ensure_db_exists()

   articles = fetch_unprocessed_articles(limit=BATCH_SIZE)
   print(f"-> Found {len(articles)} articles needing analysis.")


   if not articles:
       print("   No pending articles. Exiting.")
       return


   for i, article in enumerate(articles):
       print(f"[{i + 1}/{len(articles)}] Analyzing: {article['title'][:50]}...")
       analysis = analyze_article_with_llm(article)


       if analysis:
           update_article_in_db(article['id'], analysis)
           print("   ✓ Saved analysis.")
       else:
           print("   x Skipped (Error or empty content)")


       time.sleep(1)


   print("--- Analysis Complete ---")




if __name__ == "__main__":
   main()
