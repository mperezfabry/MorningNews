import os
import sys
import json
import time
import sqlite3
from dotenv import load_dotenv


# --- CONFIGURATION ---


# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"-> Current Directory: {current_dir}")
parent_dir = os.path.dirname(current_dir)
print(f"-> Parent Directory: {parent_dir}")


if parent_dir not in sys.path:
   sys.path.append(parent_dir)


# Environment Variables
env_path = os.path.join(current_dir, '.env')
print(f"-> Loads environment variables from: {env_path}")
load_dotenv(env_path)


# Gemini API Setup
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
   print(f"❌ Error: GEMINI_API_KEY not found.")
   print(f"   Looking for .env at: {env_path}")
   sys.exit(1)


import google.generativeai as genai


genai.configure(api_key=api_key)


# --- DYNAMIC MODEL SELECTION (The Fix) ---
# Instead of guessing the model name, we ask the API what is available.
print("-> Finding available Gemini model...")
target_model_name = None


try:
   for m in genai.list_models():
       # We need a model that supports 'generateContent'
       if 'generateContent' in m.supported_generation_methods:
           # Prefer Flash (fast/cheap), then Pro, then anything else
           if 'flash' in m.name.lower():
               target_model_name = m.name
               break  # Found ideal model
           elif 'pro' in m.name.lower() and not target_model_name:
               target_model_name = m.name


   if not target_model_name:
       # Fallback if search failed
       target_model_name = 'models/gemini-1.5-flash'


   print(f"-> Selected Model: {target_model_name}")
   model = genai.GenerativeModel(target_model_name)


except Exception as e:
   print(f"❌ Error listing models: {e}")
   print("   Using fallback 'models/gemini-1.5-flash'")
   model = genai.GenerativeModel('models/gemini-1.5-flash')


# Database Path
DB_PATH = os.path.join(current_dir, 'data', 'morningnews.db')




def ensure_schema_updates():
   """Checks the database schema and adds necessary AI columns."""
   print("-> Checking database schema for AI columns...")
   if not os.path.exists(DB_PATH):
       print(f"❌ Error: Database not found at {DB_PATH}. Run ingestion first.")
       return


   conn = sqlite3.connect(DB_PATH)
   cursor = conn.cursor()


   new_columns = {
       "sentiment_score": "FLOAT",
       "bias_score": "FLOAT",
       "is_clickbait": "BOOLEAN",
       "ai_summary": "TEXT"
   }


   cursor.execute("PRAGMA table_info(articles)")
   existing_cols = [row[1] for row in cursor.fetchall()]


   for col, dtype in new_columns.items():
       if col not in existing_cols:
           print(f"   + Adding column: {col}")
           try:
               cursor.execute(f"ALTER TABLE articles ADD COLUMN {col} {dtype}")
           except sqlite3.OperationalError as e:
               print(f"   ! Error adding {col}: {e}")


   conn.commit()
   conn.close()




def fetch_unprocessed_articles(limit=10):
   """Retrieves a batch of articles that haven't been analyzed yet."""
   conn = sqlite3.connect(DB_PATH)
   conn.row_factory = sqlite3.Row
   cursor = conn.cursor()


   cursor.execute("""
       SELECT id, title, description, content, source
       FROM articles
       WHERE sentiment_score IS NULL
       LIMIT ?
   """, (limit,))


   rows = cursor.fetchall()
   conn.close()
   return [dict(row) for row in rows]




def analyze_article_with_llm(article):
   """Sends article data to Gemini and requests a structured JSON response."""
   # Safety check for empty content
   desc = article.get('description') or ""
   if len(desc) < 5 and not article.get('content'):
       return None  # Skip articles with no info to analyze


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
       response = model.generate_content(prompt)
       text_response = response.text.strip()


       # Cleanup code blocks if Gemini adds them
       if text_response.startswith("```"):
           text_response = text_response.split("```")[1]
           if text_response.startswith("json"):
               text_response = text_response[4:]


       data = json.loads(text_response.strip())
       return data
   except Exception as e:
       # print(f"   ! Analysis Failed: {e}") # Optional: Uncomment to debug
       return None




def update_article_in_db(article_id, analysis):
   """Saves the AI analysis results back to the database."""
   conn = sqlite3.connect(DB_PATH)
   cursor = conn.cursor()


   cursor.execute("""
       UPDATE articles
       SET sentiment_score = ?, bias_score = ?, is_clickbait = ?, ai_summary = ?
       WHERE id = ?
   """, (
       analysis.get('sentiment_score'),
       analysis.get('bias_score'),
       analysis.get('is_clickbait'),
       analysis.get('ai_summary'),
       article_id
   ))


   conn.commit()
   conn.close()




def main():
   print("--- Starting AI Analysis Pipeline ---")


   ensure_schema_updates()


   articles = fetch_unprocessed_articles(limit=20)
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

