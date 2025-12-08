import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, List, Any

# --- PATH SETUP (Crucial for Imports) ---
# Add the project root directory (parent_dir) to sys.path so we can import 'storage'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = current_dir

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import helper functions
try:
    from storage import get_db_connection, fetch_latest_articles
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'storage'. Ensure 'storage.py' is in the root directory.")
    sys.exit(1)


# --- REPORT GENERATION FUNCTIONS ---

def get_ingestion_log_summary() -> Dict[str, Any]:
    """Fetches key metrics from the ingestion_log table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query for overall run count, total fetched, total inserted, and latest run time
    summary_query = cursor.execute("""
        SELECT 
            COUNT(*) AS total_runs,
            SUM(fetched) AS total_fetched,
            SUM(inserted) AS total_inserted,
            MAX(finished_at) AS latest_run_time
        FROM ingestion_log
        WHERE topic NOT LIKE '%Error%'
    """).fetchone()

    # Fetch the details of the most recent successful run for a rate limit snapshot
    latest_run_query = cursor.execute("""
        SELECT 
            topic, fetched, inserted, duplicates, rate_limit, finished_at
        FROM ingestion_log
        WHERE topic NOT LIKE '%Error%'
        ORDER BY finished_at DESC
        LIMIT 1
    """).fetchone()

    summary = dict(summary_query) if summary_query else {}
    latest_run = dict(latest_run_query) if latest_run_query else {}
    conn.close()

    rate_limit_info = {}
    if latest_run.get('rate_limit'):
        try:
            rate_limit_info = json.loads(latest_run['rate_limit'])
        except json.JSONDecodeError:
            pass

    # Format the latest run time
    latest_time_str = summary.get('latest_run_time')
    if latest_time_str:
        try:
            # Parse the standard ISO string (which ends in 'Z' or '+00:00')
            dt_utc = datetime.fromisoformat(latest_time_str.replace('Z', '+00:00'))
            # Format for user display (e.g., 'Dec 1, 2025 10:30 AM UTC')
            summary['latest_run_time_formatted'] = dt_utc.strftime('%b %d, %Y %I:%M %p UTC')
        except ValueError:
            summary['latest_run_time_formatted'] = "N/A"
    else:
        summary['latest_run_time_formatted'] = "No successful runs yet"

    return {
        'total_runs': summary.get('total_runs', 0),
        'total_fetched': summary.get('total_fetched', 0),
        'total_inserted': summary.get('total_inserted', 0),
        'latest_run_time': summary.get('latest_run_time_formatted'),
        'rate_limit': rate_limit_info.get('credits_left', 'N/A'),
        'rate_limit_total': rate_limit_info.get('max_credits', 'N/A'),
        'latest_topic': latest_run.get('topic', 'N/A').replace('Tavily_', '').replace('NewsAPI_', ''),
        'latest_inserted': latest_run.get('inserted', 0),
        'latest_duplicates': latest_run.get('duplicates', 0),
    }


def generate_report_html(articles_by_topic: Dict[str, List[Dict]], summary: Dict[str, Any]) -> str:
    """Creates a beautiful, single-file HTML dashboard using Tailwind CSS."""

    # --- Helper function to format articles into HTML list items ---
    def render_articles(articles: List[Dict]) -> str:
        if not articles:
            return '<p class="text-gray-500 italic p-4">No recent articles found for this topic.</p>'

        items = []
        for article in articles:
            published_dt_str = article.get('published_at', 'N/A')
            try:
                # Format to a readable date (e.g., '10:00 AM, Dec 1')
                dt = datetime.fromisoformat(published_dt_str.replace('Z', '+00:00'))
                time_str = dt.strftime('%I:%M %p, %b %d')
            except:
                time_str = 'N/A'

            items.append(f"""
                <li class="mb-4 border-b border-gray-100 pb-4">
                    <a href="{article.get('url', '#')}" target="_blank" class="block hover:bg-indigo-50 p-2 -mx-2 rounded-lg transition duration-150">
                        <p class="font-semibold text-gray-900 text-lg leading-snug">{article.get('title', 'No Title')}</p>
                        <p class="text-sm text-gray-600 mt-1">{article.get('description', 'No description available.')}</p>
                        <div class="mt-2 text-xs flex items-center text-indigo-600">
                            <span class="font-medium mr-4">{article.get('source', 'Unknown Source')}</span>
                            <span class="text-gray-400">|</span>
                            <span class="ml-4 flex items-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                {time_str}
                            </span>
                        </div>
                    </a>
                </li>
            """)
        return '\n'.join(items)

    # --- Metrics Section HTML ---
    metrics_html = f"""
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
        <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <p class="text-sm font-medium text-gray-500">Total Runs</p>
            <p class="text-3xl font-bold text-gray-900 mt-1">{summary['total_runs']}</p>
            <p class="text-xs text-gray-400 mt-1">Ingestion attempts made</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <p class="text-sm font-medium text-gray-500">Total Inserted</p>
            <p class="text-3xl font-bold text-green-600 mt-1">{summary['total_inserted']}</p>
            <p class="text-xs text-gray-400 mt-1">Unique articles in database</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <p class="text-sm font-medium text-gray-500">Last Successful Run</p>
            <p class="text-base font-semibold text-gray-900 mt-1">{summary['latest_run_time']}</p>
            <p class="text-xs text-gray-400 mt-1">Latest topic: {summary['latest_topic']}</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <p class="text-sm font-medium text-gray-500">Tavily Credits Left</p>
            <p class="text-3xl font-bold text-indigo-600 mt-1">{summary['rate_limit']}</p>
            <p class="text-xs text-gray-400 mt-1">Out of total {summary['rate_limit_total']}</p>
        </div>
    </div>
    """

    # --- Articles Section HTML ---
    articles_html = ""
    for topic, articles in articles_by_topic.items():
        articles_html += f"""
        <div class="w-full lg:w-1/2 p-4">
            <div class="bg-white p-6 rounded-xl shadow-xl h-full">
                <h2 class="text-2xl font-bold text-gray-900 mb-6 border-b pb-3 text-indigo-700">{topic}</h2>
                <ul class="list-none p-0 m-0">
                    {render_articles(articles)}
                </ul>
            </div>
        </div>
        """

    # --- Full HTML Template ---
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MorningNews Ingestion Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f4f7f9;
        }}
    </style>
</head>
<body>
    <div class="container mx-auto p-4 sm:p-8 lg:p-12">
        <!-- Header -->
        <header class="mb-12">
            <h1 class="text-5xl font-extrabold text-gray-900 mb-2">MorningNews Dashboard</h1>
            <p class="text-xl text-gray-600">Real-time summary of news ingestion and database status.</p>
        </header>

        <!-- Metrics Summary -->
        {metrics_html}

        <!-- Latest Articles by Topic -->
        <h2 class="text-3xl font-bold text-gray-900 mb-6 mt-12 border-b-2 border-indigo-200 pb-2">Latest News Articles</h2>
        <div class="flex flex-wrap -m-4">
            {articles_html}
        </div>

        <!-- Footer -->
        <footer class="mt-16 text-center text-sm text-gray-500 border-t pt-8">
            Data sourced via Tavily API and stored in morning_news.db.
        </footer>
    </div>
</body>
</html>
"""


def main():
    """Main function to run the report generation process."""
    print("\n--- Generating MorningNews Dashboard ---")

    # 1. Fetch Summary Metrics
    summary = get_ingestion_log_summary()
    print("-> Fetched Ingestion Summary.")
    # print(summary) # Debug

    # 2. Fetch Latest Articles (Default: 5 per topic)
    articles_by_topic = fetch_latest_articles(limit_per_topic=5)
    print(
        f"-> Fetched {sum(len(v) for v in articles_by_topic.values())} total articles across {len(articles_by_topic)} topics.")

    # 3. Generate HTML Content
    html_content = generate_report_html(articles_by_topic, summary)

    # 4. Save the HTML to a file
    report_file_path = os.path.join(parent_dir, 'morning_news_report.html')
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ Dashboard generated successfully!")
    print(f"File saved to: {report_file_path}")
    print(f"Open '{report_file_path}' in your browser to view the report.")


if __name__ == "__main__":
    main()