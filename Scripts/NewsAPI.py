import os
import json
from datetime import datetime
from dotenv import load_dotenv
from newsapi import NewsApiClient
from typing import List, Dict, Any

# Loads environment variables from a .env file into the environment
load_dotenv()

# Categories we will pull for the dashboard
CATEGORIES = ['business', 'technology', 'science', 'health', 'sports']

# Pulls 30 articles per category (can be changed)
PAGE_SIZE = 30

# US headlines only for now. Also can change this aspect. Feel free to adjust this, especially if we decide pull data from outside the US. We probably have to add an additional function(s) that can translate other languages to english.
COUNTRY = 'us'


# Initializes the NewsAPI client
def get_api_client():
    api_key = os.getenv('NEWS_API_KEY')
    # print(api_key)                      # If this line is still here when it gets to the repo, you can remove it. I only have it here to make sure API connect to key in my personal .env
    if not api_key:
        raise ValueError('No NewsAPI key found. Set NEWS_API_KEY in .env.')
    return NewsApiClient(api_key=api_key)


# This function is a basic quality filter that maybe one of you can look at and improve it if you see any flaws or just overall issues
# It tosses incomplete articles/ just junk from the API like placeholders in titles, empty titles, empty description, broken or missing URLs
def clean_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_list = []
    for article in articles:
        title = article.get('title')
        url = article.get('url')
        description = article.get('description')

        # Quality checks
        if not title or title.lower() == '[removed]':  # Skips if title missing or placeholder is present
            continue

        # URL to link sources
        if not url:
            continue

        # Description for context
        if not description:
            continue

        # Flatten source for easier DB ingestion later
        # NewsAPI gives: 'source': {'id': 'cnn', 'name': 'CNN'}
        # We ensure 'source_name' is easily accessible
        if isinstance(article.get('source'), dict):
            article['source_name'] = article['source'].get('name', 'Unknown Source')
        else:
            article['source_name'] = 'Unknown Source'

        clean_list.append(article)
    return clean_list


# This function will grab headlines for one category filter them and save a list as json
#
def fetch_and_save_category(client, category):
    print(f"Start fetch for {category.upper()}...")
    try:
        # Use client's top_headlines method.
        response = client.get_top_headlines(
            category=category,
            country=COUNTRY,
            page_size=PAGE_SIZE
        )
    except Exception as e:
        print(f"Error fetching articles for {category}: {e}")
        return

    # This if checks the API success status
    if response.get('status') != 'ok':
        print(f"API returned ERROR!!! Error fetching articles for {category}: {response.get('code', 'Unknown ERROR')}")
        return

    articles = response.get('articles', [])
    print(f" Pulled/Fetched {len(articles)} articles")

    # SIMPLE cleaning filter on the fetched articles
    final_articles = clean_articles(articles)

    print(
        f" Kept {len(final_articles)} articles after quality check and filtering out of articles that does not meet criteria")

    if not final_articles:
        return

    # Save Logic
    # Make sure the data folder does exist...
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    # timestamps in the filenames for unique trackable files.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{category.lower()}_headlines.json"
    filepath = os.path.join(data_dir, filename)
    # print(timestamp)
    # print(filename)
    # print(filepath)

    # Save cleaned articles list to a file
    with open(filepath, 'w', encoding='utf-8') as f:
        # Set indent to 4 so that json file is readable
        json.dump(final_articles, f, ensure_ascii=False, indent=4)

    print(f"SAVED{len(final_articles)} articles to {filepath}")


# Function first gets client object
# then interates through each categories list and processes each one
def main():
    try:
        news_client = get_api_client()
        # print(news_client)
        for cat in CATEGORIES:
            fetch_and_save_category(news_client, cat)
        print("\n ALL DONE! All ingestion tasks successfully finsihed \n ")
    except Exception as e:
        print(f"\n Some unexpected error occurred in the main function execution: {e}")


if __name__ == "__main__":
    main()