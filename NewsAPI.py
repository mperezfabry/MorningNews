import os
import requests
from datetime import date, timedelta
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class NewsAPI:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("No API key found. Please set NEWS_API_KEY in .env file")
        self.headers = {'X-Api-Key': self.api_key}
        self.base_url = 'https://newsapi.org/v2/everything'

    def get_news(self, params):
        response = requests.get(self.base_url, headers=self.headers, params=params)
        return response.json()

def print_results(data, limit=5):
    """Print first few articles from the response"""
    if data['status'] == 'ok':
        print(f"\nFound {data['totalResults']} articles. Showing first {limit}:")
        for article in data['articles'][:limit]:
            print(f"\nTitle: {article['title']}")
            print(f"Source: {article['source']['name']}")
            print(f"Date: {article['publishedAt']}")
            print(f"URL: {article['url']}\n")
    else:
        print(f"Error: {data}")

def main():
    news_api = NewsAPI()
    today = date.today()

    # Example 1: Basic keyword search
    print("\n=== Example 1: Search for Apple news ===")
    params = {
        'q': 'Apple',
        'from': today.isoformat(),
        'sortBy': 'popularity',
        'language': 'en'
    }
    print_results(news_api.get_news(params))

    # Example 2: Advanced search with multiple keywords and operators
    print("\n=== Example 2: Advanced search with operators ===")
    params = {
        'q': '"artificial intelligence" AND (Microsoft OR Google) -Apple',
        'from': (today - timedelta(days=7)).isoformat(),
        'sortBy': 'relevancy',
        'language': 'en'
    }
    print_results(news_api.get_news(params))

    # Example 3: Search in title only with date range
    print("\n=== Example 3: Title-only search with date range ===")
    params = {
        'qInTitle': 'iPhone',
        'from': (today - timedelta(days=30)).isoformat(),
        'to': today.isoformat(),
        'sortBy': 'publishedAt',
        'language': 'en'
    }
    print_results(news_api.get_news(params))

    # Example 4: Multiple domains search
    print("\n=== Example 4: Search specific domains ===")
    params = {
        'domains': 'techcrunch.com,wired.com',
        'q': 'startup',
        'sortBy': 'publishedAt',
        'pageSize': 5
    }
    print_results(news_api.get_news(params))

    # Example 5: Exclude domains and use pagination
    print("\n=== Example 5: Exclude domains and use pagination ===")
    params = {
        'q': 'technology',
        'excludeDomains': 'theverge.com,engadget.com',
        'page': 2,
        'pageSize': 5,
        'sortBy': 'popularity',
        'language': 'en'
    }
    print_results(news_api.get_news(params))

if __name__ == '__main__':
    main()