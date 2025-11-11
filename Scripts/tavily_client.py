"""Thin wrapper around Tavily search API for MorningNews."""

import os
import time
from typing import Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilyError(RuntimeError):
    """Raised when Tavily search fails."""


class TavilyClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = TAVILY_ENDPOINT,
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("No Tavily API key found. Set TAVILY_API_KEY in .env.")
        self.endpoint = endpoint
        self.timeout = timeout
        self.last_response_meta: Dict = {}

    def search(
        self,
        query: str,
        *,
        depth: str = "basic",
        max_results: int = 5,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        """Execute a Tavily search and return the result list."""
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": depth,
            "max_results": max_results,
            "include_domains": include_domains or [],
            "exclude_domains": exclude_domains or [],
        }
        start = time.perf_counter()
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TavilyError(f"Tavily request failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_response_meta = {
            "provider": "Tavily",
            "status_code": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
        }

        json_payload = response.json()
        if "results" not in json_payload:
            raise TavilyError(f"Tavily returned unexpected payload: {json_payload}")
        return json_payload["results"]

    def latest_response_meta(self) -> Dict:
        """Return metadata from the last Tavily call."""
        return dict(self.last_response_meta)
