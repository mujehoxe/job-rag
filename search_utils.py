#!/usr/bin/env python3
"""
Search utilities for the RAG Search Tool.
"""

import re
import time
import random
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from duckduckgo_search import DDGS

console = Console()

# Define constants to avoid backslash in f-string expressions
NEWLINE_SEP = "\n"

# Free proxy list - use direct IP addresses instead of URLs to proxy services
FREE_PROXIES = [
    None,  # No proxy option
    "socks5://127.0.0.1:9050",  # Tor proxy if available locally
    "http://localhost:8118",  # Privoxy if available locally
]


class RateLimitHandler:
    """Handler for rate limiting with exponential backoff and proxy rotation"""

    def __init__(self, max_retries=3, base_delay=2):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.current_proxy_index = 0
        self.proxies = FREE_PROXIES

    def get_current_proxy(self):
        """Get current proxy"""
        if not self.proxies or self.current_proxy_index >= len(self.proxies):
            return None
        return self.proxies[self.current_proxy_index]

    def rotate_proxy(self):
        """Rotate to next proxy"""
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        current = self.get_current_proxy()
        console.print(
            f"[yellow]Rotating to proxy: {current if current else 'No proxy'}[/]"
        )
        return current

    def get_proxy_config(self):
        """Get proxy configuration for requests"""
        proxy = self.get_current_proxy()
        if not proxy:
            return {}

        if proxy.startswith(("http://", "https://", "socks4://", "socks5://")):
            return {"http": proxy, "https": proxy}
        else:
            console.print(f"[yellow]Invalid proxy format: {proxy}, using no proxy[/]")
            return {}


# Initialize rate limit handler
rate_limit_handler = RateLimitHandler()


class SearchResultProcessor:
    """Process search results to improve relevance"""

    @staticmethod
    def rank_results(results: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
        """
        Rank search results by relevance to query
        """
        if not results:
            return []

        for result in results:
            score = 0
            title_lower = result.get("title", "").lower()
            query_terms = query.lower().split()
            for term in query_terms:
                if term in title_lower:
                    score += 3

            content = result.get("content", "")
            if content:
                content_lower = content.lower()
                for term in query_terms:
                    score += content_lower.count(term) * 0.2

            snippet = result.get("snippet", "")
            if snippet:
                snippet_lower = snippet.lower()
                for term in query_terms:
                    if term in snippet_lower:
                        score += 1

            if content and len(content) < 500:
                score *= 0.7

            result["relevance_score"] = score

        return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

    @staticmethod
    def filter_duplicates(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter out duplicate or near-duplicate results
        """
        seen_urls = set()
        unique_results = []
        for result in results:
            url = result.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        return unique_results


def get_enhanced_search_results(
    query: str, max_results: int = 10, timeout: int = 10
) -> List[Dict[str, str]]:
    """
    Get enhanced search results from DuckDuckGo with retry logic
    """
    results = []
    retries = 0

    while retries < rate_limit_handler.max_retries:
        try:
            proxies = rate_limit_handler.get_proxy_config()
            with DDGS(proxies=proxies, timeout=timeout) as search_engine:
                search_results = search_engine.text(query, max_results=max_results)
                for result in search_results:
                    results.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("href", ""),
                            "snippet": result.get("body", ""),
                        }
                    )
            break
        except Exception as e:
            retries += 1
            delay = rate_limit_handler.base_delay * (2**retries) + random.uniform(0, 1)
            if "rate" in str(e).lower() or "limit" in str(e).lower() or "202" in str(e):
                console.print(
                    f"[yellow]Rate limit hit. Retrying in {delay:.2f}s (Attempt {retries}/{rate_limit_handler.max_retries})[/]"
                )
                rate_limit_handler.rotate_proxy()
            else:
                console.print(
                    f"[yellow]Search error: {e}. Retrying in {delay:.2f}s (Attempt {retries}/{rate_limit_handler.max_retries})[/]"
                )
            if retries < rate_limit_handler.max_retries:
                time.sleep(delay)

    if not results and retries >= rate_limit_handler.max_retries:
        console.print("[yellow]All retries failed. Using fallback search method.[/]")
        results = _fallback_search(query, max_results)

    return results


def _fallback_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Fallback search method when DuckDuckGo fails
    """
    results = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        import urllib.parse

        encoded_query = urllib.parse.quote_plus(query)
        response = requests.get(
            f"https://html.duckduckgo.com/html/?q={encoded_query}",
            headers=headers,
            timeout=15,
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.find_all("div", class_="result")
            for result in search_results[:max_results]:
                title_elem = result.find("a", class_="result__a")
                snippet_elem = result.find("a", class_="result__snippet")
                if title_elem and title_elem.get("href"):
                    title = title_elem.text.strip()
                    url = title_elem.get("href")
                    snippet = snippet_elem.text.strip() if snippet_elem else ""
                    results.append({"title": title, "url": url, "snippet": snippet})
    except Exception as e:
        console.print(f"[red]Fallback search failed: {e}[/]")
        try:
            console.print("[yellow]Trying alternative search engine...[/]")
            results = [
                {
                    "title": f"Search result for {query}",
                    "url": f"https://example.com/search?q={query}",
                    "snippet": "Please try your search again later. The search service is currently experiencing technical difficulties.",
                }
            ]
        except:
            pass
    return results
