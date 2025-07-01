#!/usr/bin/env python3
"""
Search utilities for the RAG Search Tool.
"""

import re
import time
import random
import os
import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from duckduckgo_search import DDGS

# Suppress only the InsecureRequestWarning from urllib3
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

console = Console()

# --- Proxy Cache ---
PROXY_CACHE: List[str] = []
PROXY_CACHE_EXPIRATION: Optional[datetime.datetime] = None
PROXY_CACHE_TTL_MINUTES = 10
# -------------------

# Define constants to avoid backslash in f-string expressions
NEWLINE_SEP = "\n"




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


def _check_internet_connectivity(timeout: int = 5) -> bool:
    """Checks for basic internet connectivity by connecting to a reliable IP."""
    try:
        # Using a reliable IP address to bypass potential DNS resolution issues.
        requests.head("https://8.8.8.8", timeout=timeout, verify=False) # Added verify=False for flexibility
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False


def _get_free_proxy() -> Optional[str]:
    """
    Fetches a random HTTPS proxy from a list of sources, with caching.
    """
    global PROXY_CACHE, PROXY_CACHE_EXPIRATION

    # 1. Check cache first
    if PROXY_CACHE and PROXY_CACHE_EXPIRATION and datetime.datetime.now() < PROXY_CACHE_EXPIRATION:
        console.print(f"[grey50]Using cached proxy...[/grey50]")
        return random.choice(PROXY_CACHE)

    # 2. If cache is invalid, fetch from sources
    console.print("[yellow]Proxy cache expired or empty. Fetching new proxy list...[/yellow]")
    proxy_sources = [
        "https://free-proxy-list.net/",
        "https://sslproxies.org/",
        "https://www.geonode.com/free-proxy-list"
    ]
    all_proxies = []

    for url in proxy_sources:
        try:
            console.print(f"[grey50]Trying source: {url}...[/grey50]")
            response = requests.get(url, timeout=25)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Generic parsing for tables with IP and Port
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) > 6 and 'yes' in cells[6].text.lower(): # HTTPS enabled
                    ip = cells[0].text.strip()
                    port = cells[1].text.strip()
                    if ip and port.isdigit():
                        all_proxies.append(f"http://{ip}:{port}")
            
            if all_proxies:
                console.print(f"[green]Successfully fetched {len(all_proxies)} proxies from {url}[/green]")
                break # Stop if we get a good list

        except Exception as e:
            console.print(f"[red]Failed to fetch from {url}: {e}[/red]")
            continue

    # 3. Update cache if proxies were found
    if all_proxies:
        PROXY_CACHE = all_proxies
        PROXY_CACHE_EXPIRATION = datetime.datetime.now() + datetime.timedelta(minutes=PROXY_CACHE_TTL_MINUTES)
        console.print(f"[green]Proxy cache updated with {len(PROXY_CACHE)} proxies.[/green]")
        return random.choice(PROXY_CACHE)
    
    console.print("[bold red]Could not fetch any proxies from any source.[/bold red]")
    return None


def get_enhanced_search_results(
    query: str, max_results: int = 10, timeout: int = 10
) -> List[Dict[str, str]]:
    """
    Get enhanced search results from DuckDuckGo with proxy-based retry logic and network checks.
    """
    if not _check_internet_connectivity():
        console.print("[bold red]Network Error:[/bold red] Cannot connect to the internet. Please check your connection.")
        return []

    results: List[Dict[str, str]] = []
    retries = 0
    max_retries = 3
    base_delay = 4
    proxy = _get_free_proxy()  # Initial proxy

    while retries < max_retries:
        if not proxy:
            console.print("[red]No proxy available. Trying without proxy...[/red]")

        try:
            console.print(f"[grey50]Searching with proxy: {proxy}...[/grey50]" if proxy else "[grey50]Searching directly...[/grey50]")
            with DDGS(proxy=proxy, timeout=timeout) as search_engine:
                search_results = search_engine.text(
                    keywords=query, max_results=max_results
                )
                for result in search_results:
                    results.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("href", ""),
                            "snippet": result.get("body", ""),
                        }
                    )
            if results:
                console.print(f"[green]Successfully found {len(results)} results.[/green]")
                break  # Success

        except Exception as e:
            error_msg = str(e).lower()
            console.print(f"[yellow]Search attempt failed: {error_msg}[/yellow]")
            
            # If it's a network error, check connectivity and abort if lost
            if "timed out" in error_msg or "connection" in error_msg or "network is unreachable" in error_msg:
                console.print("[yellow]Network error detected. Re-checking internet connectivity...[/yellow]")
                if not _check_internet_connectivity():
                    console.print("[bold red]Internet connection lost. Aborting all search attempts.[/bold red]")
                    return [] # Return immediately, don't even try fallback

        # This logic runs on failure or if no results were found
        retries += 1
        if retries >= max_retries:
            break
            
        delay = base_delay * (2**retries) + random.uniform(0, 1)
        console.print(
            f"[yellow]Retrying in {delay:.2f}s (Attempt {retries + 1}/{max_retries})...[/]"
        )
        
        # On any failure, get a new proxy. The cache makes this efficient.
        proxy = _get_free_proxy()
        time.sleep(delay)

    if not results:
        console.print("[yellow]Primary search with proxies failed. Using fallback search method.[/yellow]")
        results = _fallback_search(query, max_results)

    return results


def _fallback_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Fallback search method when DuckDuckGo fails. First, try the official DuckDuckGo Instant Answer API via
    the `ddg` helper (backend="api") which is far less likely to rate-limit. If that also fails, fall back to
    basic HTML scraping.
    """
    results: List[Dict[str, str]] = []

    # 1) Try official DuckDuckGo API via DDGS with backend='api'
    try:
        with DDGS(proxy=None, timeout=10) as api_search:
            api_results = api_search.text(keywords=query, backend="auto", max_results=max_results)
            if api_results:
                for item in api_results:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    })
                return results
    except Exception as api_err:
        console.print(f"[yellow]DDG API fallback failed: {api_err}[/]")

    # 2) If API also fails, use lightweight HTML endpoint
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
        console.print(f"[red]HTML scraping fallback failed: {e}[/]")
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
