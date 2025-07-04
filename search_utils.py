#!/usr/bin/env python3
"""
Search utilities for the RAG Search Tool with multiple search engines.
"""

import re
import time
import random
import os
import datetime
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, quote_plus

import requests
from bs4 import BeautifulSoup
from rich.console import Console

# Suppress only the InsecureRequestWarning from urllib3
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

console = Console()

# --- Rate Limiting ---
LAST_SEARCH_TIME = {}
MIN_SEARCH_INTERVAL = 1.0  # Reduced to 1 second for API-based searches

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
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        return unique_results


def _enforce_rate_limit(engine_name: str):
    """Enforce rate limiting per search engine"""
    current_time = time.time()
    if engine_name in LAST_SEARCH_TIME:
        time_since_last = current_time - LAST_SEARCH_TIME[engine_name]
        if time_since_last < MIN_SEARCH_INTERVAL:
            sleep_time = MIN_SEARCH_INTERVAL - time_since_last
            console.print(f"[yellow]Rate limiting {engine_name}: sleeping {sleep_time:.1f}s[/yellow]")
            time.sleep(sleep_time)
    
    LAST_SEARCH_TIME[engine_name] = time.time()


def _search_brave(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search using Brave Search API (5,000 free queries/month, 1 query/second)"""
    _enforce_rate_limit("brave")
    results = []
    
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        console.print("[yellow]BRAVE_SEARCH_API_KEY not found in environment[/yellow]")
        return results
    
    try:
        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json"
        }
        
        params = {
            "q": query,
            "count": max_results,
            "offset": 0,
            "mkt": "en-US",
            "safesearch": "moderate",
            "textDecorations": False,
            "textFormat": "Raw"
        }
        
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        web_results = data.get("web", {}).get("results", [])
        
        for result in web_results[:max_results]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("description", ""),
                "source": "brave"
            })
        
        console.print(f"[green]Brave Search: Found {len(results)} results[/green]")

    except Exception as e:
        console.print(f"[red]Brave Search failed: {e}[/red]")
    
    return results


def _search_serper(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search using Serper.dev API (2,500 free queries/month)"""
    _enforce_rate_limit("serper")
    results = []
    
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        console.print("[yellow]SERPER_API_KEY not found in environment[/yellow]")
        return results
    
    try:
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": max_results
        }
        
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        organic_results = data.get("organic", [])
        
        for result in organic_results[:max_results]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "serper"
            })
        
        console.print(f"[green]Serper: Found {len(results)} results[/green]")

    except Exception as e:
        console.print(f"[red]Serper search failed: {e}[/red]")
    
    return results


def _search_google_custom(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search using Google Custom Search API (100 free queries/day, $5 per 1000 additional)"""
    _enforce_rate_limit("google_custom")
    results = []
    
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key or not search_engine_id:
        console.print("[yellow]GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID not found in environment[/yellow]")
        return results
    
    try:
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": min(max_results, 10),  # Google Custom Search max is 10 per request
            "safe": "medium"
        }
        
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        for item in items[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "google_custom"
            })
        
        console.print(f"[green]Google Custom Search: Found {len(results)} results[/green]")
        
    except Exception as e:
        console.print(f"[red]Google Custom Search failed: {e}[/red]")

    return results


def _search_duckduckgo_lite(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search using DuckDuckGo Lite (free, no API key needed)"""
    _enforce_rate_limit("ddg_lite")
    results = []
    
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            search_results = ddgs.text(keywords=query, max_results=max_results, backend="lite")
            
            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "ddg_lite"
                })
        
        console.print(f"[green]DuckDuckGo Lite: Found {len(results)} results[/green]")
        
    except Exception as e:
        console.print(f"[red]DuckDuckGo Lite search failed: {e}[/red]")
    
    return results


def _search_bing_scraping(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search using Bing (free tier through scraping, fallback option)"""
    _enforce_rate_limit("bing_scraping")
    results = []
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse Bing search results
        for result in soup.find_all('li', class_='b_algo')[:max_results]:
            title_elem = result.find('h2')
            if title_elem and title_elem.find('a'):
                title = title_elem.get_text().strip()
                url = title_elem.find('a')['href']
                
                snippet_elem = result.find('p', class_='b_lineclamp2 b_algoSlug') or result.find('p')
                snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "bing_scraping"
                })
                
        console.print(f"[green]Bing Scraping: Found {len(results)} results[/green]")
        
    except Exception as e:
        console.print(f"[red]Bing scraping search failed: {e}[/red]")
    
    return results


def get_enhanced_search_results(
    query: str, max_results: int = 15, timeout: int = 15
) -> List[Dict[str, str]]:
    """
    Get search results using multiple search engines with rate limiting.
    Prioritizes API-based services over scraping.
    """
    all_results = []
    target_results = max_results
    
    # Define search engines in order of preference (API-based first)
    search_engines = [
        ("Brave Search API", _search_brave),
        ("Serper API", _search_serper),
        ("Google Custom Search", _search_google_custom),
        ("DuckDuckGo Lite", _search_duckduckgo_lite),
        ("Bing Scraping", _search_bing_scraping),
    ]
    
    for engine_name, search_func in search_engines:
        if len(all_results) >= target_results:
            break
            
        console.print(f"[cyan]Trying {engine_name}...[/cyan]")
        
        try:
            results = search_func(query, max_results - len(all_results))
            if results:
                all_results.extend(results)
                console.print(f"[green]{engine_name} contributed {len(results)} results[/green]")
                # If we get good results from API services, we can be more selective
                if "API" in engine_name and len(all_results) >= target_results // 2:
                    break
            else:
                console.print(f"[yellow]{engine_name} returned no results[/yellow]")
                
        except Exception as e:
            console.print(f"[red]{engine_name} failed: {e}[/red]")
            continue
        
        # Shorter delay between API services, longer for scraping
        if len(all_results) < target_results:
            delay = 0.5 if "API" in engine_name else 2.0
            time.sleep(delay)
    
    # Remove duplicates and rank results
    unique_results = SearchResultProcessor.filter_duplicates(all_results)
    ranked_results = SearchResultProcessor.rank_results(unique_results, query)
    
    console.print(f"[bold green]Total unique results: {len(unique_results)}[/bold green]")
    
    return ranked_results[:max_results]


def _fallback_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Simple fallback when all search engines fail
    """
    console.print("[yellow]All search engines failed. Using fallback...[/yellow]")
    
    return [
                {
                    "title": f"Search result for {query}",
                    "url": f"https://example.com/search?q={query}",
            "snippet": "Search services are currently experiencing technical difficulties. Please try again later.",
            "source": "fallback"
                }
            ]
