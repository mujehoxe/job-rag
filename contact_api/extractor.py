#!/usr/bin/env python3
"""
Contact Extractor class.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import g4f
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress

from .utils import ContentExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")

# Initialize console
console = Console()


class ContactExtractor:
    """Extract contact information from a given domain."""

    def __init__(
        self,
        g4f_model: str = os.getenv("G4F_MODEL", "gpt-4o"),
        search_results_limit: int = 10,
        max_search_queries: int = 5,
        ai_summary_timeout: int = 300,
    ):
        self.g4f_model = g4f_model
        self.search_results_limit = search_results_limit
        self.max_search_queries = max_search_queries
        self.ai_summary_timeout = ai_summary_timeout
        self.visited_urls = set()

    def extract_contacts(self, domain: str) -> Dict[str, Any]:
        """Main function to extract contact information."""
        console.print(Panel(f"[bold cyan]Starting contact extraction for:[/] {domain}", expand=False))

        documents = self._perform_web_searches(domain)
        if not documents:
            log.warning("No documents found from web searches.")
            return {}

        contact_info = self._extract_from_documents(domain, documents)

        ai_summary = self._generate_ai_summary(domain, documents, contact_info)
        contact_info["ai_summary"] = ai_summary

        console.print(Panel("[bold green]Contact extraction complete.[/]", expand=False))
        return contact_info

    def _perform_web_searches(self, domain: str) -> List[Dict[str, str]]:
        """Perform web searches and collect relevant documents."""
        search_queries = self._generate_search_queries(domain)
        all_documents = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Performing web searches...[/]", total=len(search_queries))
            for query in search_queries:
                try:
                    with DDGS(timeout=20) as ddgs:
                        results = list(ddgs.text(query, max_results=self.search_results_limit))
                        for res in results:
                            if res['href'] not in self.visited_urls:
                                content = ContentExtractor.extract_from_url(res['href'])
                                if content:
                                    all_documents.append({
                                        'url': res['href'],
                                        'content': content,
                                        'query': query
                                    })
                                self.visited_urls.add(res['href'])
                except Exception as e:
                    log.error(f"Error during search for query '{query}': {e}")
                progress.update(task, advance=1)
        return all_documents

    def _generate_search_queries(self, domain: str) -> List[str]:
        """Generate a list of search queries for a domain."""
        return [
            f'contact information for {domain}',
            f'"{domain}" email addresses',
            f'about us {domain}',
            f'linkedin profiles for {domain} employees',
            f'key people at {domain}',
        ][:self.max_search_queries]

    def _extract_from_documents(self, domain: str, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract contact details from the collected documents."""
        contacts = {
            'emails': set(),
            'phones': set(),
            'social_media': {},
            'people': set()
        }

        with Progress() as progress:
            task = progress.add_task("[cyan]Extracting contact info...[/]", total=len(documents))
            for doc in documents:
                content = doc.get('content', '')
                if not content:
                    continue

                contacts['emails'].update(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content))
                contacts['phones'].update(re.findall(r'\+?\d{1,3}?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', content))
                
                # Social media
                sm_patterns = {
                    'linkedin': r'linkedin\.com/in/[\w-]+',
                    'twitter': r'twitter\.com/[\w]+',
                    'facebook': r'facebook\.com/[\w.]+'
                }
                for platform, pattern in sm_patterns.items():
                    found = re.findall(pattern, content)
                    if found:
                        if platform not in contacts['social_media']:
                            contacts['social_media'][platform] = set()
                        contacts['social_media'][platform].update(found)

                progress.update(task, advance=1)

        # Convert sets to lists for JSON serialization
        contacts['emails'] = list(contacts['emails'])
        contacts['phones'] = list(contacts['phones'])
        for platform in contacts['social_media']:
            contacts['social_media'][platform] = list(contacts['social_media'][platform])

        return contacts

    def _generate_ai_summary(
        self, domain: str, documents: List[Dict[str, str]], contact_info: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive AI summary of the findings."""
        console.print("[cyan]Generating AI summary...[/]")

        # Prepare content for the AI
        document_content = "\n".join([f"<document url='{d['url']}' query='{d['query']}'>\n{d['content']}\n</document>" for d in documents])
        contact_info_text = f"<extracted_contacts>\n{contact_info}\n</extracted_contacts>"

        system_prompt = (
            "You are a meticulous data analyst. Your task is to synthesize information from various web pages "
            "to provide a detailed summary of contact information for a given domain. Your output must be structured, "
            "accurate, and comprehensive. First, provide a 'Chain of Thought' section explaining your reasoning, "
            "then provide the final summary."
        )

        user_prompt = (
            f"Based on the following web content and extracted contact information for '{domain}', "
            f"provide a comprehensive summary. Identify key people, their roles, and any available contact details. "
            f"List all found email addresses, phone numbers, and social media profiles.\n\n"
            f"<data>\n{contact_info_text}\n{document_content}\n</data>"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = g4f.ChatCompletion.create(
                model=self.g4f_model,
                messages=messages,
                timeout=self.ai_summary_timeout,
            )
            return str(response)
        except Exception as e:
            log.error(f"AI summary generation failed: {e}")
            return "AI summary could not be generated due to an error."
