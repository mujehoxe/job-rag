#!/usr/bin/env python3
"""
Content extraction utilities.
"""

import re
from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

# Define constants to avoid backslash in f-string expressions
NEWLINE_SEP = "\n"


class ContentExtractor:
    """Extract content from different types of websites with fallback mechanisms"""

    @staticmethod
    def extract_from_url(url: str, timeout: int = 10) -> Optional[str]:
        """Extract content from a URL with fallback mechanisms"""
        try:
            domain = urlparse(url).netloc

            # Try primary extraction method
            content = None
            
            # Use specific extractors for known domains
            if "wikipedia.org" in domain:
                content = ContentExtractor._extract_wikipedia(url, timeout)
            elif "github.com" in domain:
                content = ContentExtractor._extract_github(url, timeout)
            elif "stackoverflow.com" in domain or "stackexchange.com" in domain:
                content = ContentExtractor._extract_stackoverflow(url, timeout)
            elif "arxiv.org" in domain:
                content = ContentExtractor._extract_arxiv(url, timeout)
            elif "medium.com" in domain or "towardsdatascience.com" in domain:
                content = ContentExtractor._extract_medium(url, timeout)
            else:
                # Generic extractor for other sites
                content = ContentExtractor._extract_generic(url, timeout)
            
            # If primary extraction failed, try fallback methods
            if not content:
                console.print(f"[yellow]Primary extraction failed for {url}, trying fallbacks...[/yellow]")
                
                # Try Wayback Machine
                content = ContentExtractor._extract_from_wayback(url, timeout)
                
                # If still no content, try Google Cache (less reliable)
                if not content:
                    content = ContentExtractor._extract_from_google_cache(url, timeout)
                
                # Last resort: return basic info
                if not content:
                    console.print(f"[red]All extraction methods failed for {url}[/red]")
                    return f"Content extraction failed for: {url}"
            
            return content
            
        except Exception as e:
            console.print(f"[dim]Error extracting from {url}: {e}[/]")
            return f"Content extraction failed for: {url}"

    @staticmethod
    def _extract_generic(url: str, timeout: int) -> Optional[str]:
        """Generic content extractor for most websites"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script.extract()

            # Get text
            text = soup.get_text(separator=NEWLINE_SEP, strip=True)

            # Clean up text
            text = ContentExtractor._clean_text(text)

            return text
        except Exception as e:
            console.print(f"[dim]Failed to extract generic content from {url}: {e}[/]")
            return None

    @staticmethod
    def _extract_wikipedia(url: str, timeout: int) -> Optional[str]:
        """Extract content from Wikipedia"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Get the main content
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                return ContentExtractor._extract_generic(url, timeout)

            # Remove unwanted elements
            for unwanted in content_div.find_all(
                ["table", "style", "script", "sup", "span.mw-editsection"]
            ):
                unwanted.extract()

            # Get all paragraphs and headings
            elements = content_div.find_all(["p", "h2", "h3", "h4", "h5", "h6"])
            text = "\n\n".join([elem.get_text() for elem in elements])

            # Clean up text
            text = ContentExtractor._clean_text(text)

            return text
        except Exception as e:
            console.print(
                f"[dim]Failed to extract Wikipedia content from {url}: {e}[/]"
            )
            return ContentExtractor._extract_generic(url, timeout)

    @staticmethod
    def _extract_github(url: str, timeout: int) -> Optional[str]:
        """Extract content from GitHub"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Check if it's a README page
            readme = soup.find("article", {"class": "markdown-body"})
            if readme:
                text = readme.get_text(separator=NEWLINE_SEP, strip=True)
                return ContentExtractor._clean_text(text)

            # Check if it's a code file
            code_content = soup.find("table", {"class": "highlight"})
            if code_content:
                text = code_content.get_text(separator=NEWLINE_SEP, strip=True)
                return ContentExtractor._clean_text(text)

            # Check if it's an issue or PR
            issue_content = soup.find("div", {"class": "comment-body"})
            if issue_content:
                text = issue_content.get_text(separator=NEWLINE_SEP, strip=True)
                return ContentExtractor._clean_text(text)

            # Fallback to generic extraction
            return ContentExtractor._extract_generic(url, timeout)
        except Exception as e:
            console.print(f"[dim]Failed to extract GitHub content from {url}: {e}[/]")
            return ContentExtractor._extract_generic(url, timeout)

    @staticmethod
    def _extract_stackoverflow(url: str, timeout: int) -> Optional[str]:
        """Extract content from Stack Overflow or Stack Exchange"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Get the question
            question = soup.find("div", {"class": "question"})
            question_text = ""
            if question:
                question_title = soup.find("h1", {"class": "fs-headline1"})
                question_body = question.find("div", {"class": "s-prose"})

                if question_title:
                    question_text += (
                        f"Question: {question_title.get_text(strip=True)}\n\n"
                    )
                if question_body:
                    body_text = question_body.get_text(
                        separator=NEWLINE_SEP, strip=True
                    )
                    question_text += f"{body_text}\n\n"

            # Get the answers
            answers = soup.find_all("div", {"class": "answer"})
            answers_text = ""

            for i, answer in enumerate(answers, 1):
                answer_body = answer.find("div", {"class": "s-prose"})
                if answer_body:
                    is_accepted = "accepted-answer" in answer.get("class", [])
                    answers_text += (
                        f"Answer {i}{' (Accepted)' if is_accepted else ''}:\n"
                    )
                    body_text = answer_body.get_text(separator=NEWLINE_SEP, strip=True)
                    answers_text += f"{body_text}\n\n"

            text = question_text + answers_text
            return ContentExtractor._clean_text(text)
        except Exception as e:
            console.print(
                f"[dim]Failed to extract Stack Overflow content from {url}: {e}[/]"
            )
            return ContentExtractor._extract_generic(url, timeout)

    @staticmethod
    def _extract_arxiv(url: str, timeout: int) -> Optional[str]:
        """Extract content from arXiv papers"""
        try:
            if "pdf" in url:
                abstract_url = url.replace("pdf", "abs")
            else:
                abstract_url = url

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(abstract_url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            title_element = soup.find("h1", {"class": "title"})
            title = (
                title_element.get_text(strip=True).replace("Title:", "")
                if title_element
                else ""
            )

            authors_element = soup.find("div", {"class": "authors"})
            authors = (
                authors_element.get_text(strip=True).replace("Authors:", "")
                if authors_element
                else ""
            )

            abstract_element = soup.find("blockquote", {"class": "abstract"})
            abstract = (
                abstract_element.get_text(strip=True).replace("Abstract:", "")
                if abstract_element
                else ""
            )

            text = f"Title: {title}\nAuthors: {authors}\nAbstract: {abstract}"
            return ContentExtractor._clean_text(text)
        except Exception as e:
            console.print(f"[dim]Failed to extract arXiv content from {url}: {e}[/]")
            return ContentExtractor._extract_generic(url, timeout)

    @staticmethod
    def _extract_medium(url: str, timeout: int) -> Optional[str]:
        """Extract content from Medium or related sites"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            title_element = soup.find("h1")
            title = title_element.get_text(strip=True) if title_element else ""

            article = soup.find("article")
            if not article:
                return ContentExtractor._extract_generic(url, timeout)

            paragraphs = article.find_all(["p", "h2", "h3", "h4", "pre", "blockquote"])
            content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])

            text = f"Title: {title}\n\n{content}"
            return ContentExtractor._clean_text(text)
        except Exception as e:
            console.print(f"[dim]Failed to extract Medium content from {url}: {e}[/]")
            return ContentExtractor._extract_generic(url, timeout)

    @staticmethod
    def _extract_from_wayback(url: str, timeout: int) -> Optional[str]:
        """Extract content from Wayback Machine"""
        try:
            # Get the latest snapshot from Wayback Machine
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # First, check if there are any snapshots
            availability_api = f"https://archive.org/wayback/available?url={url}"
            response = requests.get(availability_api, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                archived_snapshots = data.get("archived_snapshots", {})
                closest = archived_snapshots.get("closest", {})
                
                if closest.get("available"):
                    snapshot_url = closest.get("url")
                    console.print(f"[green]Found Wayback Machine snapshot: {snapshot_url}[/green]")
                    
                    # Extract content from the snapshot
                    snapshot_response = requests.get(snapshot_url, headers=headers, timeout=timeout)
                    snapshot_response.raise_for_status()
                    
                    soup = BeautifulSoup(snapshot_response.text, "html.parser")
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                        script.extract()
                    
                    text = soup.get_text(separator=NEWLINE_SEP, strip=True)
                    return ContentExtractor._clean_text(text)
                else:
                    console.print(f"[yellow]No Wayback Machine snapshot available for {url}[/yellow]")
                    
        except Exception as e:
            console.print(f"[dim]Wayback Machine extraction failed for {url}: {e}[/]")
        
        return None

    @staticmethod
    def _extract_from_google_cache(url: str, timeout: int) -> Optional[str]:
        """Extract content from Google Cache (less reliable)"""
        try:
            # Google Cache URL format
            cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(cache_url, headers=headers, timeout=timeout)
            
            if response.status_code == 200 and "No information is available" not in response.text:
                console.print(f"[green]Found Google Cache for {url}[/green]")
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    script.extract()
                
                text = soup.get_text(separator=NEWLINE_SEP, strip=True)
                return ContentExtractor._clean_text(text)
            else:
                console.print(f"[yellow]No Google Cache available for {url}[/yellow]")
                
        except Exception as e:
            console.print(f"[dim]Google Cache extraction failed for {url}: {e}[/]")
        
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean up extracted text"""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > 10000:
            text = text[:10000] + "..."
        return text.strip()
    
    @staticmethod
    def extract_sitemap_urls(domain: str) -> List[str]:
        """Extract URLs from sitemap for a given domain"""
        sitemap_urls = []
        
        try:
            import xml.etree.ElementTree as ET
            from urllib.parse import urljoin
            
            base_url = f"https://{domain}" if not domain.startswith('http') else domain
            
            # Try common sitemap locations
            potential_sitemaps = [
                urljoin(base_url, '/sitemap.xml'),
                urljoin(base_url, '/sitemap_index.xml'),
                urljoin(base_url, '/sitemaps.xml')
            ]
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            for sitemap_url in potential_sitemaps:
                try:
                    response = requests.get(sitemap_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        root = ET.fromstring(response.content)
                        
                        # Handle sitemap namespace
                        namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                        
                        # Extract URLs
                        for url_elem in root.findall('.//sitemap:url', namespaces):
                            loc_elem = url_elem.find('sitemap:loc', namespaces)
                            if loc_elem is not None:
                                url = loc_elem.text
                                # Filter for relevant pages
                                if ContentExtractor._is_business_relevant_url(url):
                                    sitemap_urls.append(url)
                        
                        # If we found URLs, break
                        if sitemap_urls:
                            break
                            
                except Exception as e:
                    console.print(f"[dim]Failed to fetch sitemap {sitemap_url}: {e}[/]")
                    continue
            
            # Also try robots.txt for sitemap references
            try:
                robots_url = urljoin(base_url, '/robots.txt')
                response = requests.get(robots_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    for line in response.text.split('\n'):
                        if line.strip().lower().startswith('sitemap:'):
                            sitemap_ref = line.split(':', 1)[1].strip()
                            try:
                                response = requests.get(sitemap_ref, headers=headers, timeout=10)
                                if response.status_code == 200:
                                    root = ET.fromstring(response.content)
                                    namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                                    
                                    for url_elem in root.findall('.//sitemap:url', namespaces):
                                        loc_elem = url_elem.find('sitemap:loc', namespaces)
                                        if loc_elem is not None:
                                            url = loc_elem.text
                                            if ContentExtractor._is_business_relevant_url(url):
                                                sitemap_urls.append(url)
                            except:
                                continue
            except:
                pass
            
        except Exception as e:
            console.print(f"[dim]Sitemap extraction failed for {domain}: {e}[/]")
        
        return list(set(sitemap_urls))  # Remove duplicates
    
    @staticmethod
    def _is_business_relevant_url(url: str) -> bool:
        """Check if a URL is relevant for business information"""
        business_keywords = [
            'about', 'contact', 'team', 'people', 'staff', 'leadership',
            'management', 'executives', 'careers', 'jobs', 'company',
            'locations', 'offices', 'phone', 'email', 'social'
        ]
        
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in business_keywords)
