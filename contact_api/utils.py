#!/usr/bin/env python3
"""
Content extraction utilities.
"""

import re
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

# Define constants to avoid backslash in f-string expressions
NEWLINE_SEP = "\n"


class ContentExtractor:
    """Extract content from different types of websites"""

    @staticmethod
    def extract_from_url(url: str, timeout: int = 10) -> Optional[str]:
        """Extract content from a URL based on its domain"""
        try:
            domain = urlparse(url).netloc

            # Use specific extractors for known domains
            if "wikipedia.org" in domain:
                return ContentExtractor._extract_wikipedia(url, timeout)
            elif "github.com" in domain:
                return ContentExtractor._extract_github(url, timeout)
            elif "stackoverflow.com" in domain or "stackexchange.com" in domain:
                return ContentExtractor._extract_stackoverflow(url, timeout)
            elif "arxiv.org" in domain:
                return ContentExtractor._extract_arxiv(url, timeout)
            elif "medium.com" in domain or "towardsdatascience.com" in domain:
                return ContentExtractor._extract_medium(url, timeout)
            else:
                # Generic extractor for other sites
                return ContentExtractor._extract_generic(url, timeout)
        except Exception as e:
            console.print(f"[dim]Error extracting from {url}: {e}[/]")
            return None

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
    def _clean_text(text: str) -> str:
        """Clean up extracted text"""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > 10000:
            text = text[:10000] + "..."
        return text.strip()
