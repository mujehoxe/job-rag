#!/usr/bin/env python3
"""
Strategic Search class for the RAG API with logical search ordering and visible AI analysis.
"""

import os
import json
import re
from typing import Dict, List, Set

import g4f
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .keyword_extractor import KeywordExtractor
from search_utils import get_enhanced_search_results, SearchResultProcessor
from contact_api.utils import ContentExtractor
from api_key_manager import APIKeyManager

# Default configuration
# Increased max rounds to allow for deeper business intelligence investigation
MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "5"))  # Reduced for faster evaluation while still being thorough
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "15"))  # Increased for more comprehensive results
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "20"))  # Longer timeout for better content extraction
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.15"))  # Lower threshold for business intel

console = Console()

# These will be initialized in rag_search.py and passed in
HAS_SENTENCE_TRANSFORMERS = False
text_splitter = None
sentence_transformer = None

class StrategicSearch:
    """
    Class to handle a strategic, iterative search process driven by AI analysis.
    """

    def __init__(
        self,
        max_rounds=MAX_SEARCH_ROUNDS,
        max_results=MAX_SEARCH_RESULTS,
        timeout=SEARCH_TIMEOUT,
        min_relevance_score=MIN_RELEVANCE_SCORE,
    ):
        """Initialize the strategic search"""
        self.max_rounds = max_rounds
        self.max_results = max_results
        self.timeout = timeout
        self.min_relevance_score = min_relevance_score
        self.keyword_extractor = KeywordExtractor()
        self.search_rounds_context = []
        self.cumulative_context = []  # Store context across rounds
        self.found_info_tracker = {  # Track what we've found to avoid duplicates
            'phone_numbers': set(),
            'emails': set(),
            'social_profiles': set(),
            'company_pages': set(),
            'key_people': set()
        }
        # Initialize the content extractor
        self.content_extractor = ContentExtractor()

    def strategic_search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a strategic, iterative search with AI analysis between rounds.
        """
        console.print(
            f"[bold]Search configuration: {self.max_rounds} rounds, {self.max_results} max results, {self.timeout}s timeout[/]"
        )
        self.search_rounds_context = [f"Starting strategic search for query: {query}"]

        keywords, entities = self.keyword_extractor.extract_keywords(query)
        self.search_rounds_context.append(f"Extracted keywords: {keywords}, Entities: {entities}")

        all_documents = []
        search_queries = self._generate_initial_queries(keywords, entities)
        if not search_queries:
            search_queries = [query]  # Fallback to raw query if no keywords found

        console.print(Panel.fit(
            "\n".join(f"- {q}" for q in search_queries),
            title="💡 Initial Search Strategy",
            border_style="green"
        ))

        for round_count in range(1, self.max_rounds + 1):
            console.print(f"\n[bold blue]🔍 Starting Search Round {round_count}/{self.max_rounds}[/bold blue]")
            self.search_rounds_context.append(f"\n--- Starting Search Round {round_count}/{self.max_rounds} ---")

            # Execute search for the current set of queries
            round_documents = self._execute_search_round(search_queries)
            if round_documents:
                # Filter out documents we already have before adding
                existing_urls = {doc.get('source_url') or doc.get('url') or doc.get('href') for doc in all_documents}
                new_documents = [doc for doc in round_documents 
                               if (doc.get('source_url') or doc.get('url') or doc.get('href')) not in existing_urls]
                
                if new_documents:
                    all_documents.extend(new_documents)
                    console.print(f"[green]📄 Added {len(new_documents)} new documents (total: {len(all_documents)})[/green]")
                else:
                    console.print(f"[yellow]⚠️ No new documents found this round (total: {len(all_documents)})[/yellow]")

            # Analyze results and decide next steps
            console.print(f"[yellow]🤖 Analyzing results from round {round_count}...[/yellow]")
            analysis = self._analyze_and_refine(query, all_documents, round_count)
            self.search_rounds_context.append(f"Round {round_count} AI Analysis: {analysis.get('thinking_process', 'N/A')}")

            if analysis.get('sufficient', False):
                console.print("[bold green]AI has determined the collected information is sufficient. Ending search.[/]")
                self.search_rounds_context.append("Information deemed sufficient.")
                break

            search_queries = analysis.get('new_queries', [])
            if not search_queries:
                console.print("[yellow]AI could not generate new queries. Ending search.[/yellow]")
                self.search_rounds_context.append("Could not generate new queries.")
                break

            if round_count == self.max_rounds:
                console.print("[yellow]Max search rounds reached.[/]")
                self.search_rounds_context.append("Max search rounds reached.")

        if not all_documents:
            console.print("[bold red]No documents found after all search rounds.[/]")
            return []

        console.print(f"\nTotal unique documents found: {len(all_documents)}")
        ranked_docs = self._rank_and_filter_documents(query, all_documents)
        self._display_search_results(ranked_docs)
        return ranked_docs

    def _execute_search_round(self, search_queries: List[str]) -> List[Dict[str, str]]:
        """Executes a single round of searching for a list of queries."""
        round_documents = []
        valid_search_queries = [q for q in search_queries if q.strip()]

        console.print(f"Executing {len(valid_search_queries)} search queries this round...")
        for i, search_query in enumerate(valid_search_queries[:3], 1):  # Execute 3 queries per round for better coverage
            console.print(f"[cyan]🔎 Query {i}/3: {search_query}[/cyan]")
            self.search_rounds_context.append(f"Searching for: '{search_query}'")
            try:
                results = self._search_and_extract(search_query)
                if results:
                    round_documents.extend(results)
                    console.print(f"[green]✅ Found {len(results)} results[/green]")
                    self.search_rounds_context.append(f"Found {len(results)} results for '{search_query}'")
                else:
                    console.print(f"[yellow]⚠️ No results found[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ Search error: {e}[/red]")
                self.search_rounds_context.append(f"Search error for '{search_query}': {e}")
        return round_documents

    def _search_and_extract(self, search_query: str) -> List[Dict[str, str]]:
        """
        Performs a web search for a single query and extracts content from the results.
        """
        processed_documents = []
        try:
            search_results = get_enhanced_search_results(
                search_query,
                max_results=self.max_results,
                timeout=self.timeout,
            )

            # Skip full content extraction for speed - use snippets instead
            for i, doc in enumerate(search_results, 1):
                url = doc.get("href") or doc.get("url")
                if url:
                    # Use snippet + title instead of full content extraction for speed
                    title = doc.get('title', 'No title')
                    snippet = doc.get('snippet', 'No snippet available')
                    
                    # Create content from available metadata
                    doc["content"] = f"URL: {url}\nTitle: {title}\nSnippet: {snippet}"
                    doc["source_url"] = url
                    
                    # Do selective content extraction for business-relevant domains AND external social media
                    should_extract = (
                        any(domain in url.lower() for domain in ['milestonehomesre.com']) or
                        any(platform in url.lower() for platform in ['linkedin.com', 'facebook.com', 'twitter.com', 'instagram.com'])
                    )
                    
                    if should_extract:
                        console.print(f"[dim]Quick extraction: {url[:50]}...[/dim]")
                        try:
                            # Quick text-only extraction with timeout
                            content = self._quick_text_extract(url)
                            if content and len(content.strip()) > 100:
                                doc["content"] = f"URL: {url}\nTitle: {title}\n\n{content[:1000]}"  # Limit content length
                        except Exception as e:
                            console.print(f"[yellow]Quick extraction failed for {url}: {e}[/yellow]")
                            # Keep the snippet-based content
                    
                    # Check if this is a business-relevant page
                    title = doc.get('title', '')
                    snippet = doc.get('snippet', '')
                    if self._is_business_relevant_page(url, title, snippet):
                        doc["business_relevant"] = True
                    
                    processed_documents.append(doc)

        except Exception as e:
            console.print(f"[bold red]Error during search for '{search_query}': {e}[/bold red]")

        return processed_documents

    def _generate_initial_queries(self, keywords: List[str], entities: List[str]) -> List[str]:
        """Generates a set of initial search queries from keywords and entities."""
        queries = set()
        combined = entities + keywords
        if not combined:
            return []

        # Check if this appears to be a company enrichment query
        is_company_enrichment = any(keyword.lower() in ['social', 'media', 'linkedin', 'phone', 'contact', 'email', 'whatsapp', 'decision', 'maker', 'ceo', 'founder'] for keyword in keywords)
        
        if is_company_enrichment and entities:
            # Generate specialized company enrichment queries
            main_entity = entities[0]  # Assume first entity is the company
            
            # Add domain-specific searches first (highest priority)
            potential_domains = [
                f'{main_entity.lower().replace(" ", "")}.com',
                f'{main_entity.lower().replace(" ", "")}.co',
                f'{main_entity.lower().replace(" ", "")}.io'
            ]
            for domain in potential_domains:
                queries.add(f'site:{domain} contact')
                queries.add(f'site:{domain} about')
            
            # LinkedIn company page (most likely to have good contact info)
            queries.add(f'site:linkedin.com/company "{main_entity}"')
            queries.add(f'site:linkedin.com "{main_entity}"')
            
            # Start with broader, more likely to succeed queries
            queries.add(f'{main_entity}')  # Simple company name search
            queries.add(f'"{main_entity}" contact information')
            queries.add(f'"{main_entity}" phone email')
            
            # External social media searches (focus on finding profiles on other platforms)
            queries.add(f'site:twitter.com "{main_entity}"')
            queries.add(f'site:facebook.com "{main_entity}"')
            queries.add(f'site:instagram.com "{main_entity}"')
            
            # Leadership searches (valuable for contact info)
            queries.add(f'"{main_entity}" CEO founder')
            queries.add(f'"{main_entity}" leadership team')
            queries.add(f'"{main_entity}" management contact')
        else:
            # Standard query generation
            queries.add(" ".join(combined))
            for entity in entities:
                queries.add(f'{entity} contact information')
                queries.add(f'{entity} social media profiles')
            for keyword in keywords:
                if keyword not in entities:
                    queries.add(f'{" ".join(entities)} {keyword}')

        return list(queries)

    def _analyze_and_refine(self, query: str, documents: List[Dict[str, str]], round_num: int) -> Dict:
        """
        Uses an AI to analyze collected documents, check for sufficiency,
        and generate new queries if needed.
        """
        # Analyze what contact information has been found so far
        found_info = self._analyze_found_information(documents)
        
        doc_summary = []
        for i, doc in enumerate(documents[:10], 1):
            title = doc.get('title', 'No Title')
            snippet = doc.get('snippet', 'No Snippet')
            url = doc.get('source_url') or doc.get('href', '')
            doc_summary.append(f"Doc {i}: {title}\nURL: {url}\nSnippet: {snippet}")

        summary_text = "\n\n".join(doc_summary) if doc_summary else "No documents found yet."

        # Get available search engines info
        api_manager = APIKeyManager()
        available_engines = [engine for engine, available in api_manager.check_available_engines().items() if available]
        
        system_prompt = (
            "You are a business intelligence researcher. You must analyze search results and respond with ONLY valid JSON.\n\n"
            
            "CRITICAL: Your response must be ONLY a valid JSON object. Do not include any text before or after the JSON.\n\n"
            
            "Required JSON format:\n"
            "{\n"
            "  \"thinking_process\": \"string describing what was found and what's still needed\",\n"
            "  \"company_industry\": \"detected industry or 'unknown'\",\n"
            "  \"company_location\": \"detected city/region or 'unknown'\",\n"
            "  \"sufficient\": true or false,\n"
            "  \"new_queries\": [\"query1\", \"query2\", \"query3\"]\n"
            "}\n\n"
            
            f"Available Search Engines: {', '.join(available_engines)}\n"
            "Search Engine Compatibility:\n"
            "- Brave/Google: Supports site: operators, quoted phrases, boolean operators\n"
            "- Serper: Supports site: operators, quoted phrases\n"
            "- DuckDuckGo: Limited operator support, prefers simple queries\n"
            "- Bing: Supports most operators but prefers natural language\n\n"
            
            "Rules:\n"
            "- thinking_process: Single line, no line breaks, describe findings and strategy\n"
            "- company_industry: Identify industry (real estate, tech, healthcare, etc.) from content\n"
            "- company_location: Identify main city/region of operations from content\n"
            "- sufficient: true only if you have phone, email, social media profiles (not posts), AND key people\n"
            "- new_queries: Create 3 search-engine optimized queries for the specific company\n"
            "- For social media posts found, generate queries to find the actual profile pages\n"
            "- Use company_industry and company_location to disambiguate from similar named companies\n"
            "- NEVER search for generic topics like 'how to find contact info'\n"
            "- Focus ONLY on the specific company from the original query\n\n"
            
            "Query Optimization Tips:\n"
            "- Use site: operators for targeted searches on professional platforms\n"
            "- Include industry/location context to avoid confusion with similarly named companies\n"
            "- For social media profiles: search for 'site:platform.com company-name' not individual posts\n\n"
            
            "IMPORTANT: Return ONLY the JSON object, nothing else."
        )
        user_prompt = f"""Original User Query: "{query}"

Round {round_num} Analysis:

CONTACT INFORMATION FOUND SO FAR:
{found_info}

DOCUMENTS ANALYZED:
---
{summary_text}
---

Based on the original query and what information is still missing, provide your analysis in JSON format.
Focus ONLY on the company mentioned in the original query. Do NOT search for generic topics.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Use local g4f API for analysis
            import requests
            
            api_base = os.getenv("LOCAL_API_BASE", "http://localhost:8080/v1")
            model = os.getenv("G4F_MODEL", "llama-4-scout")
            
            # Direct use of the local g4f API with reduced timeout
            response = requests.post(
                f"{api_base}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1500
                },
                timeout=10  # Reduced timeout to prevent hanging
            )
            response.raise_for_status()
            result = response.json()
            response_content = result["choices"][0]["message"]["content"]
            
            console.print(f"[dim]Raw AI response received: {len(response_content)} chars[/dim]")
            
            # Debug: show first part of response
            console.print(f"[dim]Response preview: {response_content[:200]}...[/dim]")
            if len(response_content) > 400:
                console.print(f"[dim]Response end: ...{response_content[-100:]}[/dim]")

            # Clean the response to ensure it's valid JSON
            clean_response = self._robust_json_clean(response_content)
            
            console.print(f"[dim]Cleaned response: {len(clean_response)} chars[/dim]")
            console.print(f"[dim]Cleaned preview: {clean_response[:200]}...[/dim]")

            analysis = json.loads(clean_response)

            # Display comprehensive analysis
            thinking = analysis.get('thinking_process', 'No thinking process provided')
            industry = analysis.get('company_industry', 'unknown')
            location = analysis.get('company_location', 'unknown')
            sufficient = analysis.get('sufficient', False)
            new_queries = analysis.get('new_queries', [])

            console.print(f"╭─────────── 🤖 Round {round_num} Analysis ────────────╮")
            console.print(f"│ [bold]Thinking:[/bold] {thinking}")
            console.print(f"│ [bold]Industry:[/bold] {industry}")
            console.print(f"│ [bold]Location:[/bold] {location}")
            console.print(f"│ [bold]Sufficient:[/bold] {sufficient}")
            if new_queries:
                console.print(f"│ [bold]Next Queries:[/bold]")
                for i, query in enumerate(new_queries[:3], 1):
                    console.print(f"│   {i}. {query}")
            console.print("╰────────────────────────────────────────────────────╯")
            
            return analysis

        except json.JSONDecodeError as json_err:
            console.print(f"[red]JSON parsing failed: {json_err}[/red]")
            console.print(f"[red]Raw response: {response_content[:200]}...[/red]")
            console.print(f"[red]Cleaned response: {clean_response[:200]}...[/red]")
            # Return a minimal valid response to continue the search
            return {
                "thinking_process": "JSON parsing failed",
                "sufficient": False,
                "new_queries": [f"{query.split()[0] if query.split() else 'company'} contact"]
            }
            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]g4f API request failed: {e}[/red]")
            # Extract company name from the original query better
            company_name = self._extract_company_name_from_query(query)
            
            # Use rule-based analysis as fallback
            return self._rule_based_analysis(company_name, found_info, round_num, query)
            
        except json.JSONDecodeError as e:
            console.print(f"[red]JSON parsing failed: {e}[/red]")
            console.print(f"[dim yellow]Skipping AI analysis for this round due to parsing issues[/dim yellow]")
            
            # Try to extract thinking process from raw response even if JSON parsing failed
            thinking_process = "Could not parse AI analysis properly."
            if "thinking_process" in response_content:
                try:
                    # Try to extract thinking process text
                    lines = response_content.split('\n')
                    for i, line in enumerate(lines):
                        if 'thinking_process' in line.lower() or 'analysis' in line.lower():
                            # Get next few lines as thinking process
                            thinking_lines = lines[i:i+5]
                            thinking_process = ' '.join(thinking_lines).replace('"', '').replace(',', '')
                            break
                except:
                    pass
            
            # Generate fallback queries based on what we haven't found yet
            company_name = query.split()[0] if query else "company"
            
            if round_num <= 2:
                fallback_queries = [
                    f'site:linkedin.com "{company_name}"',
                    f'site:facebook.com "{company_name}"',
                    f'"{company_name}" phone number'
                ]
            else:
                fallback_queries = [
                    f'site:yellowpages.com "{company_name}"',
                    f'site:yelp.com "{company_name}"',
                    f'"{company_name}" contact information'
                ]
            
            console.print(f"╭─────────── 🤖 Round {round_num} Analysis (Fallback) ────────────╮")
            console.print(f"│ {thinking_process[:60]}{'...' if len(thinking_process) > 60 else ''} │")
            console.print("╰────────────────────────────────────────────────────╯")
            
            return {
                "thinking_process": "Could not analyze results due to parsing error. Using fallback queries.",
                "sufficient": False,
                "new_queries": fallback_queries
            }
        except Exception as e:
            console.print(f"[bold red]Error in AI analysis: {e}[/bold red]")
            return {
                "thinking_process": "Could not analyze results due to an error.",
                "sufficient": False,
                "new_queries": [f'{query} contact information']
            }

    def _rank_and_filter_documents(
        self, query: str, documents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Rank and filter the collected documents
        """
        if not documents:
            return []

        unique_documents = SearchResultProcessor.filter_duplicates(documents)

        if HAS_SENTENCE_TRANSFORMERS and sentence_transformer:
            try:
                query_embedding = sentence_transformer.encode(query)
                for doc in unique_documents:
                    doc_text = f"{doc.get('title', '')} {doc.get('snippet', '')}"
                    doc_embedding = sentence_transformer.encode(doc_text)
                    similarity = sentence_transformer.util.cos_sim(query_embedding, doc_embedding)
                    doc["relevance_score"] = float(similarity[0][0])

                filtered_docs = [
                    doc
                    for doc in unique_documents
                    if doc.get("relevance_score", 0) > self.min_relevance_score
                ]

                ranked_docs = sorted(
                    filtered_docs,
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True,
                )
                return ranked_docs

            except Exception as e:
                console.print(f"[yellow]Error in semantic ranking: {e}[/yellow]")
                return SearchResultProcessor.rank_results(unique_documents, query)
        else:
            return SearchResultProcessor.rank_results(unique_documents, query)

    def _display_search_results(self, documents: List[Dict[str, str]]):
        """Display search results in a formatted table"""
        if not documents:
            return

        table = Table(title="Top 5 Search Results")
        table.add_column("Title", style="cyan", no_wrap=True)
        table.add_column("URL", style="magenta")
        table.add_column("Relevance", style="green")

        for doc in documents[:5]:
            relevance = f"{doc.get('relevance_score', 0):.2f}" if doc.get('relevance_score') is not None else "N/A"
            table.add_row(doc.get('title', 'No Title'), doc.get('href', ''), relevance)

        console.print(table)
    
    def _is_business_relevant_page(self, url: str, title: str, content: str) -> bool:
        """Check if a page is relevant for business intelligence"""
        business_keywords = [
            'about', 'contact', 'team', 'people', 'staff', 'employees', 'leadership',
            'management', 'executives', 'founders', 'board', 'directors', 'careers',
            'jobs', 'company', 'organization', 'mission', 'vision', 'history',
            'locations', 'offices', 'headquarters', 'address', 'phone', 'email'
        ]
        
        # Check URL path
        url_lower = url.lower()
        if any(keyword in url_lower for keyword in business_keywords):
            return True
        
        # Check title
        title_lower = title.lower() if title else ""
        if any(keyword in title_lower for keyword in business_keywords):
            return True
        
        # Check content (first 500 chars)
        content_sample = content[:500].lower() if content else ""
        keyword_count = sum(1 for keyword in business_keywords if keyword in content_sample)
        
        return keyword_count >= 2  # At least 2 business keywords in content
    
    def _is_main_domain_page(self, url: str) -> bool:
        """Check if this is a main domain page (not subdomain or deep path)"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Check if it's a main domain (not subdomain)
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) > 2 and domain_parts[0] not in ['www', 'm', 'mobile']:
                return False  # Likely a subdomain
            
            # Check if it's not too deep in the path structure
            path_parts = [p for p in parsed.path.split('/') if p]
            return len(path_parts) <= 2  # Root or one level deep
            
        except Exception:
            return False
    
    def _extract_sitemap_urls(self, base_url: str) -> List[str]:
        """Extract relevant URLs from sitemap for future searches"""
        relevant_urls = []
        
        try:
            from urllib.parse import urljoin, urlparse
            import requests
            import xml.etree.ElementTree as ET
            
            parsed_base = urlparse(base_url)
            base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
            
            # Common sitemap locations
            sitemap_urls = [
                urljoin(base_domain, '/sitemap.xml'),
                urljoin(base_domain, '/sitemap_index.xml'),
                urljoin(base_domain, '/robots.txt')  # Check robots.txt for sitemap
            ]
            
            for sitemap_url in sitemap_urls:
                try:
                    response = requests.get(sitemap_url, timeout=10)
                    if response.status_code == 200:
                        
                        # If it's robots.txt, look for sitemap references
                        if 'robots.txt' in sitemap_url:
                            for line in response.text.split('\n'):
                                if line.strip().lower().startswith('sitemap:'):
                                    actual_sitemap = line.split(':', 1)[1].strip()
                                    sitemap_urls.append(actual_sitemap)
                            continue
                        
                        # Parse XML sitemap
                        try:
                            root = ET.fromstring(response.content)
                            
                            # Handle different sitemap formats
                            namespaces = {
                                'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'
                            }
                            
                            # Look for URL entries
                            for url_elem in root.findall('.//sitemap:url', namespaces):
                                loc_elem = url_elem.find('sitemap:loc', namespaces)
                                if loc_elem is not None:
                                    url = loc_elem.text
                                    if self._is_business_relevant_page(url, "", ""):
                                        relevant_urls.append(url)
                            
                            # Also check for sitemap index
                            for sitemap_elem in root.findall('.//sitemap:sitemap', namespaces):
                                loc_elem = sitemap_elem.find('sitemap:loc', namespaces)
                                if loc_elem is not None:
                                    # Recursively check sub-sitemaps (limited to avoid infinite loops)
                                    if len(relevant_urls) < 20:  # Limit total URLs
                                        sub_urls = self._extract_sitemap_urls(loc_elem.text)
                                        relevant_urls.extend(sub_urls)
                            
                            break  # Found a working sitemap
                            
                        except ET.ParseError:
                            continue  # Try next sitemap URL
                            
                except requests.RequestException:
                    continue  # Try next sitemap URL
            
            # Prioritize business-relevant pages
            business_priority_keywords = ['about', 'contact', 'team', 'people', 'leadership']
            priority_urls = []
            other_urls = []
            
            for url in relevant_urls:
                if any(keyword in url.lower() for keyword in business_priority_keywords):
                    priority_urls.append(url)
                else:
                    other_urls.append(url)
            
            # Return prioritized list
            return priority_urls + other_urls
            
        except Exception as e:
            console.print(f"[yellow]Sitemap extraction failed: {e}[/yellow]")
            return []
    
    def _analyze_found_information(self, documents: List[Dict[str, str]]) -> str:
        """Analyze what contact information has been found in the documents"""
        found_info = {
            'phones': [],
            'emails': [],
            'social_media': [],
            'linkedin_profiles': [],
            'company_website': None,
            'key_people': []
        }
        
        # Simple regex patterns for quick analysis
        phone_pattern = r'\+?[\d\s\-\(\)]{7,}'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        linkedin_pattern = r'linkedin\.com/(?:in/|company/)([a-zA-Z0-9-]+)'
        
        for doc in documents:
            content = doc.get('content', '') + ' ' + doc.get('snippet', '')
            url = doc.get('source_url') or doc.get('href', '')
            
            # Check if this is the company's main website
            if url and any(domain in url.lower() for domain in ['milestonehomesre.com', 'milestone']):
                found_info['company_website'] = url
            
            # Extract phones
            import re
            phones = re.findall(phone_pattern, content)
            for phone in phones:
                if len(phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')) >= 7:
                    found_info['phones'].append(phone.strip())
            
            # Extract emails
            emails = re.findall(email_pattern, content, re.IGNORECASE)
            found_info['emails'].extend(emails)
            
            # Extract LinkedIn profiles
            linkedin_matches = re.findall(linkedin_pattern, content, re.IGNORECASE)
            for match in linkedin_matches:
                found_info['linkedin_profiles'].append(f"linkedin.com/{match}")
            
            # Check for social media mentions
            social_platforms = ['facebook', 'twitter', 'instagram', 'youtube', 'tiktok']
            for platform in social_platforms:
                if platform in content.lower():
                    found_info['social_media'].append(platform)
            
            # Look for key people (simple detection)
            executive_titles = ['CEO', 'founder', 'president', 'director', 'manager']
            for title in executive_titles:
                if title.lower() in content.lower():
                    found_info['key_people'].append(f"Found {title} mention")
        
        # Remove duplicates
        for key in found_info:
            if isinstance(found_info[key], list):
                found_info[key] = list(set(found_info[key]))
        
        # Create summary
        summary = []
        if found_info['company_website']:
            summary.append(f"✅ Company website: {found_info['company_website']}")
        else:
            summary.append("❌ Company website: Not found")
            
        if found_info['phones']:
            summary.append(f"✅ Phone numbers: {len(found_info['phones'])} found")
        else:
            summary.append("❌ Phone numbers: Not found")
            
        if found_info['emails']:
            summary.append(f"✅ Email addresses: {len(found_info['emails'])} found")
        else:
            summary.append("❌ Email addresses: Not found")
            
        if found_info['linkedin_profiles']:
            summary.append(f"✅ LinkedIn profiles: {len(found_info['linkedin_profiles'])} found")
        else:
            summary.append("❌ LinkedIn profiles: Not found")
            
        if found_info['social_media']:
            summary.append(f"✅ Social media: {', '.join(set(found_info['social_media']))} mentioned")
        else:
            summary.append("❌ Social media: Not found")
            
        if found_info['key_people']:
            summary.append(f"✅ Key people: {len(found_info['key_people'])} mentions found")
        else:
            summary.append("❌ Key people: Not found")
        
        return '\n'.join(summary)

    def _robust_json_clean(self, response_content: str) -> str:
        """Robust JSON cleaning to handle common AI response formatting issues"""
        import re
        
        try:
            clean_response = response_content.strip()
            
            # Remove markdown code blocks (more robust)
            if '```json' in clean_response:
                start = clean_response.find('```json') + 7
                end = clean_response.find('```', start)
                if end != -1:
                    clean_response = clean_response[start:end].strip()
            elif '```' in clean_response:
                start = clean_response.find('```') + 3
                end = clean_response.find('```', start)
                if end != -1:
                    clean_response = clean_response[start:end].strip()
            
            # Find JSON boundaries more carefully
            json_start = clean_response.find('{')
            json_end = clean_response.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                clean_response = clean_response[json_start:json_end + 1]
            else:
                # If no proper JSON boundaries found, return minimal valid JSON
                return '{"thinking_process": "Could not parse AI response", "sufficient": false, "new_queries": []}'
            
            # Handle common JSON formatting issues step by step
            
            # 1. Fix unescaped backslashes first
            clean_response = clean_response.replace('\\', '\\\\')
            
            # 2. Fix unescaped newlines and control characters ONLY within string values
            # Use regex to find strings and escape newlines only within them
            def escape_string_content(match):
                string_content = match.group(1)
                escaped = string_content.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                return f'"{escaped}"'
            
            # Match quoted strings and escape their contents
            clean_response = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', escape_string_content, clean_response)
            
            # 3. Fix trailing commas before closing braces/brackets
            clean_response = re.sub(r',(\s*[}\]])', r'\1', clean_response)
            
            # 4. Fix missing quotes around property names
            clean_response = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', clean_response)
            
            # 5. Validate basic structure
            if not clean_response.startswith('{') or not clean_response.endswith('}'):
                return '{"thinking_process": "Malformed JSON structure", "sufficient": false, "new_queries": []}'
            
            # 6. Try to parse and if it fails, return a fallback
            try:
                json.loads(clean_response)
                return clean_response
            except json.JSONDecodeError:
                # Last resort: construct a basic valid JSON
                return '{"thinking_process": "JSON parsing failed after cleaning", "sufficient": false, "new_queries": []}'
                
        except Exception as e:
            console.print(f"[red]Error in JSON cleaning: {e}[/red]")
            return '{"thinking_process": "Error during JSON cleaning", "sufficient": false, "new_queries": []}'
    
    def _quick_text_extract(self, url: str) -> str:
        """Quick text-only extraction with timeout and no media processing"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Quick request with short timeout
            response = requests.get(url, timeout=5, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Parse HTML and extract text only
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract text from main content areas
            text_content = ""
            
            # Look for main content containers
            main_containers = soup.find_all(['main', 'article', 'section', 'div'], 
                                          class_=lambda x: x and any(keyword in x.lower() for keyword in 
                                                                   ['content', 'main', 'article', 'text', 'body']))
            
            if main_containers:
                for container in main_containers[:3]:  # Limit to first 3 containers
                    text_content += container.get_text(separator=' ', strip=True) + " "
            else:
                # Fallback: get all text from body
                body = soup.find('body')
                if body:
                    text_content = body.get_text(separator=' ', strip=True)
            
            # Clean up text
            text_content = ' '.join(text_content.split())  # Remove extra whitespace
            return text_content[:2000] if text_content else ""  # Limit to 2000 chars
            
        except Exception as e:
            console.print(f"[dim red]Quick extraction error: {e}[/dim red]")
            return ""
    
    def _extract_company_name_from_query(self, query: str) -> str:
        """Extract company name from query using the same logic as keyword extractor"""
        import re
        
        query_lower = query.lower().strip()
        
        # Pattern: "Find contact information for [COMPANY]"
        company_patterns = [
            r"find\s+contact\s+information\s+(?:and\s+\w+\s+)*for\s+(.+?)(?:\s+(?:payment|communication|data|real\s+estate|monitoring|platform|processor|api|service).*)?$",
            r"find\s+.*?\s+for\s+(.+?)(?:\s+(?:payment|communication|data|real\s+estate|monitoring|platform|processor|api|service).*)?$",
            r"(?:get|find|locate|search)\s+.*?\s+(?:for|about|of)\s+(.+?)(?:\s+(?:company|corporation|inc|llc|ltd).*)?$"
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, query_lower)
            if match:
                potential_company = match.group(1).strip()
                # Clean up the company name
                potential_company = re.sub(r'\s+(?:payment|communication|data|real\s+estate|monitoring|platform|processor|api|service|company|corporation|inc|llc|ltd).*$', '', potential_company)
                potential_company = potential_company.strip()
                if potential_company and len(potential_company) > 1:
                    return potential_company
        
        # Fallback: look for company name after "for"
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() == 'for' and i + 1 < len(words):
                company_words = []
                for j in range(i + 1, len(words)):
                    if words[j].lower() in ['payment', 'communication', 'data', 'real', 'estate', 'monitoring', 'platform', 'processor', 'api', 'service', 'company', 'corporation']:
                        break
                    company_words.append(words[j])
                if company_words:
                    return ' '.join(company_words)
        
        return query.split()[0] if query.split() else "company"
    
    def _rule_based_analysis(self, company_name: str, found_info: str, round_num: int, query: str) -> Dict:
        """Rule-based analysis as fallback when AI analysis fails"""
        
        # Check what we have found
        has_phone = "✅ Phone numbers:" in found_info and "Not found" not in found_info.split("Phone numbers:")[1].split("\n")[0]
        has_email = "✅ Email addresses:" in found_info and "Not found" not in found_info.split("Email addresses:")[1].split("\n")[0]
        has_linkedin = "✅ LinkedIn profiles:" in found_info and "Not found" not in found_info.split("LinkedIn profiles:")[1].split("\n")[0]
        has_social = "✅ Social media:" in found_info and "Not found" not in found_info.split("Social media:")[1].split("\n")[0]
        has_website = "✅ Company website:" in found_info and "Not found" not in found_info.split("Company website:")[1].split("\n")[0]
        
        # Determine if we have sufficient information
        sufficient = has_email and (has_phone or has_linkedin) and has_social
        
        # Generate new queries based on what's missing and round number
        new_queries = []
        
        if round_num <= 2:
            # Early rounds: Focus on main company presence
            if not has_website:
                new_queries.append(f'site:{company_name.lower().replace(" ", "")}.com')
                new_queries.append(f'"{company_name}" official website')
            if not has_linkedin:
                new_queries.append(f'site:linkedin.com/company "{company_name}"')
                
        elif round_num <= 4:
            # Mid rounds: Focus on missing contact info
            if not has_email:
                new_queries.append(f'"{company_name}" contact email')
                new_queries.append(f'"{company_name}" support email')
            if not has_phone:
                new_queries.append(f'"{company_name}" phone number contact')
                new_queries.append(f'"{company_name}" customer service phone')
                
        elif round_num <= 6:
            # Later rounds: Social media and broader searches
            if not has_social:
                new_queries.append(f'site:twitter.com "{company_name}"')
                new_queries.append(f'site:facebook.com "{company_name}"')
            new_queries.append(f'"{company_name}" headquarters address')
            
        else:
            # Final rounds: Exhaustive searches
            new_queries.extend([
                f'site:instagram.com "{company_name}"',
                f'site:youtube.com "{company_name}"',
                f'"{company_name}" leadership team contact'
            ])
        
        # Ensure we always have at least 3 queries
        while len(new_queries) < 3:
            if round_num % 2 == 0:
                new_queries.append(f'"{company_name}" about contact')
            else:
                new_queries.append(f'"{company_name}" business information')
        
        thinking = f"Round {round_num}: Missing " + ", ".join([
            "phone" if not has_phone else "",
            "email" if not has_email else "",
            "LinkedIn" if not has_linkedin else "",
            "social media" if not has_social else ""
        ]).strip(", ")
        
        console.print(f"╭─────────── 🤖 Round {round_num} Analysis (Rule-Based) ────────────╮")
        console.print(f"│ [bold]Status:[/bold] {thinking}")
        console.print(f"│ [bold]Sufficient:[/bold] {sufficient}")
        console.print(f"│ [bold]Next Queries:[/bold]")
        for i, query in enumerate(new_queries[:3], 1):
            console.print(f"│   {i}. {query}")
        console.print("╰────────────────────────────────────────────────────────────────╯")
        
        return {
            "thinking_process": thinking,
            "company_industry": "unknown",
            "company_location": "unknown", 
            "sufficient": sufficient,
            "new_queries": new_queries[:3]
        }
