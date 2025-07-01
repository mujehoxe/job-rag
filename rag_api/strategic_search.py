#!/usr/bin/env python3
"""
Strategic Search class for the RAG API.
"""

import os
import json
from typing import Dict, List

import g4f
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .keyword_extractor import KeywordExtractor
from search_utils import get_enhanced_search_results, SearchResultProcessor
from contact_api.utils import ContentExtractor

# Default configuration
# Increased max rounds to allow for deeper investigation
MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "5"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5")) # Reduced for more focused rounds
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "15"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.2"))

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
            self.search_rounds_context.append(f"\n--- Starting Search Round {round_count}/{self.max_rounds} ---")

            # Execute search for the current set of queries
            round_documents = self._execute_search_round(search_queries)
            if round_documents:
                all_documents.extend(round_documents)
                all_documents = SearchResultProcessor.filter_duplicates(all_documents)  # De-duplicate

            # Analyze results and decide next steps
            analysis = self._analyze_and_refine(query, all_documents, round_count)
            self.search_rounds_context.append(f"Round {round_count} AI Analysis: {analysis.get('thinking_process', 'N/A')}")

            # Display the AI's thinking process to the user
            console.print(Panel(analysis.get('thinking_process', 'No analysis provided.'), title=f"🤖 Round {round_count} Analysis", border_style="blue", expand=False))

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
        for search_query in valid_search_queries[:2]:  # Limit queries per round to reduce rate limits
            self.search_rounds_context.append(f"Searching for: '{search_query}'")
            try:
                results = self._search_and_extract(search_query)
                if results:
                    round_documents.extend(results)
                    self.search_rounds_context.append(f"Found {len(results)} results for '{search_query}'")
            except Exception as e:
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

            for doc in search_results:
                url = doc.get("href")
                if url and not url.endswith(('.pdf', '.xml', '.zip')):
                    try:
                        content = self.content_extractor.extract_from_url(url)
                        if content:
                            doc["content"] = content
                        else:
                            doc["content"] = doc.get("snippet", "")
                        processed_documents.append(doc)
                    except Exception as e:
                        console.print(f"[yellow]Error extracting content from {url}: {e}[/yellow]")
                        doc["content"] = doc.get("snippet", "")
                        processed_documents.append(doc)
                else:
                    doc["content"] = doc.get("snippet", "")
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
        doc_summary = []
        for i, doc in enumerate(documents[:10], 1):
            title = doc.get('title', 'No Title')
            snippet = doc.get('snippet', 'No Snippet')
            doc_summary.append(f"Doc {i}: {title}\nSnippet: {snippet}")

        summary_text = "\n\n".join(doc_summary) if doc_summary else "No documents found yet."

        system_prompt = (
            "You are a world-class investigative researcher. Your goal is to uncover social media links, LinkedIn profiles, and phone numbers for a given company or website. "
            "You will be given a user's query and a history of previous search results. Your task is to generate a list of new, highly-targeted search queries to find the missing information.\n\n"
            "**Instructions & Rules:**\n"
            "1. **Analyze the History:** Carefully review the previous search results. Do not repeat queries that have already failed or yielded no new information.\n"
            "2. **Be Specific:** Create very specific queries. Instead of 'company social media', use 'site:linkedin.com/company/ company-name' or 'company-name twitter'.\n"
            "3. **Use Advanced Operators:** Employ advanced search operators to narrow your search. Examples:\n"
            "   - `site:linkedin.com \"milestonehomesre\"` (Search only on LinkedIn)\n"
            "   - `milestonehomesre.com \"contact us\"` (Look for a contact page)\n"
            "   - `milestonehomesre.com phone number` (Directly search for a phone number)\n"
            "   - `inurl:facebook.com \"milestonehomesre\"` (Find a Facebook URL)\n\n"
            "4. **Iterate and Refine:** If initial broad searches fail, generate more creative and targeted queries. Think about what an expert human researcher would do next.\n"
            "5. **Format:** Return a list of queries, one per line. Do not add any other text or explanation.\n\n"
            "Based on the user's query and the history, generate the next set of search queries in the 'new_queries' field of the JSON."
            "The JSON output MUST contain three keys:\n"
            "1. \"thinking_process\": (string) Briefly explain your reasoning. Have you found the answer? If not, what information is missing? What is your plan for the next search round?\n"
            "2. \"sufficient\": (boolean) Set to true if you believe the documents contain enough information to comprehensively answer the user's query, otherwise false.\n"
            "3. \"new_queries\": (list of strings) If sufficient is false, provide a list of 3-5 new, specific, and diverse search queries to find the missing information. If sufficient is true, provide an empty list."
        )
        user_prompt = f"""Original User Query: "{query}"

Round {round_num} Analysis:
Documents Found So Far:
---
{summary_text}
---

Now, provide your analysis in the required JSON format.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = g4f.ChatCompletion.create(
                model=os.getenv("G4F_MODEL", "gpt-3.5-turbo"),
                messages=messages,
                timeout=30
            )
            # Clean the response to ensure it's valid JSON
            clean_response = response.strip().replace('`', '')
            if clean_response.startswith('json'):
                clean_response = clean_response[4:]
            
            analysis = json.loads(clean_response)
            return analysis
        except (json.JSONDecodeError, Exception) as e:
            console.print(f"[bold red]Error parsing AI analysis. Details: {e}[/bold red]")
            return {"thinking_process": "Could not analyze results due to an error.", "sufficient": False, "new_queries": [query]}

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
