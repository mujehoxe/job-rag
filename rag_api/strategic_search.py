#!/usr/bin/env python3
"""
Strategic Search class for the RAG API.
"""

import os
import json
from typing import Dict, List

import g4f
import numpy as np
from rich.console import Console
from rich.table import Table

from .keyword_extractor import KeywordExtractor
from search_utils import get_enhanced_search_results, SearchResultProcessor
from contact_api.utils import ContentExtractor

# Default configuration
MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "3"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "15"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.2"))

console = Console()

# These will be initialized in rag_search.py and passed in
HAS_SENTENCE_TRANSFORMERS = False
text_splitter = None
sentence_transformer = None

class StrategicSearch:
    """Class to handle strategic, iterative search with analysis between rounds"""

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

        # Store context from search rounds
        self.search_rounds_context = []

    def strategic_search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a strategic, iterative search with analysis between rounds

        Args:
            query: User query

        Returns:
            List of relevant documents
        """
        console.print(
            f"[bold]Search configuration: {self.max_rounds} rounds, {self.max_results} max results, {self.timeout}s timeout[/]"
        )

        # Reset search context
        self.search_rounds_context = []
        self.search_rounds_context.append(
            f"Starting strategic search for query: {query}"
        )

        # Extract initial keywords and entities
        keywords, entities = self.keyword_extractor.extract_keywords(query)

        if keywords:
            self.search_rounds_context.append(
                f"Extracted keywords: {', '.join(keywords)}"
            )
        if entities:
            self.search_rounds_context.append(
                f"Extracted entities: {', '.join(entities)}"
            )

        # Initialize variables
        all_documents = []
        round_count = 0
        sufficient = False

        # Generate initial search queries
        search_queries = self._generate_search_queries(query, keywords, entities)
        self.search_rounds_context.append(
            f"Generated {len(search_queries)} initial search queries"
        )

        # Main search loop
        while round_count < self.max_rounds and not sufficient:
            round_count += 1
            self.search_rounds_context.append(
                f"\nStarting search round {round_count}/{self.max_rounds}"
            )

            # Execute search queries for this round
            round_documents = []
            for search_query in search_queries[:3]:  # Limit to top 3 queries per round
                self.search_rounds_context.append(f"Searching for: {search_query}")

                # Search and extract content
                try:
                    results = self._search_and_extract(search_query)
                    if results:
                        round_documents.extend(results)
                        self.search_rounds_context.append(
                            f"Found {len(results)} results"
                        )
                    else:
                        self.search_rounds_context.append("No results found")
                except Exception as e:
                    self.search_rounds_context.append(f"Search error: {str(e)}")

            # Add documents from this round to all documents
            all_documents.extend(round_documents)

            # If we have documents, analyze if we have sufficient information
            if round_documents:
                # Display the results from this round
                self._display_search_results(round_documents)

                # Check if we have sufficient information
                sufficient = self._analyze_sufficiency(query, all_documents)

                if sufficient:
                    self.search_rounds_context.append(
                        "Analysis: Sufficient information found"
                    )
                    break
                else:
                    # Generate new search queries based on what we've learned
                    search_queries = self._refine_search_strategy(query, all_documents)
                    self.search_rounds_context.append(
                        f"Analysis: Need more information. Refined search strategy with {len(search_queries)} new queries"
                    )
            else:
                self.search_rounds_context.append("No documents found in this round")
                # If no documents found, try a different strategy
                search_queries = self._generate_fallback_queries(query)
                self.search_rounds_context.append(
                    f"Generated {len(search_queries)} fallback queries"
                )

        # Final processing of all documents
        if all_documents:
            # Rank and filter all documents
            final_documents = self._rank_and_filter_documents(query, all_documents)
            self.search_rounds_context.append(
                f"Final processing: {len(final_documents)} documents after ranking and filtering"
            )
            return final_documents
        else:
            self.search_rounds_context.append("No relevant documents found after all rounds")
            return []

    def _generate_search_queries(
        self, query: str, keywords: List[str], entities: List[str]
    ) -> List[str]:
        """
        Generate search queries based on keywords and entities

        Args:
            query: Original user query
            keywords: List of extracted keywords
            entities: List of extracted entities

        Returns:
            List of search queries
        """
        queries = []

        # Always include original query in first round
        queries.append(query)

        # Domain-specific queries
        domain_keywords = [
            kw for kw in keywords if ".com" in kw or ".org" in kw or ".net" in kw
        ]

        if domain_keywords:
            # Enhanced contact information queries
            for domain in domain_keywords:
                queries.append(f"{domain} contact information")
                queries.append(f"{domain} contact us")
                queries.append(f"{domain} phone number")
                queries.append(f"{domain} email address")
                queries.append(f"{domain} headquarters location")
                queries.append(f"{domain} office address")
                queries.append(f"{domain} about us")
                queries.append(f"{domain} team")
                queries.append(f"{domain} management")
                queries.append(f"{domain} staff directory")

        # Social media queries
        social_media_keywords = [
            kw
            for kw in keywords
            if "social media" in kw.lower()
            or "linkedin" in kw.lower()
            or "facebook" in kw.lower()
            or "twitter" in kw.lower()
            or "instagram" in kw.lower()
        ]

        if social_media_keywords or domain_keywords:
            # If we have domain keywords, add social media queries for them
            for domain in domain_keywords or [query]:
                queries.append(f"{domain} social media accounts")
                queries.append(f"{domain} LinkedIn")
                queries.append(f"{domain} Facebook")
                queries.append(f"{domain} Twitter")
                queries.append(f"{domain} Instagram")
                queries.append(f"{domain} YouTube")
                queries.append(f"{domain} social profiles")

        # People queries
        people_keywords = [
            kw
            for kw in keywords
            if "person" in kw.lower()
            or "people" in kw.lower()
            or "employee" in kw.lower()
            or "staff" in kw.lower()
            or "team" in kw.lower()
            or "management" in kw.lower()
            or "leadership" in kw.lower()
        ]

        if people_keywords or domain_keywords:
            # If we have domain keywords, add people queries for them
            for domain in domain_keywords or [query]:
                queries.append(f"{domain} employees")
                queries.append(f"{domain} staff")
                queries.append(f"{domain} team")
                queries.append(f"{domain} management")
                queries.append(f"{domain} leadership")
                queries.append(f"{domain} CEO")
                queries.append(f"{domain} founder")
                queries.append(f"{domain} director")

        # Add queries for missing aspects
        for aspect in entities:
            queries.append(f"{query} {aspect}")

        # Ensure we have a reasonable number of queries
        if len(queries) > 10:
            return queries[:10]  # Limit to 10 queries per round
        elif len(queries) < 2:
            # Fallback to keyword combinations if we don't have enough queries
            keywords = keywords + entities
            if len(keywords) >= 3:
                for i in range(
                    min(len(keywords) - 2, 3)
                ):  # Add up to 3 keyword combinations
                    queries.append(f"{keywords[i]} {keywords[i+1]} {keywords[i+2]}")

        return list(set(queries))  # Remove duplicates

    def _search_and_extract(self, search_query: str) -> List[Dict[str, str]]:
        """
        Search for documents and extract content

        Args:
            search_query: Search query

        Returns:
            List of extracted documents
        """
        results = get_enhanced_search_results(
            search_query,
            max_results=self.max_results,
            timeout=self.timeout,
        )

        processed_documents = []
        for doc in results:
            url = doc.get("url", "")
            if url:
                # Add search query that found this document
                doc["source_query"] = search_query

                # Try to get content
                try:
                    content = ContentExtractor.extract_from_url(
                        url, timeout=self.timeout
                    )
                    if content:
                        # Create document with extracted content
                        processed_doc = {
                            "title": doc.get("title", "No title"),
                            "content": content,
                            "url": url,
                            "snippet": doc.get("snippet", ""),
                            "source_query": search_query,
                        }

                        # Split into chunks if text splitter is available
                        if text_splitter:
                            try:
                                chunks = text_splitter.split_text(content)
                                for i, chunk in enumerate(chunks):
                                    chunk_doc = processed_doc.copy()
                                    chunk_doc["content"] = chunk
                                    chunk_doc["chunk_id"] = i
                                    chunk_doc["original_title"] = doc.get(
                                        "title", "No title"
                                    )
                                    processed_documents.append(chunk_doc)
                            except Exception as e:
                                console.print(
                                    f"[yellow]Error splitting document: {e}[/]"
                                )
                                processed_documents.append(processed_doc)
                        else:
                            processed_documents.append(processed_doc)
                    else:
                        # If no content extracted, use the snippet
                        doc["content"] = doc.get("snippet", "")
                        processed_documents.append(doc)
                except Exception as e:
                    console.print(
                        f"[yellow]Error extracting content from {url}: {e}[/]"
                    )
                    # Use the document with just the snippet
                    doc["content"] = doc.get("snippet", "")
                    processed_documents.append(doc)
            else:
                doc["source_query"] = search_query
                processed_documents.append(doc)

        return processed_documents

    def _analyze_sufficiency(self, query: str, documents: List[Dict[str, str]]) -> bool:
        """
        Analyze collected documents to determine if we have sufficient information

        Args:
            query: Original user query
            documents: Collected documents

        Returns:
            True if sufficient information is found, False otherwise
        """
        if not documents:
            return False

        try:
            # Prepare document summaries
            doc_summaries = []
            for i, doc in enumerate(documents[:10]):  # Limit to 10 docs for analysis
                title = doc.get("original_title", "") or doc.get("title", "No title")
                doc_summaries.append(
                    f"Doc {i+1}: {title} - {doc.get('snippet', '')[:100]}"
                )

            doc_context = "\n".join(doc_summaries)

            # Ask AI to analyze if we have sufficient information
            system_prompt = """
            You are an information sufficiency analyzer. Based on the original query and the collected documents,
            determine if we have sufficient information to answer the query.
            Return ONLY a JSON object with the following structure:
            {
                "sufficient": true/false,
                "missing_aspects": ["list", "of", "missing", "information", "aspects"],
                "found_aspects": ["list", "of", "found", "information", "aspects"]
            }
            Be strict in your assessment - only return sufficient:true if the documents clearly contain the information needed to answer the query.
            """

            response = g4f.ChatCompletion.create(
                model=os.getenv("G4F_MODEL", "llama-4-scout"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Original query: {query}\n\nCollected document summaries:\n{doc_context}\n\nDo we have sufficient information to answer the query?",
                    },
                ],
            )

            # Parse response as JSON
            try:
                analysis = json.loads(response)
                return analysis["sufficient"]
            except:
                # Fallback if JSON parsing fails
                console.print(
                    "[yellow]Failed to parse AI analysis response. Using fallback analysis.[/]"
                )
                # Simple heuristic: if we have more than 10 documents, consider it sufficient
                return len(documents) >= 10

        except Exception as e:
            console.print(f"[yellow]Error in analyzing collected information: {e}[/]")
            # Simple fallback analysis
            return len(documents) >= 15  # Higher threshold for fallback

    def _refine_search_strategy(
        self, query: str, documents: List[Dict[str, str]]
    ) -> List[str]:
        """
        Refine the search strategy based on the collected documents

        Args:
            query: Original user query
            documents: Collected documents

        Returns:
            List of refined search queries
        """
        if not documents:
            return []

        # Remove duplicates
        unique_documents = SearchResultProcessor.filter_duplicates(documents)

        # Rank by relevance
        if HAS_SENTENCE_TRANSFORMERS and sentence_transformer:
            try:
                # Embed the query
                query_embedding = sentence_transformer.encode(query)

                # Embed the documents
                for doc in unique_documents:
                    doc_text = f"{doc.get('title', '')} {doc.get('snippet', '')}"
                    doc_embedding = sentence_transformer.encode(doc_text)

                    # Calculate similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    doc["relevance_score"] = float(similarity)

                # Filter by minimum relevance score
                filtered_docs = [
                    doc
                    for doc in unique_documents
                    if doc.get("relevance_score", 0) > self.min_relevance_score
                ]

                # Sort by relevance
                ranked_docs = sorted(
                    filtered_docs,
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True,
                )

                # Extract key terms from successful query
                key_terms = query.replace(query, "").strip().split()

                # Generate new search queries based on key terms and ranked documents
                new_queries = []
                for term in key_terms:
                    new_queries.append(f"{term} {query}")
                for doc in ranked_docs:
                    new_queries.append(
                        f"{doc.get('title', '')} {doc.get('snippet', '')}"
                    )

                # Ensure we have a reasonable number of queries
                if len(new_queries) > 10:
                    return new_queries[:10]  # Limit to 10 queries per round
                elif len(new_queries) < 2:
                    # Fallback to keyword combinations if we don't have enough queries
                    keywords = [doc.get("title", "") for doc in ranked_docs] + key_terms
                    if len(keywords) >= 3:
                        for i in range(
                            min(len(keywords) - 2, 3)
                        ):  # Add up to 3 keyword combinations
                            new_queries.append(
                                f"{keywords[i]} {keywords[i+1]} {keywords[i+2]}"
                            )

                return list(set(new_queries))  # Remove duplicates

            except Exception as e:
                console.print(f"[yellow]Error in semantic ranking: {e}[/]")
                # Fall back to regular ranking
                ranked_docs = SearchResultProcessor.rank_results(
                    unique_documents, query
                )
                return ranked_docs
        else:
            # Fall back to regular ranking
            ranked_docs = SearchResultProcessor.rank_results(unique_documents, query)
            return ranked_docs

    def _generate_fallback_queries(self, query: str) -> List[str]:
        """
        Generate fallback search queries if no relevant documents are found

        Args:
            query: Original user query

        Returns:
            List of fallback search queries
        """
        # Fallback to keyword combinations if no relevant documents are found
        keywords, entities = self.keyword_extractor.extract_keywords(query)
        combined_keywords = keywords + entities
        if len(combined_keywords) >= 3:
            return [
                f"{combined_keywords[i]} {combined_keywords[i+1]} {combined_keywords[i+2]}"
                for i in range(min(len(combined_keywords) - 2, 3))
            ]
        else:
            return [query]  # Just return the original query as fallback

    def _rank_and_filter_documents(
        self, query: str, documents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Rank and filter the collected documents

        Args:
            query: Original user query
            documents: All collected documents

        Returns:
            Ranked and filtered documents
        """
        if not documents:
            return []

        # Remove duplicates
        unique_documents = SearchResultProcessor.filter_duplicates(documents)

        # Rank by relevance
        if HAS_SENTENCE_TRANSFORMERS and sentence_transformer:
            try:
                # Embed the query
                query_embedding = sentence_transformer.encode(query)

                # Embed the documents
                for doc in unique_documents:
                    doc_text = f"{doc.get('title', '')} {doc.get('snippet', '')}"
                    doc_embedding = sentence_transformer.encode(doc_text)

                    # Calculate similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    doc["relevance_score"] = float(similarity)

                # Filter by minimum relevance score
                filtered_docs = [
                    doc
                    for doc in unique_documents
                    if doc.get("relevance_score", 0) > self.min_relevance_score
                ]

                # Sort by relevance
                ranked_docs = sorted(
                    filtered_docs,
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True,
                )

                return ranked_docs

            except Exception as e:
                console.print(f"[yellow]Error in semantic ranking: {e}[/]")
                # Fall back to regular ranking
                return SearchResultProcessor.rank_results(unique_documents, query)
        else:
            # Fall back to regular ranking
            return SearchResultProcessor.rank_results(unique_documents, query)

    def _display_search_results(self, documents: List[Dict[str, str]]):
        """Display search results in a formatted table"""
        if not documents:
            return

        table = Table(title="Search Results")
        table.add_column("Title", style="cyan", no_wrap=True)
        table.add_column("URL", style="magenta")
        table.add_column("Relevance", style="green")

        for doc in documents[:5]:  # Display top 5
            relevance = f"{doc.get('relevance_score', 0):.2f}" if doc.get('relevance_score') else "N/A"
            table.add_row(doc.get('title', 'No Title'), doc.get('url', ''), relevance)

        console.print(table)

