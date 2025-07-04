#!/usr/bin/env python3
"""
Defines the RAGAssistant class for the RAG API.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import g4f
from rich.console import Console

from .strategic_search import StrategicSearch

# Default configuration - Removed MAX_DOCUMENTS_TO_USE limit for better business intelligence
MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "8"))  # Increased for thorough investigation

console = Console()

class RAGAssistant:
    """RAG-enhanced AI assistant using g4f"""

    def __init__(self, chroma_collection=None):
        """Initialize the RAG assistant"""
        # Use llama-4-scout model with local API
        self.model = os.getenv("G4F_MODEL", "llama-4-scout")
        # Removed max_docs limit - use all relevant documents for better business intelligence
        self.searcher = StrategicSearch(max_rounds=MAX_SEARCH_ROUNDS)

        # Store search context and thinking process
        self.search_context = []

        # ChromaDB
        self.document_collection = chroma_collection
        self.has_chromadb = self.document_collection is not None

        # Configure for local API endpoint
        self.api_base = os.getenv("LOCAL_API_BASE", "http://localhost:8080/v1")
        self.provider = None  # No provider needed for local API
        console.print(f"[green]Using local API with {self.model} model at {self.api_base}[/]")

    def ask(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Ask a question with RAG enhancement

        Args:
            query: User query

        Returns:
            A tuple containing the AI response and the list of documents used.
        """
        # Reset search context for new query
        self.search_context = []
        self.search_context.append(f"Initial query: {query}")

        # 1. Strategic search for relevant documents
        console.print("[bold green]Starting strategic search...[/]")
        documents = self.searcher.strategic_search(query)

        # Store search context from the searcher
        if hasattr(self.searcher, "search_rounds_context"):
            self.search_context.extend(self.searcher.search_rounds_context)

        if not documents:
            console.print(
                "[yellow]No relevant documents found. Using AI without RAG.[/]"
            )
            self.search_context.append(
                "No relevant documents found. Using AI without RAG."
            )
            return self._ai_response(query, None), []

        # 2. Display found documents count
        console.print(f"[bold green]Found {len(documents)} relevant documents.[/]")
        self.search_context.append(f"Found {len(documents)} relevant documents.")

        # 3. Use all relevant documents for better business intelligence
        self.search_context.append(
            f"Using all {len(documents)} documents for comprehensive analysis."
        )

        # 4. Store documents in vector database if available
        if self.has_chromadb:
            try:
                self._store_documents_in_chroma(documents)
            except Exception as e:
                console.print(f"[yellow]Failed to store documents in vector DB: {e}[/]")
                self.search_context.append(
                    f"Failed to store documents in vector DB: {e}"
                )

        # 5. Process with AI using RAG
        console.print("[bold green]Generating AI response...[/]")
        response = self._ai_response(query, documents)

        return response, documents

    def _ai_response(
        self, query: str, documents: Optional[List[Dict[str, str]]]
    ) -> str:
        """Get response from AI with or without RAG"""
        try:
            system_prompt = (
                "You are a powerful business intelligence research assistant. Your task is to analyze the provided documents "
                "and answer the user's query based *only* on the information in those documents. "
                "Extract all relevant facts and contact details. ALWAYS cite your sources by including the URL "
                "where you found specific information. Format citations as [Source: URL]. "
                "If the documents do not contain the answer, state that clearly."
            )

            if documents:
                doc_texts = []
                for i, doc in enumerate(documents, 1):
                    title = doc.get("original_title") or doc.get("title", "No Title")
                    content = doc.get("content", "")[:2000]  # Increased content length for better analysis
                    source_url = doc.get("source_url") or doc.get("href") or doc.get("url", "")
                    
                    # Include source URL in document text for AI to reference
                    doc_text = f"Document {i}: {title}\nSource URL: {source_url}\nContent: {content}"
                    
                    # Add business relevance indicator if available
                    if doc.get("business_relevant"):
                        doc_text += "\n[Business-relevant page detected]"
                    
                    # Add sitemap URLs if available for future reference
                    if doc.get("sitemap_urls"):
                        doc_text += f"\n[Additional relevant pages found: {', '.join(doc['sitemap_urls'][:3])}]"
                    
                    doc_texts.append(doc_text)

                all_docs = "\n\n---\n\n".join(doc_texts)

                # Construct a new user message with the documents
                user_message = (
                    f"Query: {query}\n\n"
                    f"Please analyze the following documents and answer the query based on their content:\n\n"
                    f"{all_docs}"
                )
            else:
                user_message = query

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Generate response using local API
            import requests
            
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                console.print(f"[bold red]Error calling local API: {e}[/bold red]")
                # Fallback to g4f if local API fails
                try:
                    response = g4f.ChatCompletion.create(
                        model=self.model, messages=messages
                    )
                    return response
                except Exception as g4f_error:
                    console.print(f"[bold red]G4F fallback also failed: {g4f_error}[/bold red]")
                    raise e

        except Exception as e:
            console.print(f"[bold red]Error in AI response generation: {e}[/]")
            # Fallback to a simple response
            return "I am sorry, but I encountered an error while generating the response."

    def _store_documents_in_chroma(self, documents: List[Dict[str, str]]) -> None:
        """Store documents in ChromaDB if available"""
        if not self.has_chromadb:
            return

        try:
            # Check if collection has embedding function
            if not hasattr(self.document_collection, '_embedding_function') or self.document_collection._embedding_function is None:
                console.print(f"[yellow]ChromaDB collection has no embedding function, skipping vector storage[/yellow]")
                return
                
            # Prepare documents for ChromaDB
            docs_to_store = []
            metadatas = []
            ids = []

            for doc in documents:
                # Ensure document has a URL
                url = doc.get("url") or doc.get("source_url") or doc.get("href")
                if url:
                    docs_to_store.append(doc.get("content", ""))
                    metadatas.append(
                        {
                            "title": doc.get("title", ""),
                            "url": url,
                            "snippet": doc.get("snippet", ""),
                        }
                    )
                    ids.append(url)  # Use URL as ID

            # Add to collection if there are documents to store
            if docs_to_store:
                # Check for duplicate IDs first
                try:
                    existing_ids = set()
                    try:
                        existing_data = self.document_collection.get()
                        existing_ids = set(existing_data.get('ids', []))
                    except:
                        pass  # Collection might be empty
                    
                    # Filter out existing documents
                    new_docs = []
                    new_metadatas = []
                    new_ids = []
                    
                    for doc, metadata, doc_id in zip(docs_to_store, metadatas, ids):
                        if doc_id not in existing_ids:
                            new_docs.append(doc)
                            new_metadatas.append(metadata)
                            new_ids.append(doc_id)
                    
                    if new_docs:
                        self.document_collection.add(
                            documents=new_docs, metadatas=new_metadatas, ids=new_ids
                        )
                        self.search_context.append(
                            f"Stored {len(new_docs)} new documents in vector DB."
                        )
                    else:
                        self.search_context.append(
                            "All documents already exist in vector DB."
                        )
                except Exception as add_error:
                    console.print(f"[yellow]ChromaDB add operation failed: {add_error}[/yellow]")

        except Exception as e:
            console.print(f"[yellow]Error storing documents in ChromaDB: {e}[/yellow]")
            self.search_context.append(f"ChromaDB storage skipped due to: {e}")
