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

# Default configuration
MAX_DOCUMENTS_TO_USE = int(os.getenv("MAX_DOCUMENTS_TO_USE", "5"))
MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "3"))

console = Console()

class RAGAssistant:
    """RAG-enhanced AI assistant using g4f"""

    def __init__(self, chroma_collection=None, max_docs=MAX_DOCUMENTS_TO_USE):
        """Initialize the RAG assistant"""
        # Use deepseek model with G4F
        self.model = os.getenv("G4F_MODEL", "deepseek")
        self.max_docs = max_docs
        self.searcher = StrategicSearch(max_rounds=MAX_SEARCH_ROUNDS)

        # Store search context and thinking process
        self.search_context = []

        # ChromaDB
        self.document_collection = chroma_collection
        self.has_chromadb = self.document_collection is not None

        # Use Pollinations AI provider
        try:
            self.provider = g4f.Provider.Pollinations
            console.print(
                "[green]Using Pollinations AI provider with deepseek model[/]"
            )
        except Exception as e:
            console.print(
                f"[yellow]Error initializing Pollinations provider: {e}. Will try other providers.[/]"
            )
            self.provider = None

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

        # 3. Limit documents to use
        documents = documents[: self.max_docs]
        self.search_context.append(
            f"Using top {len(documents)} documents for response generation."
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
                "You are a powerful research assistant. Your task is to analyze the provided documents "
                "and answer the user's query based *only* on the information in those documents. "
                "Extract all relevant facts and contact details. If the documents do not contain the answer, "
                "state that clearly."
            )

            if documents:
                doc_texts = []
                for i, doc in enumerate(documents, 1):
                    title = doc.get("original_title") or doc.get("title", "No Title")
                    content = doc.get("content", "")[:1500]
                    doc_text = f"Document {i}: {title}\nContent: {content}"
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

            # Generate response
            if self.provider:
                response = g4f.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    provider=self.provider,
                )
            else:
                response = g4f.ChatCompletion.create(
                    model=self.model, messages=messages
                )

            return response

        except Exception as e:
            console.print(f"[bold red]Error in AI response generation: {e}[/]")
            # Fallback to a simple response
            return "I am sorry, but I encountered an error while generating the response."

    def _store_documents_in_chroma(self, documents: List[Dict[str, str]]) -> None:
        """Store documents in ChromaDB if available"""
        if not self.has_chromadb:
            return

        try:
            # Prepare documents for ChromaDB
            docs_to_store = []
            metadatas = []
            ids = []

            for doc in documents:
                # Ensure document has a URL
                if "url" in doc and doc["url"]:
                    docs_to_store.append(doc["content"])
                    metadatas.append(
                        {
                            "title": doc.get("title", ""),
                            "url": doc["url"],
                            "snippet": doc.get("snippet", ""),
                        }
                    )
                    ids.append(doc["url"])  # Use URL as ID

            # Add to collection if there are documents to store
            if docs_to_store:
                self.document_collection.add(
                    documents=docs_to_store, metadatas=metadatas, ids=ids
                )
                self.search_context.append(
                    f"Stored {len(docs_to_store)} documents in vector DB."
                )

        except Exception as e:
            console.print(f"[yellow]Error storing documents in ChromaDB: {e}[/]")
            self.search_context.append(f"Error storing documents in ChromaDB: {e}")
