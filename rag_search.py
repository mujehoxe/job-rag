#!/usr/bin/env python3
"""
RAG Search Tool - A tool to retrieve relevant documents from the internet and use them to enhance AI responses.
"""

import os
import sys
import json
import time

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from datetime import datetime
from typing import Dict, List, Any

# Try to import readline for better text input handling
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Modular Imports ---
from rag_api.assistant import RAGAssistant
from api_key_manager import APIKeyManager

# --- Configuration ---
MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "8"))  # Increased for thorough business intelligence
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "20"))  # Longer timeout for better content extraction

console = Console()

# --- Optional Dependency Handling ---
document_collection = None

# Try to import sentence-transformers and chromadb
try:
    # First check if we can import torch correctly
    try:
        import torch
        console.print(f"[green]PyTorch version: {torch.__version__}[/]")
    except ImportError as torch_err:
        console.print(f"[yellow]PyTorch import error: {torch_err}. Vector search will be disabled.[/]")
        raise ImportError("PyTorch not available")
        
    import chromadb
    from chromadb.utils import embedding_functions

    # Use a simpler embedding function if available
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
    except Exception as e:
        console.print(f"[yellow]Could not initialize SentenceTransformer: {e}. Trying default embedding.[/]")
        embedding_function = None
    
    chroma_client = chromadb.Client()
    document_collection = chroma_client.get_or_create_collection(
        name="web_documents", embedding_function=embedding_function
    )
    console.print("[green]Successfully initialized ChromaDB and embedding models.[/]")
except (ImportError, Exception) as e:
    console.print(f"[yellow]Warning: Vector search/storage not available: {e}[/]")
    console.print("[yellow]Continuing with basic keyword search.[/]")
    document_collection = None


# --- Session Saving ---
def save_session(query: str, response: str, documents: List[Dict[str, Any]]) -> None:
    """Save the session to a file."""
    os.makedirs("sessions", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sessions/session_{timestamp}.json"

    session_data = {
        "query": query,
        "response": response,
        "documents": documents,
        "timestamp": timestamp,
    }

    with open(filename, "w") as f:
        json.dump(session_data, f, indent=2)

    console.print(f"[dim]Session saved to {filename}[/]")


# --- Input Handling ---
def get_input(prompt_text: str) -> str:
    """Get input with readline support if available."""
    if HAS_READLINE:
        history_file = os.path.join(os.path.expanduser("~"), ".rag_history")
        try:
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
        except Exception as e:
            console.print(f"[dim]Could not read history file: {e}[/]")

        console.print(prompt_text, end="")
        user_input = input().replace('\\040', ' ')

        try:
            readline.write_history_file(history_file)
        except Exception as e:
            console.print(f"[dim]Could not write history file: {e}[/]")
        return user_input
    else:
        return Prompt.ask(prompt_text, default="", show_default=False)


# --- Main Application ---
def main():
    """Main function to run the RAG search assistant."""
    console.print(
        Panel(
            "[bold green]Strategic RAG Search Assistant[/]\nAsk questions to get answers enhanced with web search.",
            title="🔍 RAG Search CLI",
            expand=False,
        )
    )

    # Check and setup API keys
    api_manager = APIKeyManager()
    availability = api_manager.check_available_engines()
    
    # Show available search engines
    if any(availability.values()):
        available_engines = [engine for engine, available in availability.items() if available]
        console.print(f"[green]✅ Available search engines: {', '.join(available_engines)}[/green]")
    else:
        console.print("[yellow]⚠️  No API keys configured. Using fallback search methods.[/yellow]")
        if Confirm.ask("Would you like to setup API keys now for better search results?"):
            api_manager.setup_missing_keys()
            console.print()

    # Initialize the RAG assistant, passing the ChromaDB collection if available
    rag = RAGAssistant(chroma_collection=document_collection)

    while True:
        query = get_input("\n[bold green]You: [/]")
        if query.lower() in ["exit", "quit", "q"]:
            break

        start_time = time.time()
        response, documents = rag.ask(query)
        elapsed_time = time.time() - start_time

        console.print("\n[bold blue]AI:[/] ", end="")
        console.print(Markdown(response))
        console.print(f"[dim](Response generated in {elapsed_time:.2f}s)[/]")

        if documents:
            save_session(query, response, documents)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting...[/]")
        sys.exit(0)
