# Advanced RAG and Contact Extraction Toolkit

This project provides a powerful toolkit for performing advanced Retrieval-Augmented Generation (RAG) searches and extracting contact information from websites. It is designed with a modular architecture, separating the core logic into a reusable API and providing simple command-line interfaces (CLIs) for easy use.

## Project Structure

The project is organized into two main components:

1.  **`contact_api`**: A Python package containing the core logic for web content extraction, contact information parsing, and AI-powered summarization.
2.  **CLI Tools**:
    *   `rag_search.py`: A sophisticated RAG search tool that uses a strategic, iterative search process to answer questions with enhanced, AI-generated responses.
    *   `contact_extractor.py`: A CLI tool for extracting contact information (emails, phone numbers, social media links) from a given domain.

## Features

### RAG Search (`rag_search.py`)

- **Strategic Iterative Search**: Adapts its search strategy based on intermediate results.
- **AI-Powered Keyword Extraction**: Uses AI to identify relevant keywords in your query.
- **Semantic Search**: Employs sentence transformers to find semantically relevant content.
- **Vector Storage**: Caches document embeddings in ChromaDB for efficient retrieval.
- **AI-Enhanced Responses**: Generates comprehensive, cited answers based on search results.

### Contact Extractor (`contact_extractor.py` & `contact_api`)

- **Multi-Source Extraction**: Gathers information from web searches, direct homepage visits, and contact/about pages.
- **Comprehensive Parsing**: Extracts emails, phone numbers, social media profiles (LinkedIn, Twitter, GitHub), and physical addresses.
- **AI-Powered Summarization**: Uses an AI model to generate a clean, structured summary of the findings.
- **Modular and Reusable**: The core logic is available as a `ContactExtractor` class in the `contact_api` package for programmatic use.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables**:
    Copy `.env.example` to `.env` and customize the variables as needed.
    ```bash
    cp .env.example .env
    ```

## Usage

### RAG Search Tool

To start the RAG search assistant, run:

```bash
python rag_search.py
```

Follow the on-screen prompts to ask questions.

### Contact Extraction Tool

To extract contact information for a specific domain, run:

```bash
python contact_extractor.py
```

You will be prompted to enter the domain you want to search (e.g., `example.com`).

### Using the `contact_api` Programmatically

The `ContactExtractor` class can be easily integrated into your own Python scripts.

**Example**:

```python
from contact_api.extractor import ContactExtractor

# Initialize the extractor
extractor = ContactExtractor(timeout=10, max_search_results=15)

# Define the domain to search
domain = "example.com"

# Run the extraction process
contact_info = extractor.run(domain)

# Print the results
print(contact_info)
```

## Configuration

Key environment variables in your `.env` file:

- `G4F_MODEL`: The AI model to use for summarization and analysis (e.g., `deepseek`).
- `G4F_PROVIDER`: The provider for the G4F model.
- `MAX_SEARCH_RESULTS`: Maximum number of search results per query.
- `SEARCH_TIMEOUT`: Timeout for web requests in seconds.
- `MAX_DOCUMENTS_TO_USE`: Maximum number of documents to use for RAG responses.
- `MIN_RELEVANCE_SCORE`: Minimum relevance score for documents in RAG search.
- `EMBEDDING_MODEL`: The sentence transformer model for embeddings.

## Dependencies

- **Core**: `requests`, `beautifulsoup4`, `duckduckgo_search`, `g4f`, `rich`
- **Optional (for RAG search)**: `langchain`, `sentence-transformers`, `chromadb`, `numpy`

The tool is designed to function gracefully even if optional dependencies are missing, though some features like semantic search will be disabled.

## License

This project is licensed under the MIT License.

## Disclaimer

This tool is for educational and professional purposes. Please ensure you comply with the terms of service of any websites you interact with.
