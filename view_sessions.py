#!/usr/bin/env python3
"""
View past RAG sessions and their results.
"""

import os
import json
from typing import List, Dict, Any
import argparse
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()


def list_sessions() -> List[str]:
    """List all available session files"""
    if not os.path.exists("sessions"):
        console.print("[bold red]No sessions directory found.[/]")
        return []

    sessions = []
    for filename in os.listdir("sessions"):
        if filename.startswith("session_") and filename.endswith(".json"):
            sessions.append(filename)

    # Sort by timestamp (newest first)
    sessions.sort(reverse=True)
    return sessions


def load_session(filename: str) -> Dict[str, Any]:
    """Load a session from file"""
    try:
        with open(os.path.join("sessions", filename), "r") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading session {filename}: {e}[/]")
        return {}


def display_session(session: Dict[str, Any]) -> None:
    """Display a session in a nice format"""
    if not session:
        return

    # Format timestamp
    timestamp_str = session.get("timestamp", "")
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_time = timestamp_str

    # Display session header
    console.print(
        Panel.fit(f"[bold blue]Session from {formatted_time}[/]", border_style="blue")
    )

    # Display query
    console.print("\n[bold green]Query:[/]")
    console.print(session.get("query", ""))

    # Display documents if available
    documents = session.get("documents", [])
    if documents:
        console.print("\n[bold yellow]Documents used:[/]")
        table = Table(expand=True)
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Title", style="green")

        for doc in documents:
            source = (
                doc.get("url", "").split("/")[2] if doc.get("url", "") else "Unknown"
            )
            table.add_row(source, doc.get("title", "No title"))

        console.print(table)

    # Display response
    console.print("\n[bold blue]AI Response:[/]")
    console.print(Markdown(session.get("response", "")))


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="View past RAG sessions")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List all available sessions"
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1,
        help="Display the nth most recent session (default: 1)",
    )
    parser.add_argument("-a", "--all", action="store_true", help="Display all sessions")
    parser.add_argument(
        "-f", "--file", type=str, help="Display a specific session file"
    )

    args = parser.parse_args()

    sessions = list_sessions()

    if not sessions:
        console.print("[bold red]No sessions found.[/]")
        return

    if args.list:
        console.print("[bold blue]Available sessions:[/]")
        table = Table(title="Sessions", expand=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Timestamp", style="green")
        table.add_column("Query", style="white")

        for i, filename in enumerate(sessions, 1):
            session = load_session(filename)
            timestamp_str = session.get("timestamp", "Unknown")
            try:
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp_str

            query = session.get("query", "")
            if len(query) > 50:
                query = query[:47] + "..."

            table.add_row(str(i), formatted_time, query)

        console.print(table)
        return

    if args.file:
        if args.file in sessions:
            session = load_session(args.file)
            display_session(session)
        else:
            console.print(f"[bold red]Session {args.file} not found.[/]")
        return

    if args.all:
        for i, filename in enumerate(sessions):
            if i > 0:
                console.print("\n" + "-" * 80 + "\n")
            session = load_session(filename)
            display_session(session)
        return

    # Display the nth most recent session
    if 1 <= args.number <= len(sessions):
        session = load_session(sessions[args.number - 1])
        display_session(session)
    else:
        console.print(
            f"[bold red]Invalid session number. Available range: 1-{len(sessions)}[/]"
        )


if __name__ == "__main__":
    main()
