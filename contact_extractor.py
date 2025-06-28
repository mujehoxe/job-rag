#!/usr/bin/env python3
"""
Command-line tool to extract contact information from a domain.
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from contact_api.extractor import ContactExtractor


def main():
    """Main function to run the contact extractor CLI."""
    console = Console()

    # Display header
    console.print(
        Panel(
            "[bold green]Contact Extractor[/]\nFind contact information for any domain.",
            title="📞 Contact Finder",
            expand=False,
        )
    )

    # Get domain from user
    domain = Prompt.ask("[bold green]Enter the domain to search (e.g., example.com)[/]")

    if not domain:
        console.print("[red]Domain cannot be empty.[/]")
        return

    try:
        # Initialize and run the extractor
        with console.status(f"[bold green]Searching for contacts on {domain}...[/]", spinner="dots"):
            extractor = ContactExtractor()
            contact_info = extractor.extract_contacts(domain)

        # Display results
        console.print(f"\n[bold blue]Contact Information for {domain}:[/]")
        console.print(contact_info)

    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/]")


if __name__ == "__main__":
    main()
