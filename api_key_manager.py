#!/usr/bin/env python3
"""
API Key Management System for Search Engines
"""

import os
import re
from typing import Dict, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

console = Console()

# Search engine configurations
SEARCH_ENGINES = {
    "brave": {
        "name": "Brave Search API",
        "env_var": "BRAVE_SEARCH_API_KEY",
        "signup_url": "https://api.search.brave.com/app/keys",
        "free_tier": "5,000 queries/month, 1 query/second",
        "description": "Privacy-focused search with excellent results"
    },
    "serper": {
        "name": "Serper.dev",
        "env_var": "SERPER_API_KEY", 
        "signup_url": "https://serper.dev",
        "free_tier": "2,500 queries/month",
        "description": "Google search results via API"
    },
    "google": {
        "name": "Google Custom Search",
        "env_var": "GOOGLE_SEARCH_API_KEY",
        "signup_url": "https://developers.google.com/custom-search/v1/introduction",
        "free_tier": "100 queries/day, $5 per 1000 additional",
        "description": "Google search results via Custom Search API",
        "additional_setup": "Also requires GOOGLE_SEARCH_ENGINE_ID from Programmable Search Engine"
    }
}

class APIKeyManager:
    """Manage API keys for search engines"""
    
    def __init__(self, env_file_path: str = ".env"):
        self.env_file_path = env_file_path
        self.load_env_vars()
    
    def load_env_vars(self):
        """Load environment variables from .env file"""
        if os.path.exists(self.env_file_path):
            with open(self.env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value

    def check_available_engines(self) -> Dict[str, bool]:
        """Check which search engines have API keys configured"""
        availability = {}
        for engine_id, config in SEARCH_ENGINES.items():
            env_var = config["env_var"]
            api_key = os.getenv(env_var)
            
            # For Google Custom Search, also check for Search Engine ID
            if engine_id == "google":
                search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
                availability[engine_id] = bool(api_key and api_key.strip() and search_engine_id and search_engine_id.strip())
            else:
                availability[engine_id] = bool(api_key and api_key.strip())
        return availability

    def display_engine_status(self):
        """Display the status of all search engines"""
        availability = self.check_available_engines()
        
        table = Table(title="🔍 Search Engine API Status")
        table.add_column("Engine", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Free Tier", style="green")
        table.add_column("Description", style="dim")
        
        for engine_id, config in SEARCH_ENGINES.items():
            status = "✅ Configured" if availability[engine_id] else "❌ Not configured"
            status_style = "green" if availability[engine_id] else "red"
            
            table.add_row(
                config["name"],
                f"[{status_style}]{status}[/]",
                config["free_tier"],
                config["description"]
            )
        
        console.print(table)
        console.print()

    def prompt_for_api_key(self, engine_id: str) -> Optional[str]:
        """Prompt user for a specific API key"""
        config = SEARCH_ENGINES[engine_id]
        
        # Show signup information
        additional_info = f"\n⚠️  {config['additional_setup']}" if config.get('additional_setup') else ""
        signup_panel = Panel(
            f"🌐 Get your API key from: [link]{config['signup_url']}[/link]\n"
            f"📊 Free tier: {config['free_tier']}\n"
            f"📝 {config['description']}{additional_info}",
            title=f"Setup {config['name']}",
            border_style="blue"
        )
        console.print(signup_panel)
        
        # Prompt for API key
        api_key = Prompt.ask(
            f"Enter your {config['name']} API key (or press Enter to skip)",
            default="",
            show_default=False
        )
        
        if api_key.strip():
            # For Google Custom Search, also prompt for Search Engine ID
            if engine_id == "google":
                search_engine_id = Prompt.ask(
                    "Enter your Google Search Engine ID (from Programmable Search Engine)",
                    default="",
                    show_default=False
                )
                if search_engine_id.strip():
                    # Save both API key and Search Engine ID
                    self.save_api_key(engine_id, api_key.strip())
                    self.save_search_engine_id(search_engine_id.strip())
                    return api_key.strip()
                else:
                    console.print("[yellow]Search Engine ID is required for Google Custom Search[/yellow]")
                    return None
            return api_key.strip()
        else:
            console.print(f"[yellow]Skipping {config['name']}[/yellow]")
            return None

    def save_api_key(self, engine_id: str, api_key: str):
        """Save API key to .env file"""
        config = SEARCH_ENGINES[engine_id]
        env_var = config["env_var"]
        
        # Read existing .env file
        env_lines = []
        if os.path.exists(self.env_file_path):
            with open(self.env_file_path, 'r') as f:
                env_lines = f.readlines()
        
        # Check if the variable already exists
        found = False
        for i, line in enumerate(env_lines):
            if line.startswith(f"{env_var}=") or line.startswith(f"# {env_var}="):
                env_lines[i] = f"{env_var}={api_key}\n"
                found = True
                break
        
        # If not found, add it
        if not found:
            env_lines.append(f"{env_var}={api_key}\n")
        
        # Write back to file
        with open(self.env_file_path, 'w') as f:
            f.writelines(env_lines)
        
        # Update environment variable
        os.environ[env_var] = api_key
        
        console.print(f"[green]✅ Saved {config['name']} API key[/green]")

    def save_search_engine_id(self, search_engine_id: str):
        """Save Google Search Engine ID to .env file"""
        env_var = "GOOGLE_SEARCH_ENGINE_ID"
        
        # Read existing .env file
        env_lines = []
        if os.path.exists(self.env_file_path):
            with open(self.env_file_path, 'r') as f:
                env_lines = f.readlines()
        
        # Check if the variable already exists
        found = False
        for i, line in enumerate(env_lines):
            if line.startswith(f"{env_var}=") or line.startswith(f"# {env_var}="):
                env_lines[i] = f"{env_var}={search_engine_id}\n"
                found = True
                break
        
        # If not found, add it
        if not found:
            env_lines.append(f"{env_var}={search_engine_id}\n")
        
        # Write back to file
        with open(self.env_file_path, 'w') as f:
            f.writelines(env_lines)
        
        # Update environment variable
        os.environ[env_var] = search_engine_id
        
        console.print(f"[green]✅ Saved Google Search Engine ID[/green]")

    def setup_missing_keys(self):
        """Interactively setup missing API keys"""
        availability = self.check_available_engines()
        missing_engines = [engine_id for engine_id, available in availability.items() if not available]
        
        if not missing_engines:
            console.print("[green]✅ All search engines are configured![/green]")
            return
        
        console.print(f"[yellow]📝 Found {len(missing_engines)} search engines without API keys[/yellow]")
        console.print()
        
        if not Confirm.ask("Would you like to configure API keys now?"):
            console.print("[yellow]Skipping API key setup. You can run this again later.[/yellow]")
            return
        
        console.print()
        
        for engine_id in missing_engines:
            config = SEARCH_ENGINES[engine_id]
            console.print(f"[bold cyan]Setting up {config['name']}...[/bold cyan]")
            
            api_key = self.prompt_for_api_key(engine_id)
            if api_key:
                self.save_api_key(engine_id, api_key)
            
            console.print()

    def setup_engine(self, engine_id: str):
        """Setup a specific search engine"""
        if engine_id not in SEARCH_ENGINES:
            console.print(f"[red]❌ Unknown search engine: {engine_id}[/red]")
            return
        
        config = SEARCH_ENGINES[engine_id]
        console.print(f"[bold cyan]Setting up {config['name']}...[/bold cyan]")
        
        api_key = self.prompt_for_api_key(engine_id)
        if api_key:
            self.save_api_key(engine_id, api_key)

def check_and_setup_api_keys():
    """Main function to check and setup API keys"""
    manager = APIKeyManager()
    
    console.print(Panel("🔑 Search Engine API Key Manager", style="bold blue"))
    console.print()
    
    manager.display_engine_status()
    manager.setup_missing_keys()

if __name__ == "__main__":
    check_and_setup_api_keys() 