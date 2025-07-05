#!/usr/bin/env python3
"""
RAG System Evaluation Script
Evaluates the RAG system using the comprehensive evaluation framework
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG system components
from rag_api.strategic_search import StrategicSearch
from api_key_manager import APIKeyManager

# Import evaluation framework
from evaluation.evaluation_framework import RAGEvaluator, TestCase, ContactInfo
from evaluation.test_dataset import (
    ALL_TEST_CASES, 
    SAMPLE_TEST_CASES, 
    get_test_cases_by_difficulty,
    get_test_cases_by_industry,
    get_sample_test_cases
)

console = Console()

def setup_api_keys():
    """Check and setup API keys if needed"""
    api_manager = APIKeyManager()
    availability = api_manager.check_available_engines()
    
    if any(availability.values()):
        available_engines = [engine for engine, available in availability.items() if available]
        console.print(f"[green]✅ Available search engines: {', '.join(available_engines)}[/green]")
        return True
    else:
        console.print("[yellow]⚠️  No API keys configured. Some search engines may not work optimally.[/yellow]")
        # Removed interactive prompt - just proceed with available engines
        return False

def create_rag_system(max_rounds: int = 8, max_results: int = 15) -> StrategicSearch:
    """Create and configure the RAG system for evaluation"""
    console.print(f"[cyan]Initializing RAG system with {max_rounds} rounds, {max_results} max results[/cyan]")
    
    # Configure the strategic search system
    rag_system = StrategicSearch(
        max_rounds=max_rounds,
        max_results=max_results,
        timeout=20,  # 20 second timeout
        min_relevance_score=0.15
    )
    
    return rag_system

def select_test_cases() -> List[TestCase]:
    """Default test case selection - returns sample cases"""
    console.print("[yellow]No specific test cases specified, using sample test cases (5 cases)[/yellow]")
    return get_sample_test_cases(5)

def run_evaluation(test_cases: List[TestCase], rag_system: StrategicSearch) -> None:
    """Run the evaluation and generate reports"""
    console.print(f"\n[bold green]Starting evaluation with {len(test_cases)} test cases (parallel execution)[/bold green]")
    
    # Create evaluator
    evaluator = RAGEvaluator(rag_system)
    
    # Run evaluation with parallel execution (max 3 workers to avoid overwhelming APIs)
    results = evaluator.evaluate_test_suite(test_cases, max_workers=3)
    
    # Generate comprehensive report
    report = evaluator.generate_report(results)
    
    # Display results
    console.print("\n" + "="*80)
    evaluator.display_report(report)
    
    # Save report
    report_path = evaluator.save_report(report)
    
    # Export to CSV
    csv_path = evaluator.export_to_csv(results)
    
    console.print(f"\n[bold green]Evaluation complete![/bold green]")
    console.print(f"[green]📊 Report saved to: {report_path}[/green]")
    console.print(f"[green]📈 CSV data saved to: {csv_path}[/green]")
    
    # Show summary
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.success)
    success_rate = successful_cases / total_cases if total_cases > 0 else 0
    
    console.print(Panel.fit(
        f"[bold]Evaluation Summary[/bold]\n"
        f"Total test cases: {total_cases}\n"
        f"Successful cases: {successful_cases}\n"
        f"Success rate: {success_rate:.2%}\n"
        f"Average F1 score: {report['metrics']['overall_f1']['mean']:.3f}",
        title="📊 Final Results",
        border_style="green"
    ))

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument("--max-rounds", type=int, default=8, help="Maximum search rounds")
    parser.add_argument("--max-results", type=int, default=15, help="Maximum search results per round")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation with sample cases")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Run specific difficulty level")
    parser.add_argument("--industry", type=str, help="Run specific industry test cases")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold green]RAG System Evaluation[/bold green]\n"
        "This script evaluates the performance of your RAG system\n"
        "using comprehensive metrics and test cases.",
        title="🔍 RAG Evaluation Tool",
        border_style="green"
    ))
    
    # Setup API keys
    setup_api_keys()
    
    # Create RAG system
    rag_system = create_rag_system(args.max_rounds, args.max_results)
    
    # Select test cases
    if args.quick:
        test_cases = get_sample_test_cases(5)
        console.print("[yellow]Running quick evaluation with 5 sample test cases[/yellow]")
    elif args.difficulty:
        test_cases = get_test_cases_by_difficulty(args.difficulty)
        console.print(f"[yellow]Running evaluation with {args.difficulty} difficulty test cases[/yellow]")
    elif args.industry:
        test_cases = get_test_cases_by_industry(args.industry)
        console.print(f"[yellow]Running evaluation with {args.industry} industry test cases[/yellow]")
    else:
        test_cases = select_test_cases()
    
    console.print(f"[cyan]Selected {len(test_cases)} test cases for evaluation[/cyan]")
    
    # Show test case preview
    if test_cases:
        console.print("\n[bold]Test cases to evaluate:[/bold]")
        for i, tc in enumerate(test_cases[:5], 1):
            console.print(f"  {i}. {tc.company_name} ({tc.difficulty}, {tc.industry})")
        if len(test_cases) > 5:
            console.print(f"  ... and {len(test_cases) - 5} more")
    
    # Removed confirmation prompt - start evaluation automatically
    
    # Run evaluation
    try:
        run_evaluation(test_cases, rag_system)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Evaluation interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Evaluation failed: {e}[/bold red]")
        raise

if __name__ == "__main__":
    main() 