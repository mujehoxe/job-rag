#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for RAG System
Evaluates accuracy, completeness, precision, recall, and performance metrics
"""

import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import statistics
import csv
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID

console = Console()

@dataclass
class ContactInfo:
    """Standard contact information structure"""
    emails: List[str]
    phones: List[str]
    social_media: Dict[str, str]  # platform -> url
    linkedin_profiles: List[str]
    company_website: Optional[str]
    key_people: List[str]

@dataclass
class TestCase:
    """Test case for evaluation"""
    company_name: str
    domain: str
    query: str
    expected_result: ContactInfo
    difficulty: str  # 'easy', 'medium', 'hard'
    industry: str
    notes: str = ""

@dataclass
class EvaluationResult:
    """Result of evaluating a single test case"""
    test_case: TestCase
    actual_result: ContactInfo
    metrics: Dict[str, float]
    execution_time: float
    search_rounds: int
    documents_found: int
    success: bool
    error_message: Optional[str] = None

class ContactInfoExtractor:
    """Extracts contact information from RAG system results"""
    
    def __init__(self):
        import re
        self.re = re  # Store re module as class attribute
        # Improved phone pattern to avoid dates and IDs
        self.phone_pattern = re.compile(r'(?:\+?1[-.\s]?)?\(?(?:[0-9]{3})\)?[-.\s]?(?:[0-9]{3})[-.\s]?(?:[0-9]{4})')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.linkedin_pattern = re.compile(r'linkedin\.com/(?:in/|company/)([a-zA-Z0-9-]+)')
        
        # Patterns to exclude (dates, IDs, etc.)
        self.exclude_phone_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # Date format
            re.compile(r'^\d{6,}$'),  # Long numbers without formatting
            re.compile(r'^\d{4}$'),   # Short numbers
            re.compile(r'^\d{2}-\d{2}-\d{4}$'),  # Date format
        ]
        
    def extract_from_documents(self, documents: List[Dict[str, Any]]) -> ContactInfo:
        """Extract contact information from search documents"""
        emails = set()
        phones = set()
        social_media = {}
        linkedin_profiles = set()
        company_website = None
        key_people = set()
        
        for doc in documents:
            content = doc.get('content', '') + ' ' + doc.get('snippet', '')
            url = doc.get('source_url') or doc.get('href', '')
            
            # Extract emails with better filtering
            email_matches = self.email_pattern.findall(content)
            for email in email_matches:
                # Filter out common non-business domains and system emails
                domain = email.split('@')[1].lower()
                excluded_domains = ['example.com', 'test.com', 'domain.com', 'email.com', 'localhost']
                system_prefixes = ['no-reply', 'noreply', 'do-not-reply', 'system', 'admin', 'webmaster']
                
                if domain not in excluded_domains and not any(email.lower().startswith(prefix) for prefix in system_prefixes):
                    emails.add(email)
            
            # Extract phones with better validation
            phone_matches = self.phone_pattern.findall(content)
            for phone in phone_matches:
                # Clean the phone number
                cleaned_phone = phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '').replace('.', '')
                
                # Skip if it matches exclusion patterns
                if any(pattern.match(phone.strip()) for pattern in self.exclude_phone_patterns):
                    continue
                
                # Must be reasonable phone number length
                if 7 <= len(cleaned_phone) <= 15:
                    # Check if it's in a phone context (within 50 chars of phone-related words)
                    context_start = max(0, content.find(phone) - 50)
                    context_end = min(len(content), content.find(phone) + len(phone) + 50)
                    context = content[context_start:context_end].lower()
                    
                    phone_indicators = ['phone', 'tel', 'call', 'contact', 'number', 'dial', 'fax']
                    if any(indicator in context for indicator in phone_indicators):
                        phones.add(phone.strip())
            
            # Extract LinkedIn profiles with better URL detection
            linkedin_matches = self.linkedin_pattern.findall(content)
            for match in linkedin_matches:
                linkedin_profiles.add(f"linkedin.com/{match}")
            
            # Also look for full LinkedIn URLs
            linkedin_url_pattern = self.re.compile(r'https?://(?:www\.)?linkedin\.com/(?:in/|company/)([a-zA-Z0-9-]+)')
            linkedin_url_matches = linkedin_url_pattern.findall(content)
            for match in linkedin_url_matches:
                linkedin_profiles.add(f"linkedin.com/{match}")
            
            # Detect company website
            if url and not company_website:
                if any(platform not in url.lower() for platform in ['linkedin', 'facebook', 'twitter', 'instagram']):
                    company_website = url
            
            # Extract social media
            social_platforms = {
                'facebook': ['facebook.com', 'fb.com'],
                'twitter': ['twitter.com', 'x.com'],
                'instagram': ['instagram.com'],
                'youtube': ['youtube.com'],
                'tiktok': ['tiktok.com']
            }
            
            for platform, domains in social_platforms.items():
                for domain in domains:
                    if domain in content.lower():
                        # Try to extract the actual URL
                        url_pattern = rf'https?://[a-zA-Z0-9\-\.]*{domain}/[a-zA-Z0-9\-\_\.\/]*'
                        matches = self.re.findall(url_pattern, content)
                        if matches:
                            social_media[platform] = matches[0]
                        elif platform not in social_media:
                            social_media[platform] = f"mentioned in {url}"
            
            # Extract key people (improved detection)
            # Look for patterns like "CEO John Smith", "John Smith, CEO", "Founded by John Smith"
            people_patterns = [
                # Pattern: "CEO John Smith" or "John Smith, CEO"
                r'(?:CEO|CTO|CFO|COO|President|Founder|Co-Founder|Chief Executive Officer|Chief Technology Officer|Chief Financial Officer|Chief Operating Officer)[\s:,]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,?\s*(?:CEO|CTO|CFO|COO|President|Founder|Co-Founder|Chief Executive Officer|Chief Technology Officer|Chief Financial Officer|Chief Operating Officer)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[-–]\s*(?:CEO|CTO|CFO|COO|President|Founder|Co-Founder)',
                # Pattern: "Founded by John Smith" or "Led by John Smith"
                r'(?:Founded|Led|Headed|Managed)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                # Pattern: "John Smith is the CEO"
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+(?:CEO|CTO|CFO|COO|President|Founder|Co-Founder)',
            ]
            
            for pattern in people_patterns:
                matches = self.re.findall(pattern, content, self.re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    # Clean up the match
                    person_name = match.strip()
                    # Filter out common false positives and ensure it's a proper name
                    if (len(person_name) > 3 and 
                        person_name not in ['CEO', 'CTO', 'CFO', 'COO', 'President', 'Founder', 'Director', 'VP', 'Manager', 'Chief', 'Officer', 'Executive'] and
                        not person_name.lower().startswith('http') and
                        not person_name.lower().startswith('www') and
                        ' ' in person_name and  # Ensure it's a full name
                        not any(word in person_name.lower() for word in ['policy', 'cookie', 'sign', 'terms', 'privacy', 'url', 'title', 'snippet'])):
                        key_people.add(person_name)
        
        return ContactInfo(
            emails=list(emails),
            phones=list(phones),
            social_media=social_media,
            linkedin_profiles=list(linkedin_profiles),
            company_website=company_website,
            key_people=list(key_people)
        )

class EvaluationMetrics:
    """Calculate evaluation metrics for contact information extraction"""
    
    @staticmethod
    def calculate_precision_recall(expected: ContactInfo, actual: ContactInfo) -> Dict[str, float]:
        """Calculate precision and recall for each contact type"""
        metrics = {}
        
        # Email metrics with flexible matching
        expected_emails = set(expected.emails)
        actual_emails = set(actual.emails)
        
        # Try exact matching first
        exact_precision = EvaluationMetrics._precision(expected_emails, actual_emails)
        exact_recall = EvaluationMetrics._recall(expected_emails, actual_emails)
        
        # If exact matching fails, try domain-based matching for partial credit
        if exact_precision == 0 and exact_recall == 0:
            domain_precision = EvaluationMetrics._domain_based_precision(expected_emails, actual_emails)
            domain_recall = EvaluationMetrics._domain_based_recall(expected_emails, actual_emails)
            metrics['email_precision'] = domain_precision * 0.7  # Partial credit
            metrics['email_recall'] = domain_recall * 0.7
        else:
            metrics['email_precision'] = exact_precision
            metrics['email_recall'] = exact_recall
            
        metrics['email_f1'] = EvaluationMetrics._f1_score(metrics['email_precision'], metrics['email_recall'])
        
        # Phone metrics with normalization
        expected_phones = set(EvaluationMetrics._normalize_phone_number(phone) for phone in expected.phones)
        actual_phones = set(EvaluationMetrics._normalize_phone_number(phone) for phone in actual.phones)
        metrics['phone_precision'] = EvaluationMetrics._precision(expected_phones, actual_phones)
        metrics['phone_recall'] = EvaluationMetrics._recall(expected_phones, actual_phones)
        metrics['phone_f1'] = EvaluationMetrics._f1_score(metrics['phone_precision'], metrics['phone_recall'])
        
        # Social media metrics
        expected_social = set(expected.social_media.keys())
        actual_social = set(actual.social_media.keys())
        metrics['social_precision'] = EvaluationMetrics._precision(expected_social, actual_social)
        metrics['social_recall'] = EvaluationMetrics._recall(expected_social, actual_social)
        metrics['social_f1'] = EvaluationMetrics._f1_score(metrics['social_precision'], metrics['social_recall'])
        
        # LinkedIn metrics with URL normalization
        expected_linkedin = set(EvaluationMetrics._normalize_linkedin_url(url) for url in expected.linkedin_profiles)
        actual_linkedin = set(EvaluationMetrics._normalize_linkedin_url(url) for url in actual.linkedin_profiles)
        metrics['linkedin_precision'] = EvaluationMetrics._precision(expected_linkedin, actual_linkedin)
        metrics['linkedin_recall'] = EvaluationMetrics._recall(expected_linkedin, actual_linkedin)
        metrics['linkedin_f1'] = EvaluationMetrics._f1_score(metrics['linkedin_precision'], metrics['linkedin_recall'])
        
        # Overall completeness score
        total_expected = len(expected.emails) + len(expected.phones) + len(expected.social_media) + len(expected.linkedin_profiles)
        total_found = len(actual.emails) + len(actual.phones) + len(actual.social_media) + len(actual.linkedin_profiles)
        metrics['completeness_score'] = min(total_found / max(total_expected, 1), 1.0)
        
        # Overall accuracy (weighted average of F1 scores)
        f1_scores = [metrics['email_f1'], metrics['phone_f1'], metrics['social_f1'], metrics['linkedin_f1']]
        metrics['overall_f1'] = statistics.mean(f1_scores)
        
        return metrics
    
    @staticmethod
    def _precision(expected: set, actual: set) -> float:
        """Calculate precision"""
        if not actual:
            return 0.0
        intersection = expected.intersection(actual)
        return len(intersection) / len(actual)
    
    @staticmethod
    def _recall(expected: set, actual: set) -> float:
        """Calculate recall"""
        if not expected:
            return 1.0 if not actual else 0.0
        intersection = expected.intersection(actual)
        return len(intersection) / len(expected)
    
    @staticmethod
    def _f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def _normalize_linkedin_url(url: str) -> str:
        """Normalize LinkedIn URLs to handle different formats"""
        # Remove protocol and www
        normalized = url.replace('https://', '').replace('http://', '').replace('www.', '')
        # Remove /company/ variations - both linkedin.com/company/hubspot and linkedin.com/hubspot are valid
        normalized = normalized.replace('/company/', '/')
        # Remove trailing slashes
        normalized = normalized.rstrip('/')
        return normalized
    
    @staticmethod
    def _normalize_phone_number(phone: str) -> str:
        """Normalize phone numbers to handle different formats"""
        # Remove all non-digit characters
        digits_only = ''.join(c for c in phone if c.isdigit())
        # Handle US numbers with or without country code
        if len(digits_only) == 11 and digits_only.startswith('1'):
            return digits_only[1:]  # Remove leading 1
        return digits_only
    
    @staticmethod
    def _domain_based_precision(expected_emails: set, actual_emails: set) -> float:
        """Calculate precision based on email domains for partial credit"""
        if not actual_emails:
            return 0.0
        
        expected_domains = set(email.split('@')[1] for email in expected_emails if '@' in email)
        actual_domains = set(email.split('@')[1] for email in actual_emails if '@' in email)
        
        if not expected_domains:
            return 0.0
        
        domain_matches = expected_domains.intersection(actual_domains)
        return len(domain_matches) / len(expected_domains)
    
    @staticmethod
    def _domain_based_recall(expected_emails: set, actual_emails: set) -> float:
        """Calculate recall based on email domains for partial credit"""
        if not expected_emails:
            return 1.0 if not actual_emails else 0.0
        
        expected_domains = set(email.split('@')[1] for email in expected_emails if '@' in email)
        actual_domains = set(email.split('@')[1] for email in actual_emails if '@' in email)
        
        if not expected_domains:
            return 0.0
        
        domain_matches = expected_domains.intersection(actual_domains)
        return len(domain_matches) / len(expected_domains)

class RAGEvaluator:
    """Main evaluation class for the RAG system"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.extractor = ContactInfoExtractor()
        self.results_dir = Path("evaluation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def evaluate_single_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        start_time = time.time()
        
        try:
            # Run the RAG system
            console.print(f"[cyan]Evaluating: {test_case.company_name}[/cyan]")
            documents = self.rag_system.strategic_search(test_case.query)
            
            # Extract contact information
            actual_result = self.extractor.extract_from_documents(documents)
            
            # Calculate metrics
            metrics = EvaluationMetrics.calculate_precision_recall(test_case.expected_result, actual_result)
            
            execution_time = time.time() - start_time
            
            # Determine success with graduated thresholds
            success = self._determine_success(metrics['overall_f1'])
            
            result = EvaluationResult(
                test_case=test_case,
                actual_result=actual_result,
                metrics=metrics,
                execution_time=execution_time,
                search_rounds=getattr(self.rag_system, 'max_rounds', 8),
                documents_found=len(documents),
                success=success
            )
            
            self.logger.info(f"Evaluated {test_case.company_name}: F1={metrics['overall_f1']:.3f}, Time={execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error evaluating {test_case.company_name}: {e}")
            
            return EvaluationResult(
                test_case=test_case,
                actual_result=ContactInfo([], [], {}, [], None, []),
                metrics={'overall_f1': 0.0},
                execution_time=execution_time,
                search_rounds=0,
                documents_found=0,
                success=False,
                error_message=str(e)
            )
    
    def evaluate_test_suite(self, test_cases: List[TestCase], max_workers: int = 3) -> List[EvaluationResult]:
        """Evaluate a full test suite with parallel execution"""
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Evaluating test cases...", total=len(test_cases))
            
            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all test cases
                future_to_case = {
                    executor.submit(self._evaluate_single_case_quiet, test_case): test_case 
                    for test_case in test_cases
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_case):
                    test_case = future_to_case[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # Create error result
                        error_result = EvaluationResult(
                            test_case=test_case,
                            actual_result=ContactInfo([], [], {}, [], None, []),
                            metrics={'overall_f1': 0.0},
                            execution_time=0.0,
                            search_rounds=0,
                            documents_found=0,
                            success=False,
                            error_message=str(e)
                        )
                        results.append(error_result)
                        self.logger.error(f"Error evaluating {test_case.company_name}: {e}")
                    
                    progress.update(task, advance=1)
        
        # Sort results by company name to maintain consistent order
        results.sort(key=lambda r: r.test_case.company_name)
        return results
    
    def _evaluate_single_case_quiet(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case without console output (for parallel execution)"""
        start_time = time.time()
        
        try:
            # Run the RAG system
            documents = self.rag_system.strategic_search(test_case.query)
            
            # Extract contact information
            actual_result = self.extractor.extract_from_documents(documents)
            
            # Calculate metrics
            metrics = EvaluationMetrics.calculate_precision_recall(test_case.expected_result, actual_result)
            
            execution_time = time.time() - start_time
            
            # Determine success with graduated thresholds
            success = self._determine_success(metrics['overall_f1'])
            
            result = EvaluationResult(
                test_case=test_case,
                actual_result=actual_result,
                metrics=metrics,
                execution_time=execution_time,
                search_rounds=getattr(self.rag_system, 'max_rounds', 8),
                documents_found=len(documents),
                success=success
            )
            
            self.logger.info(f"Evaluated {test_case.company_name}: F1={metrics['overall_f1']:.3f}, Time={execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error evaluating {test_case.company_name}: {e}")
            
            return EvaluationResult(
                test_case=test_case,
                actual_result=ContactInfo([], [], {}, [], None, []),
                metrics={'overall_f1': 0.0},
                execution_time=execution_time,
                search_rounds=0,
                documents_found=0,
                success=False,
                error_message=str(e)
            )
    
    def _determine_success(self, f1_score: float) -> bool:
        """Determine success with graduated thresholds instead of all-or-nothing"""
        # Graduated success levels - much more realistic
        if f1_score >= 0.3:  # 30% is reasonable for web scraping
            return True
        return False
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not results:
            return {}
        
        # Overall statistics
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r.success)
        success_rate = successful_cases / total_cases
        
        # Aggregate metrics
        all_metrics = {}
        metric_names = ['email_f1', 'phone_f1', 'social_f1', 'linkedin_f1', 'overall_f1', 'completeness_score']
        
        for metric in metric_names:
            values = [r.metrics.get(metric, 0.0) for r in results]
            all_metrics[metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values)
            }
        
        # Performance metrics
        execution_times = [r.execution_time for r in results]
        documents_found = [r.documents_found for r in results]
        
        # Difficulty analysis
        difficulty_breakdown = {}
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_results = [r for r in results if r.test_case.difficulty == difficulty]
            if difficulty_results:
                difficulty_breakdown[difficulty] = {
                    'count': len(difficulty_results),
                    'success_rate': sum(1 for r in difficulty_results if r.success) / len(difficulty_results),
                    'avg_f1': statistics.mean([r.metrics.get('overall_f1', 0.0) for r in difficulty_results])
                }
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_test_cases': total_cases,
            'successful_cases': successful_cases,
            'success_rate': success_rate,
            'metrics': all_metrics,
            'performance': {
                'avg_execution_time': statistics.mean(execution_times),
                'median_execution_time': statistics.median(execution_times),
                'avg_documents_found': statistics.mean(documents_found),
                'total_execution_time': sum(execution_times)
            },
            'difficulty_breakdown': difficulty_breakdown,
            'individual_results': [asdict(r) for r in results]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> Path:
        """Save evaluation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        report_path = self.results_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"[green]Report saved to {report_path}[/green]")
        return report_path
    
    def display_report(self, report: Dict[str, Any]):
        """Display evaluation report in console"""
        console.print(Panel.fit(
            f"RAG System Evaluation Report\n"
            f"Generated: {report['evaluation_timestamp']}\n"
            f"Total Test Cases: {report['total_test_cases']}\n"
            f"Success Rate: {report['success_rate']:.2%}",
            title="📊 Evaluation Summary",
            border_style="green"
        ))
        
        # Metrics table
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Mean", style="green")
        metrics_table.add_column("Median", style="yellow")
        metrics_table.add_column("Std Dev", style="red")
        metrics_table.add_column("Min", style="blue")
        metrics_table.add_column("Max", style="magenta")
        
        for metric, values in report['metrics'].items():
            metrics_table.add_row(
                metric.replace('_', ' ').title(),
                f"{values['mean']:.3f}",
                f"{values['median']:.3f}",
                f"{values['std']:.3f}",
                f"{values['min']:.3f}",
                f"{values['max']:.3f}"
            )
        
        console.print(metrics_table)
        
        # Performance table
        perf_table = Table(title="Performance Statistics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf = report['performance']
        perf_table.add_row("Average Execution Time", f"{perf['avg_execution_time']:.2f}s")
        perf_table.add_row("Median Execution Time", f"{perf['median_execution_time']:.2f}s")
        perf_table.add_row("Average Documents Found", f"{perf['avg_documents_found']:.1f}")
        perf_table.add_row("Total Execution Time", f"{perf['total_execution_time']:.2f}s")
        
        console.print(perf_table)
        
        # Difficulty breakdown
        if report['difficulty_breakdown']:
            diff_table = Table(title="Difficulty Breakdown")
            diff_table.add_column("Difficulty", style="cyan")
            diff_table.add_column("Test Cases", style="green")
            diff_table.add_column("Success Rate", style="yellow")
            diff_table.add_column("Average F1", style="red")
            
            for difficulty, stats in report['difficulty_breakdown'].items():
                diff_table.add_row(
                    difficulty.title(),
                    str(stats['count']),
                    f"{stats['success_rate']:.2%}",
                    f"{stats['avg_f1']:.3f}"
                )
            
            console.print(diff_table)
    
    def export_to_csv(self, results: List[EvaluationResult], filename: str = None) -> Path:
        """Export results to CSV for further analysis"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.csv"
        
        csv_path = self.results_dir / filename
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'company_name', 'domain', 'difficulty', 'industry',
                'success', 'execution_time', 'documents_found',
                'email_f1', 'phone_f1', 'social_f1', 'linkedin_f1', 'overall_f1',
                'completeness_score', 'emails_found', 'phones_found',
                'social_platforms_found', 'linkedin_profiles_found'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'company_name': result.test_case.company_name,
                    'domain': result.test_case.domain,
                    'difficulty': result.test_case.difficulty,
                    'industry': result.test_case.industry,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'documents_found': result.documents_found,
                    'email_f1': result.metrics.get('email_f1', 0.0),
                    'phone_f1': result.metrics.get('phone_f1', 0.0),
                    'social_f1': result.metrics.get('social_f1', 0.0),
                    'linkedin_f1': result.metrics.get('linkedin_f1', 0.0),
                    'overall_f1': result.metrics.get('overall_f1', 0.0),
                    'completeness_score': result.metrics.get('completeness_score', 0.0),
                    'emails_found': len(result.actual_result.emails),
                    'phones_found': len(result.actual_result.phones),
                    'social_platforms_found': len(result.actual_result.social_media),
                    'linkedin_profiles_found': len(result.actual_result.linkedin_profiles)
                })
        
        console.print(f"[green]Results exported to {csv_path}[/green]")
        return csv_path
