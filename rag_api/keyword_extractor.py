#!/usr/bin/env python3
"""
Keyword Extractor class for the RAG API.
"""

import re
from typing import List, Tuple

class KeywordExtractor:
    """Extract keywords and entities from a query using AI"""

    def __init__(self):
        """Initialize the keyword extractor"""
        pass

    def extract_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Extract keywords and entities from a query with reasoning

        Args:
            query: User query

        Returns:
            Tuple of (keywords, entities)
        """
        # Reasoning step: Analyze query intent and structure
        query_lower = query.lower().strip()
        original_query = query.strip()
        
        # Extract domain from the query if present
        domain_match = re.search(
            r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", query
        )
        domain = domain_match.group(1) if domain_match else None

        # Reasoning: Determine if this is a company enrichment query
        contact_indicators = ['contact', 'email', 'phone', 'social', 'linkedin', 'facebook', 
                            'twitter', 'instagram', 'whatsapp', 'about', 'team', 'staff']
        is_contact_query = any(indicator in query_lower for indicator in contact_indicators)
        
        # Reasoning: Determine if this is about finding people/decision makers
        people_indicators = ['ceo', 'founder', 'director', 'manager', 'leadership', 'team', 
                           'decision maker', 'executive', 'president', 'owner']
        is_people_query = any(indicator in query_lower for indicator in people_indicators)
        
        # Reasoning: Determine if this is about social media specifically
        social_indicators = ['social media', 'linkedin', 'facebook', 'twitter', 'instagram', 
                           'youtube', 'tiktok', 'social', 'profiles']
        is_social_query = any(indicator in query_lower for indicator in social_indicators)

        # Initialize keyword and entity lists
        keywords = []
        entities = []

        # CRITICAL FIX: Extract company names from query patterns
        # Pattern: "Find contact information for [COMPANY]"
        # Pattern: "Find contact information and social media for [COMPANY]"
        company_patterns = [
            r"find\s+contact\s+information\s+(?:and\s+\w+\s+)*for\s+(.+?)(?:\s+(?:payment|communication|data|real\s+estate|monitoring|platform|processor|api|service).*)?$",
            r"find\s+.*?\s+for\s+(.+?)(?:\s+(?:payment|communication|data|real\s+estate|monitoring|platform|processor|api|service).*)?$",
            r"(?:get|find|locate|search)\s+.*?\s+(?:for|about|of)\s+(.+?)(?:\s+(?:company|corporation|inc|llc|ltd).*)?$"
        ]
        
        company_name = None
        for pattern in company_patterns:
            match = re.search(pattern, query_lower)
            if match:
                potential_company = match.group(1).strip()
                # Clean up the company name
                potential_company = re.sub(r'\s+(?:payment|communication|data|real\s+estate|monitoring|platform|processor|api|service|company|corporation|inc|llc|ltd).*$', '', potential_company)
                potential_company = potential_company.strip()
                if potential_company and len(potential_company) > 1:
                    company_name = potential_company
                    break
        
        # If no pattern match, try to extract company name from common word positions
        if not company_name:
            words = original_query.split()
            # Look for company name after "for" or at the end
            for i, word in enumerate(words):
                if word.lower() == 'for' and i + 1 < len(words):
                    # Take words after "for" until we hit a descriptor
                    company_words = []
                    for j in range(i + 1, len(words)):
                        if words[j].lower() in ['payment', 'communication', 'data', 'real', 'estate', 'monitoring', 'platform', 'processor', 'api', 'service', 'company', 'corporation']:
                            break
                        company_words.append(words[j])
                    if company_words:
                        company_name = ' '.join(company_words)
                        break

        # Add company name as primary entity
        if company_name:
            entities.append(company_name)
            keywords.append(company_name)
            # Also add individual words of company name
            for word in company_name.split():
                if len(word) > 2:
                    keywords.append(word)

        # Add domain as a keyword and entity if found
        if domain:
            keywords.append(domain)
            entities.append(domain)
            # Extract company name from domain for better searching
            domain_company = domain.split('.')[0]
            if domain_company != domain and domain_company not in keywords:
                keywords.append(domain_company)
                entities.append(domain_company)

        # Enhanced keyword extraction for contact information
        contact_keywords = [
            "contact",
            "email",
            "phone",
            "number",
            "address",
            "location",
            "headquarters",
            "hq",
            "office",
            "social",
            "linkedin",
            "facebook",
            "twitter",
            "instagram",
            "youtube",
            "social media",
            "profile",
            "website",
            "url",
            "handle",
            "username",
            "account",
            "page",
            "staff",
            "team",
            "employee",
            "management",
            "leadership",
            "founder",
            "ceo",
            "director",
            "manager",
            "contact us",
            "about us",
        ]

        query_lower = query.lower()
        for keyword in contact_keywords:
            if keyword in query_lower:
                keywords.append(keyword)

        # Extract other keywords based on common patterns
        if "contact" in query_lower:
            keywords.append("contact information")
        if "email" in query_lower:
            keywords.append("email address")
        if "phone" in query_lower:
            keywords.append("phone number")
        if "social" in query_lower:
            keywords.append("social media")
        if "linkedin" in query_lower:
            keywords.append("LinkedIn")
        if "facebook" in query_lower:
            keywords.append("Facebook")
        if "twitter" in query_lower:
            keywords.append("Twitter")
        if "instagram" in query_lower:
            keywords.append("Instagram")

        # Add contact-specific keywords if this is a contact query
        if is_contact_query:
            contact_terms = ["contact", "email", "phone", "social", "media", "linkedin", "about"]
            for term in contact_terms:
                if term in query_lower and term not in keywords:
                    keywords.append(term)
        
        # Add other meaningful keywords from the query (excluding common stop words and action verbs)
        stop_words = {
            "give", "find", "what", "where", "when", "about", "from", "with", 
            "can", "get", "all", "and", "the", "for", "any", "you", "your", 
            "this", "that", "these", "those", "them", "they", "their", "there",
            "information", "data", "details", "please", "help", "need", "want",
            "show", "tell", "have", "has", "will", "would", "could", "should"
        }
        
        query_words = query_lower.split()
        for word in query_words:
            if (len(word) > 3 and 
                word not in stop_words and 
                word not in keywords and
                not word.isdigit()):
                keywords.append(word)

        # Remove duplicates
        keywords = list(set(keywords))
        entities = list(set(entities))

        return keywords, entities
