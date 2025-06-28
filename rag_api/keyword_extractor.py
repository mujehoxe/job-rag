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
        Extract keywords and entities from a query

        Args:
            query: User query

        Returns:
            Tuple of (keywords, entities)
        """
        # Extract domain from the query if present
        domain_match = re.search(
            r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", query
        )
        domain = domain_match.group(1) if domain_match else None

        # Simple keyword extraction
        keywords = []
        entities = []

        # Add domain as a keyword and entity if found
        if domain:
            keywords.append(domain)
            entities.append(domain)

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

        # Add general keywords from the query
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word not in [
                "give",
                "find",
                "what",
                "where",
                "when",
                "about",
                "from",
                "with",
                "can",
                "get",
                "all",
                "and",
                "the",
                "for",
                "any",
                "you",
                "your",
                "this",
                "that",
                "these",
                "those",
                "them",
                "they",
                "their",
                "there",
            ]:
                keywords.append(word)

        # Remove duplicates
        keywords = list(set(keywords))
        entities = list(set(entities))

        return keywords, entities
