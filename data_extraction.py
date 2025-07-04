#!/usr/bin/env python3
"""
Advanced Data Extraction for Company Enrichment
"""

import re
import json
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
from rich.console import Console

console = Console()

@dataclass
class ExtractedContact:
    """Structure for extracted contact information"""
    phones: Set[str]
    emails: Set[str]
    social_media: Dict[str, str]
    whatsapp_numbers: Set[str]
    linkedin_profiles: Set[str]
    addresses: Set[str]

@dataclass
class ExtractedPerson:
    """Structure for extracted person information"""
    name: str
    title: str
    linkedin: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class CompanyDataExtractor:
    """Advanced data extraction for company enrichment"""
    
    def __init__(self):
        self.phone_patterns = [
            # International formats
            r'\+\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            # US formats
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            # UK formats
            r'\+44\s?\d{2,4}\s?\d{3,4}\s?\d{3,4}',
            # Generic international
            r'\+\d{2,3}\s?\d{2,4}\s?\d{2,4}\s?\d{2,4}',
        ]
        
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
        
        self.whatsapp_patterns = [
            r'whatsapp\.com/send\?phone=(\+?\d+)',
            r'wa\.me/(\+?\d+)',
            r'api\.whatsapp\.com/send\?phone=(\+?\d+)',
            r'whatsapp.*?(\+?\d{1,4}[-.\s]?\d{1,15})',
            r'WhatsApp.*?(\+?\d{1,4}[-.\s]?\d{1,15})',
        ]
        
        self.social_media_patterns = {
            'facebook': [
                r'facebook\.com/([a-zA-Z0-9.]+)',
                r'fb\.com/([a-zA-Z0-9.]+)',
                r'www\.facebook\.com/([a-zA-Z0-9.]+)',
            ],
            'twitter': [
                r'twitter\.com/([a-zA-Z0-9_]+)',
                r'x\.com/([a-zA-Z0-9_]+)',
            ],
            'instagram': [
                r'instagram\.com/([a-zA-Z0-9_.]+)',
            ],
            'linkedin': [
                r'linkedin\.com/in/([a-zA-Z0-9-]+)',
                r'linkedin\.com/company/([a-zA-Z0-9-]+)',
            ],
            'youtube': [
                r'youtube\.com/(?:c/|channel/|user/)?([a-zA-Z0-9_-]+)',
                r'youtu\.be/([a-zA-Z0-9_-]+)',
            ],
            'tiktok': [
                r'tiktok\.com/@([a-zA-Z0-9_.]+)',
            ]
        }
        
        self.executive_titles = [
            'CEO', 'Chief Executive Officer',
            'CTO', 'Chief Technology Officer',
            'CFO', 'Chief Financial Officer',
            'COO', 'Chief Operating Officer',
            'Founder', 'Co-Founder', 'Co-founder',
            'President', 'Vice President', 'VP',
            'Director', 'Managing Director',
            'Manager', 'General Manager',
            'Owner', 'Principal',
            'Partner', 'Senior Partner'
        ]
        
        # Name patterns for different cultures
        self.name_patterns = [
            # Western names (First Last)
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
            # Western names with middle initial (First M. Last)
            r'\b([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)\b',
            # Names with titles (Mr./Ms./Dr. First Last)
            r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
        ]
    
    def extract_all_data(self, content: str, url: str = "") -> ExtractedContact:
        """Extract all contact information from content"""
        contact = ExtractedContact(
            phones=set(),
            emails=set(),
            social_media={},
            whatsapp_numbers=set(),
            linkedin_profiles=set(),
            addresses=set()
        )
        
        # Extract phone numbers
        contact.phones.update(self._extract_phones(content))
        
        # Extract emails
        contact.emails.update(self._extract_emails(content))
        
        # Extract WhatsApp numbers
        contact.whatsapp_numbers.update(self._extract_whatsapp(content))
        
        # Extract social media
        contact.social_media.update(self._extract_social_media(content))
        
        # Extract LinkedIn profiles separately
        contact.linkedin_profiles.update(self._extract_linkedin_profiles(content))
        
        # Extract addresses
        contact.addresses.update(self._extract_addresses(content))
        
        return contact
    
    def _extract_phones(self, content: str) -> Set[str]:
        """Extract phone numbers"""
        phones = set()
        
        for pattern in self.phone_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_phone(match)
                if self._is_valid_phone(cleaned):
                    phones.add(cleaned)
        
        return phones
    
    def _extract_emails(self, content: str) -> Set[str]:
        """Extract email addresses"""
        emails = set()
        
        for pattern in self.email_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if self._is_valid_email(match):
                    emails.add(match.lower())
        
        return emails
    
    def _extract_whatsapp(self, content: str) -> Set[str]:
        """Extract WhatsApp numbers"""
        whatsapp = set()
        
        for pattern in self.whatsapp_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Handle tuple results from groups
                number = match if isinstance(match, str) else match[0] if match else ""
                cleaned = self._clean_phone(number)
                if cleaned:
                    whatsapp.add(cleaned)
        
        return whatsapp
    
    def _extract_social_media(self, content: str) -> Dict[str, str]:
        """Extract social media profiles"""
        social_media = {}
        
        for platform, patterns in self.social_media_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches and platform not in social_media:
                    username = matches[0]
                    if platform == 'linkedin':
                        if 'company' in pattern:
                            social_media[f'{platform}_company'] = f"https://linkedin.com/company/{username}"
                        else:
                            social_media[f'{platform}_profile'] = f"https://linkedin.com/in/{username}"
                    else:
                        base_url = 'x.com' if platform == 'twitter' else f'{platform}.com'
                        social_media[platform] = f"https://{base_url}/{username}"
        
        return social_media
    
    def _extract_linkedin_profiles(self, content: str) -> Set[str]:
        """Extract LinkedIn profiles specifically"""
        profiles = set()
        
        patterns = [
            r'linkedin\.com/in/([a-zA-Z0-9-]+)',
            r'linkedin\.com/company/([a-zA-Z0-9-]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if 'company' in pattern:
                    profiles.add(f"https://linkedin.com/company/{match}")
                else:
                    profiles.add(f"https://linkedin.com/in/{match}")
        
        return profiles
    
    def _extract_addresses(self, content: str) -> Set[str]:
        """Extract physical addresses"""
        addresses = set()
        
        # Address patterns (basic)
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl)[\s,]+[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}',
            r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}',
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            addresses.update(matches)
        
        return addresses
    
    def extract_key_people(self, content: str) -> List[ExtractedPerson]:
        """Extract key people and their information"""
        people = []
        
        # Look for name + title combinations
        for title in self.executive_titles:
            # Pattern: Name, Title
            pattern1 = rf'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s*{re.escape(title)}'
            matches1 = re.findall(pattern1, content, re.IGNORECASE)
            
            # Pattern: Title: Name
            pattern2 = rf'{re.escape(title)}:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)'
            matches2 = re.findall(pattern2, content, re.IGNORECASE)
            
            # Pattern: Name - Title
            pattern3 = rf'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*-\s*{re.escape(title)}'
            matches3 = re.findall(pattern3, content, re.IGNORECASE)
            
            all_matches = matches1 + matches2 + matches3
            
            for name in all_matches:
                if self._is_valid_name(name):
                    person = ExtractedPerson(name=name.strip(), title=title)
                    
                    # Try to find LinkedIn profile for this person
                    linkedin_pattern = rf'{re.escape(name)}.*?linkedin\.com/in/([a-zA-Z0-9-]+)'
                    linkedin_match = re.search(linkedin_pattern, content, re.IGNORECASE)
                    if linkedin_match:
                        person.linkedin = f"https://linkedin.com/in/{linkedin_match.group(1)}"
                    
                    people.append(person)
        
        return people
    
    def _clean_phone(self, phone: str) -> str:
        """Clean and standardize phone number"""
        if not phone:
            return ""
        
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Ensure it starts with + for international numbers
        if cleaned and not cleaned.startswith('+') and len(cleaned) > 10:
            cleaned = '+' + cleaned
        
        return cleaned
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Validate phone number"""
        if not phone:
            return False
        
        # Remove + for length check
        digits_only = phone.replace('+', '')
        
        # Should be between 7 and 15 digits
        return 7 <= len(digits_only) <= 15 and digits_only.isdigit()
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email address"""
        if not email or '@' not in email:
            return False
        
        # Basic validation
        parts = email.split('@')
        if len(parts) != 2:
            return False
        
        local, domain = parts
        
        # Check for common invalid patterns
        invalid_patterns = [
            'example.com', 'test.com', 'placeholder', 'sample',
            'yourname', 'youremail', 'email.com'
        ]
        
        return not any(pattern in email.lower() for pattern in invalid_patterns)
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate person name"""
        if not name or len(name.split()) < 2:
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            'lorem ipsum', 'john doe', 'jane doe', 'first last',
            'your name', 'full name', 'company name'
        ]
        
        return not any(pattern in name.lower() for pattern in invalid_patterns)
    
    def create_enrichment_summary(self, contact: ExtractedContact, people: List[ExtractedPerson]) -> Dict[str, any]:
        """Create a summary of extracted data"""
        return {
            'contact_summary': {
                'total_phones': len(contact.phones),
                'total_emails': len(contact.emails),
                'total_social_accounts': len(contact.social_media),
                'total_whatsapp': len(contact.whatsapp_numbers),
                'total_linkedin_profiles': len(contact.linkedin_profiles),
                'total_addresses': len(contact.addresses)
            },
            'people_summary': {
                'total_people': len(people),
                'executives': [p for p in people if any(title in p.title.upper() for title in ['CEO', 'CTO', 'CFO', 'COO', 'FOUNDER'])],
                'with_linkedin': [p for p in people if p.linkedin]
            },
            'data_quality': {
                'has_contact_info': len(contact.phones) > 0 or len(contact.emails) > 0,
                'has_social_media': len(contact.social_media) > 0,
                'has_key_people': len(people) > 0,
                'completeness_score': self._calculate_completeness_score(contact, people)
            }
        }
    
    def _calculate_completeness_score(self, contact: ExtractedContact, people: List[ExtractedPerson]) -> float:
        """Calculate completeness score (0-1)"""
        score = 0
        max_score = 6
        
        # Contact information (2 points)
        if contact.phones:
            score += 1
        if contact.emails:
            score += 1
        
        # Social media (2 points)
        if contact.social_media:
            score += 1
        if contact.linkedin_profiles:
            score += 1
        
        # Key people (2 points)
        if people:
            score += 1
        if any(p.linkedin for p in people):
            score += 1
        
        return score / max_score

def extract_company_data(content: str, url: str = "") -> Tuple[ExtractedContact, List[ExtractedPerson], Dict]:
    """Main function to extract company data"""
    extractor = CompanyDataExtractor()
    
    # Extract contact information
    contact = extractor.extract_all_data(content, url)
    
    # Extract key people
    people = extractor.extract_key_people(content)
    
    # Create summary
    summary = extractor.create_enrichment_summary(contact, people)
    
    return contact, people, summary

if __name__ == "__main__":
    # Test with sample content
    sample_content = """
    Contact Milestone Homes Real Estate at (555) 123-4567 or email us at info@milestonehomesre.com
    
    Our CEO John Smith can be reached on LinkedIn at linkedin.com/in/johnsmith-ceo
    Follow us on Facebook: facebook.com/milestonehomes
    WhatsApp: wa.me/15551234567
    
    Management Team:
    - Jane Doe, CFO
    - Mike Johnson - CTO
    """
    
    contact, people, summary = extract_company_data(sample_content)
    
    console.print("📞 Contact Information:")
    console.print(f"  Phones: {list(contact.phones)}")
    console.print(f"  Emails: {list(contact.emails)}")
    console.print(f"  Social Media: {contact.social_media}")
    console.print(f"  WhatsApp: {list(contact.whatsapp_numbers)}")
    
    console.print("\n👥 Key People:")
    for person in people:
        console.print(f"  {person.name} - {person.title}")
    
    console.print(f"\n📊 Summary: {summary}")