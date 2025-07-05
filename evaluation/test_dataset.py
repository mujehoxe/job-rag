#!/usr/bin/env python3
"""
Test Dataset for RAG System Evaluation
Contains sample companies with expected contact information
"""

from evaluation.evaluation_framework import TestCase, ContactInfo

# Sample test cases with expected results
SAMPLE_TEST_CASES = [
    TestCase(
        company_name="OpenAI",
        domain="openai.com",
        query="Find contact information for OpenAI",
        expected_result=ContactInfo(
            emails=["support@openai.com", "press@openai.com"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/OpenAI",
                "linkedin": "https://linkedin.com/company/openai",
                "youtube": "https://youtube.com/c/OpenAI"
            },
            linkedin_profiles=["linkedin.com/company/openai"],
            company_website="https://openai.com",
            key_people=["Sam Altman", "Greg Brockman", "Ilya Sutskever"]
        ),
        difficulty="easy",
        industry="Technology/AI",
        notes="Large tech company with public presence"
    ),
    
    TestCase(
        company_name="HubSpot",
        domain="hubspot.com",
        query="Find contact information and social media for HubSpot",
        expected_result=ContactInfo(
            emails=["info@hubspot.com", "support@hubspot.com"],
            phones=["+1-888-482-7768"],
            social_media={
                "twitter": "https://twitter.com/HubSpot",
                "linkedin": "https://linkedin.com/company/hubspot",
                "facebook": "https://facebook.com/HubSpot",
                "instagram": "https://instagram.com/hubspot",
                "youtube": "https://youtube.com/user/HubSpot"
            },
            linkedin_profiles=["linkedin.com/company/hubspot"],
            company_website="https://hubspot.com",
            key_people=["Yamini Rangan", "Dharmesh Shah", "Brian Halligan"]
        ),
        difficulty="easy",
        industry="Marketing/SaaS",
        notes="Well-known marketing platform with extensive online presence"
    ),
    
    TestCase(
        company_name="Stripe",
        domain="stripe.com",
        query="Find contact information for Stripe payment processor",
        expected_result=ContactInfo(
            emails=["support@stripe.com", "press@stripe.com"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/stripe",
                "linkedin": "https://linkedin.com/company/stripe"
            },
            linkedin_profiles=["linkedin.com/company/stripe"],
            company_website="https://stripe.com",
            key_people=["Patrick Collison", "John Collison"]
        ),
        difficulty="medium",
        industry="FinTech",
        notes="Payment processor with moderate online presence"
    ),
    
    TestCase(
        company_name="Shopify",
        domain="shopify.com",
        query="Find contact information and leadership for Shopify",
        expected_result=ContactInfo(
            emails=["support@shopify.com", "press@shopify.com"],
            phones=["+1-888-746-7439"],
            social_media={
                "twitter": "https://twitter.com/shopify",
                "linkedin": "https://linkedin.com/company/shopify",
                "facebook": "https://facebook.com/shopify",
                "instagram": "https://instagram.com/shopify",
                "youtube": "https://youtube.com/user/shopify"
            },
            linkedin_profiles=["linkedin.com/company/shopify"],
            company_website="https://shopify.com",
            key_people=["Tobi Lütke", "Harley Finkelstein", "Amy Shapero"]
        ),
        difficulty="easy",
        industry="E-commerce",
        notes="Major e-commerce platform with strong social presence"
    ),
    
    TestCase(
        company_name="Zoom",
        domain="zoom.us",
        query="Find contact information for Zoom",
        expected_result=ContactInfo(
            emails=["support@zoom.us", "press@zoom.us"],
            phones=["+1-888-799-9666"],
            social_media={
                "twitter": "https://twitter.com/zoom",
                "linkedin": "https://linkedin.com/company/zoom-video-communications",
                "facebook": "https://facebook.com/zoom",
                "instagram": "https://instagram.com/zoom",
                "youtube": "https://youtube.com/c/ZoomMeetings"
            },
            linkedin_profiles=["linkedin.com/company/zoom-video-communications"],
            company_website="https://zoom.us",
            key_people=["Eric Yuan", "Kelly Steckelberg", "Aparna Bawa"]
        ),
        difficulty="easy",
        industry="Video Communications",
        notes="Popular video conferencing platform"
    ),
    
    TestCase(
        company_name="Atlassian",
        domain="atlassian.com",
        query="Find contact information for Atlassian software company",
        expected_result=ContactInfo(
            emails=["support@atlassian.com", "press@atlassian.com"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/atlassian",
                "linkedin": "https://linkedin.com/company/atlassian",
                "facebook": "https://facebook.com/Atlassian",
                "youtube": "https://youtube.com/c/atlassian"
            },
            linkedin_profiles=["linkedin.com/company/atlassian"],
            company_website="https://atlassian.com",
            key_people=["Mike Cannon-Brookes", "Scott Farquhar", "Cameron Deatsch"]
        ),
        difficulty="medium",
        industry="Software Development Tools",
        notes="Developer tools company with moderate visibility"
    ),
    
    TestCase(
        company_name="Notion",
        domain="notion.so",
        query="Find contact information for Notion productivity app",
        expected_result=ContactInfo(
            emails=["support@notion.so", "press@notion.so"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/NotionHQ",
                "linkedin": "https://linkedin.com/company/notion-so",
                "instagram": "https://instagram.com/notion",
                "youtube": "https://youtube.com/c/notion"
            },
            linkedin_profiles=["linkedin.com/company/notion-so"],
            company_website="https://notion.so",
            key_people=["Ivan Zhao", "Simon Last", "Akshay Kothari"]
        ),
        difficulty="medium",
        industry="Productivity Software",
        notes="Popular productivity app with growing presence"
    ),
    
    TestCase(
        company_name="Figma",
        domain="figma.com",
        query="Find contact information for Figma design tool",
        expected_result=ContactInfo(
            emails=["support@figma.com", "press@figma.com"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/figma",
                "linkedin": "https://linkedin.com/company/figma",
                "instagram": "https://instagram.com/figma",
                "youtube": "https://youtube.com/c/figmadesign"
            },
            linkedin_profiles=["linkedin.com/company/figma"],
            company_website="https://figma.com",
            key_people=["Dylan Field", "Evan Wallace", "Kris Rasmussen"]
        ),
        difficulty="medium",
        industry="Design Software",
        notes="Design tool with strong designer community"
    ),
    
    TestCase(
        company_name="Canva",
        domain="canva.com",
        query="Find contact information for Canva design platform",
        expected_result=ContactInfo(
            emails=["support@canva.com", "press@canva.com"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/canva",
                "linkedin": "https://linkedin.com/company/canva",
                "facebook": "https://facebook.com/canva",
                "instagram": "https://instagram.com/canva",
                "youtube": "https://youtube.com/c/canva"
            },
            linkedin_profiles=["linkedin.com/company/canva"],
            company_website="https://canva.com",
            key_people=["Melanie Perkins", "Cliff Obrecht", "Cameron Adams"]
        ),
        difficulty="easy",
        industry="Design Software",
        notes="Popular design platform with broad consumer appeal"
    ),
    
    TestCase(
        company_name="Airtable",
        domain="airtable.com",
        query="Find contact information for Airtable database platform",
        expected_result=ContactInfo(
            emails=["support@airtable.com", "press@airtable.com"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/airtable",
                "linkedin": "https://linkedin.com/company/airtable",
                "facebook": "https://facebook.com/airtable",
                "youtube": "https://youtube.com/c/airtable"
            },
            linkedin_profiles=["linkedin.com/company/airtable"],
            company_website="https://airtable.com",
            key_people=["Howie Liu", "Andrew Ofstad", "Emmett Nicholas"]
        ),
        difficulty="medium",
        industry="Database/Productivity",
        notes="Database platform with growing business user base"
    ),
    
    # More challenging test cases
    TestCase(
        company_name="Acme Corp",
        domain="acmecorp.example",
        query="Find contact information for Acme Corp",
        expected_result=ContactInfo(
            emails=[],
            phones=[],
            social_media={},
            linkedin_profiles=[],
            company_website=None,
            key_people=[]
        ),
        difficulty="hard",
        industry="Fictional",
        notes="Non-existent company to test error handling"
    ),
    
    TestCase(
        company_name="Discord",
        domain="discord.com",
        query="Find contact information for Discord communication platform",
        expected_result=ContactInfo(
            emails=["support@discord.com", "press@discord.com"],
            phones=[],
            social_media={
                "twitter": "https://twitter.com/discord",
                "linkedin": "https://linkedin.com/company/discord",
                "facebook": "https://facebook.com/discord",
                "instagram": "https://instagram.com/discord",
                "youtube": "https://youtube.com/c/discord"
            },
            linkedin_profiles=["linkedin.com/company/discord"],
            company_website="https://discord.com",
            key_people=["Jason Citron", "Stanislav Vishnevskiy", "Tomasz Marcinkowski"]
        ),
        difficulty="medium",
        industry="Communication/Gaming",
        notes="Gaming-focused communication platform"
    ),
    
    TestCase(
        company_name="Twilio",
        domain="twilio.com",
        query="Find contact information for Twilio communication API",
        expected_result=ContactInfo(
            emails=["support@twilio.com", "press@twilio.com"],
            phones=["+1-415-390-2337"],
            social_media={
                "twitter": "https://twitter.com/twilio",
                "linkedin": "https://linkedin.com/company/twilio-inc-",
                "facebook": "https://facebook.com/Twilio",
                "youtube": "https://youtube.com/user/twilio"
            },
            linkedin_profiles=["linkedin.com/company/twilio-inc-"],
            company_website="https://twilio.com",
            key_people=["Jeff Lawson", "Khozema Shipchandler", "Elena Donio"]
        ),
        difficulty="medium",
        industry="Communication APIs",
        notes="API company with developer focus"
    ),
    
    TestCase(
        company_name="Datadog",
        domain="datadoghq.com",
        query="Find contact information for Datadog monitoring platform",
        expected_result=ContactInfo(
            emails=["support@datadoghq.com", "press@datadoghq.com"],
            phones=["+1-866-329-4466"],
            social_media={
                "twitter": "https://twitter.com/datadoghq",
                "linkedin": "https://linkedin.com/company/datadog",
                "facebook": "https://facebook.com/datadoghq",
                "youtube": "https://youtube.com/c/datadoghq"
            },
            linkedin_profiles=["linkedin.com/company/datadog"],
            company_website="https://datadoghq.com",
            key_people=["Olivier Pomel", "Alexis Lê-Quôc", "David Obstler"]
        ),
        difficulty="medium",
        industry="Monitoring/DevOps",
        notes="Infrastructure monitoring platform"
    ),
    
    TestCase(
        company_name="Snowflake",
        domain="snowflake.com",
        query="Find contact information for Snowflake data platform",
        expected_result=ContactInfo(
            emails=["support@snowflake.com", "press@snowflake.com"],
            phones=["+1-844-766-9355"],
            social_media={
                "twitter": "https://twitter.com/snowflakedb",
                "linkedin": "https://linkedin.com/company/snowflake-computing",
                "facebook": "https://facebook.com/snowflakedb",
                "youtube": "https://youtube.com/c/snowflakedb"
            },
            linkedin_profiles=["linkedin.com/company/snowflake-computing"],
            company_website="https://snowflake.com",
            key_people=["Frank Slootman", "Benoit Dageville", "Thierry Cruanes"]
        ),
        difficulty="medium",
        industry="Data/Cloud",
        notes="Cloud data platform with enterprise focus"
    )
]

# Additional test cases for specific industries
REAL_ESTATE_TEST_CASES = [
    TestCase(
        company_name="Redfin",
        domain="redfin.com",
        query="Find contact information for Redfin real estate",
        expected_result=ContactInfo(
            emails=["support@redfin.com", "press@redfin.com"],
            phones=["+1-844-759-7732"],
            social_media={
                "twitter": "https://twitter.com/redfin",
                "linkedin": "https://linkedin.com/company/redfin",
                "facebook": "https://facebook.com/redfin",
                "instagram": "https://instagram.com/redfin",
                "youtube": "https://youtube.com/c/redfin"
            },
            linkedin_profiles=["linkedin.com/company/redfin"],
            company_website="https://redfin.com",
            key_people=["Glenn Kelman", "Chris Nielsen", "Adam Wiener"]
        ),
        difficulty="easy",
        industry="Real Estate",
        notes="Well-known real estate platform"
    ),
    
    TestCase(
        company_name="Zillow",
        domain="zillow.com",
        query="Find contact information for Zillow real estate platform",
        expected_result=ContactInfo(
            emails=["support@zillow.com", "press@zillow.com"],
            phones=["+1-844-587-7467"],
            social_media={
                "twitter": "https://twitter.com/zillow",
                "linkedin": "https://linkedin.com/company/zillow",
                "facebook": "https://facebook.com/zillow",
                "instagram": "https://instagram.com/zillow",
                "youtube": "https://youtube.com/user/ZillowVideos"
            },
            linkedin_profiles=["linkedin.com/company/zillow"],
            company_website="https://zillow.com",
            key_people=["Rich Barton", "Jeremy Wacksman", "Susan Daimler"]
        ),
        difficulty="easy",
        industry="Real Estate",
        notes="Major real estate platform with strong online presence"
    )
]

# Small/medium business test cases (more challenging)
SMB_TEST_CASES = [
    TestCase(
        company_name="Local Coffee Shop",
        domain="localcoffeeshop.com",
        query="Find contact information for Local Coffee Shop",
        expected_result=ContactInfo(
            emails=["info@localcoffeeshop.com"],
            phones=["+1-555-123-4567"],
            social_media={
                "facebook": "https://facebook.com/localcoffeeshop",
                "instagram": "https://instagram.com/localcoffeeshop"
            },
            linkedin_profiles=[],
            company_website="https://localcoffeeshop.com",
            key_people=["John Smith"]
        ),
        difficulty="hard",
        industry="Food & Beverage",
        notes="Small local business with limited online presence"
    )
]

# Combine all test cases
ALL_TEST_CASES = SAMPLE_TEST_CASES + REAL_ESTATE_TEST_CASES + SMB_TEST_CASES

def get_test_cases_by_difficulty(difficulty: str):
    """Get test cases filtered by difficulty level"""
    return [tc for tc in ALL_TEST_CASES if tc.difficulty == difficulty]

def get_test_cases_by_industry(industry: str):
    """Get test cases filtered by industry"""
    return [tc for tc in ALL_TEST_CASES if industry.lower() in tc.industry.lower()]

def get_sample_test_cases(n: int = 5):
    """Get a sample of n test cases for quick testing"""
    return ALL_TEST_CASES[:n]

if __name__ == "__main__":
    print(f"Total test cases: {len(ALL_TEST_CASES)}")
    print(f"Easy cases: {len(get_test_cases_by_difficulty('easy'))}")
    print(f"Medium cases: {len(get_test_cases_by_difficulty('medium'))}")
    print(f"Hard cases: {len(get_test_cases_by_difficulty('hard'))}")
    print("\nIndustries covered:")
    industries = set(tc.industry for tc in ALL_TEST_CASES)
    for industry in sorted(industries):
        count = len(get_test_cases_by_industry(industry))
        print(f"  {industry}: {count} cases")
